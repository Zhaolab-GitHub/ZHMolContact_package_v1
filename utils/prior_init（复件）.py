"""Load binding-site priors and optionally prepare/predict protein domains.

When protein FASTAs are supplied, ``prior_init`` prepares/predicts domains
under ``prot_site_dir/features/protein/<protein_id>/`` by default. Existing
outputs are reused, and all intermediates remain in that protein's folder.
"""

from pathlib import Path
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm.auto import tqdm

from utils.ZHMolSite_predict import predict_binding_sites


def _site_array(frame, column, source):
    values = frame[[column]].to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all() or (
        (values < 0) | (values > 1)
    ).any():
        raise ValueError(f"Invalid binding-site scores in {source}")
    return values


def predict_protein_sites(prot_fasta_file, output_file, gpsite_dir=None, gpu="0"):
    """Run GPSite and retain its outputs beside the cached result table.

    Each FASTA must contain one protein with 1-2000 standard amino acids.
    ``gpsite_dir`` is the checkout containing ``script/predict.py``.
    Use ``gpu=None`` for CPU inference.
    Each attempt gets a permanent directory under ``output_file.parent``;
    intermediate files and ``gpsite.log`` are retained even on failure.
    """
    if gpsite_dir is None:
        gpsite_dir = Path(__file__).resolve().parents[1] / "third_party" / "GPSite"
    gpsite_dir = Path(gpsite_dir).expanduser().resolve()
    script = gpsite_dir / "script" / "predict.py"
    if not script.is_file():
        raise FileNotFoundError(f"GPSite script not found: {script}")

    record = SeqIO.read(str(Path(prot_fasta_file).expanduser()), "fasta")
    sequence = str(record.seq).upper()
    if not 1 <= len(sequence) <= 2000:
        raise ValueError("GPSite requires a protein sequence of 1-2000 residues.")
    if set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
        raise ValueError("GPSite would remove nonstandard amino acids; check the FASTA.")

    output_file = Path(output_file).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # mkdtemp creates a persistent folder; each attempt starts with fresh inputs.
    work = Path(tempfile.mkdtemp(
        prefix=f"{output_file.stem}_gpsite_", dir=str(output_file.parent),
    ))
    print(f"GPSite output directory: {work}", flush=True)
    fasta = work / "input.fa"
    # A fixed internal ID makes the result independent of the FASTA header.
    fasta.write_text(f">protein\n{sequence}\n", encoding="utf-8")
    command = [sys.executable, str(script), "-i", str(fasta), "-o", str(work)]
    if gpu is not None and str(gpu) != "":
        command += ["--gpu", str(gpu)]
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    log_file = work / "gpsite.log"
    with log_file.open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            command, cwd=str(gpsite_dir), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
    if returncode:
        print(f"GPSite failed; see {log_file}", flush=True)
        raise subprocess.CalledProcessError(returncode, command)

    prediction = work / "input" / "pred" / "protein.txt"
    frame = pd.read_csv(prediction, sep="\t")
    if "AA" not in frame or "".join(frame["AA"].astype(str)) != sequence:
        raise ValueError("GPSite output residues do not match the input protein.")
    _site_array(frame, "RNA_binding", prediction)
    frame.to_csv(output_file, sep="\t", index=False)
    return frame


def _protein_sequence(fasta_file):
    record = SeqIO.read(str(Path(fasta_file).expanduser()), "fasta")
    sequence = str(record.seq).upper()
    if not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
        raise ValueError(f"Expected one protein with standard amino acids: {fasta_file}")
    return sequence


def _dombpred_script(dombpred_dir=None):
    """Locate the Python 3.9 source inside the project's third_party folder."""
    if dombpred_dir is not None:
        script = Path(dombpred_dir).expanduser().resolve() / "DomBpred.py"
        if not script.is_file():
            raise FileNotFoundError(f"Python 3.9 DomBpred source not found: {script}")
        return script
    third_party = Path(__file__).resolve().parents[1] / "third_party"
    for name in ("DomBpred_python39", "DomBpred-main", "DomBpred", ""):
        script = third_party / name / "DomBpred.py"
        if script.is_file():
            return script
    matches = list(third_party.glob("*/DomBpred.py"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Cannot locate a unique DomBpred.py under {third_party}; set dombpred_dir."
    )


def _tool_path(program):
    program = os.path.expanduser(str(program))
    resolved = shutil.which(program)
    if resolved is None:
        raise FileNotFoundError(f"Executable not found: {program}")
    return Path(resolved).resolve()


def _run_domain_tool(command, cwd, log_file):
    """Use argument lists, retain tool logs, and stop on external failures."""
    print(f"Running {Path(command[0]).name}; log: {log_file}", flush=True)
    with Path(log_file).open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [str(value) for value in command], cwd=str(cwd),
            stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"{Path(command[0]).name} failed (exit {result.returncode}); see {log_file}"
        )


def _load_esmfold(device, chunk_size):
    import esm

    model = esm.pretrained.esmfold_v1().eval().to(device)
    model.set_chunk_size(chunk_size)
    if str(device).startswith("cpu"):
        model.esm.float()
    return model


def _distogram_to_distance(logits):
    """Convert ESMFold logits to an expected-distance matrix in Angstroms.

    Use the OpenFold/AlphaFold 64-bin convention: 63 boundaries from 2.3125
    to 21.6875 A. Finite bins use midpoints; the two open tails are represented
    by the adjacent boundary +/- half a bin (2.15625 and 21.84375 A). This is
    an explicit distance approximation, not the trRosetta 37-bin conversion.
    Binning reference: aqlaboratory/openfold, openfold/utils/loss.py,
    distogram_loss(min_bin=2.3125, max_bin=21.6875, no_bins=64).
    """
    if hasattr(logits, "detach"):
        logits = logits.detach().float().cpu().numpy()
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim == 4 and logits.shape[0] == 1:
        logits = logits[0]
    if logits.ndim != 3 or logits.shape[0] != logits.shape[1] or logits.shape[2] != 64:
        raise ValueError(f"Expected ESMFold logits of shape (L, L, 64), got {logits.shape}")
    if not np.isfinite(logits).all():
        raise ValueError("ESMFold returned non-finite distance logits.")
    probabilities = logits - logits.max(axis=-1, keepdims=True)
    np.exp(probabilities, out=probabilities)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    edges = np.linspace(2.3125, 21.6875, 63, dtype=np.float32)
    width = edges[1] - edges[0]
    centers = np.r_[edges[0] - width / 2, (edges[:-1] + edges[1:]) / 2,
                    edges[-1] + width / 2].astype(np.float32)
    distance = probabilities @ centers
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0)
    return distance


def _check_ss2(path, sequence):
    rows = [line.split() for line in Path(path).read_text().splitlines()]
    rows = [row for row in rows if len(row) == 6 and row[0].isdigit()]
    if ([int(row[0]) for row in rows] != list(range(1, len(sequence) + 1))
            or "".join(row[1] for row in rows) != sequence
            or any(row[2] not in "CHE" for row in rows)):
        raise ValueError(f"PSIPRED residues do not match the protein: {path}")


def _check_distance(path, length):
    with np.load(path, allow_pickle=False) as archive:
        distance = archive["dist"]
    if distance.shape == (length, length, 1):
        distance = distance[..., 0]
    if (distance.shape != (length, length) or not np.isfinite(distance).all()
            or (distance < 0).any()):
        raise ValueError(f"Invalid DomBpred distance matrix: {path}")
    if (distance[~np.eye(length, dtype=bool)] <= 0).any():
        raise ValueError(f"Off-diagonal distances must be positive: {path}")
    return distance


def prepare_dombpred_inputs(
    prot_fasta_file, output_dir, protein_id,
    dombpred_dir=None, sdsl_file=None, psipred_program=None,
    jackhmmer_program="jackhmmer", esmfold_model=None,
    device="cuda:0", chunk_size=64, _model_cache=None,
):
    """Prepare/reuse the four input files required by DomBpred.

    ``output_dir`` is the folder for one protein, e.g. ``protein/8H1B_A``.
    SDSL defaults to ``dombpred_dir/data/SDSL.txt``. PSIPRED defaults to
    ``third_party/psipred.4.02/psipred/runpsipred_single`` relative to the
    project root. The database and executables must already be installed.
    Supply a loaded ``esmfold_model``
    to reuse it. PSIPRED runs in its own permanent folder, so no files in the
    caller's working directory are moved or deleted.
    """
    import torch

    if not protein_id or protein_id in (".", "..") or any(c in protein_id for c in "/\\"):
        raise ValueError("protein_id must be a filename-safe string.")
    sequence = _protein_sequence(prot_fasta_file)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fasta = output_dir / f"{protein_id}.fasta"
    if fasta.is_file() and _protein_sequence(fasta) != sequence:
        raise ValueError(f"Cached protein sequence differs from the input: {fasta}")
    # A normalized single-line FASTA also avoids DomBpred's multiline parser limit.
    if fasta.resolve() == Path(prot_fasta_file).expanduser().resolve():
        fasta = output_dir / "dombpred_input.fasta"
    fasta.write_text(f">{protein_id}\n{sequence}\n", encoding="utf-8")
    msa = output_dir / f"{protein_id}.txt"
    ss2 = output_dir / f"{protein_id}_protein.ss2"
    dist = output_dir / f"{protein_id}.npz"

    if not msa.is_file():
        if sdsl_file is None:
            sdsl_file = _dombpred_script(dombpred_dir).parent / "data" / "SDSL.txt"
        database = Path(sdsl_file).expanduser().resolve()
        if not database.is_file():
            raise FileNotFoundError(f"SDSL database not found: {database}; set sdsl_file.")
        pending = output_dir / f"{protein_id}.pending.txt"
        pending.unlink(missing_ok=True)
        _run_domain_tool(
            [_tool_path(jackhmmer_program), "--notextw", "-A", pending, fasta, database],
            output_dir, output_dir / "jackhmmer.log",
        )
        if not pending.is_file() or not pending.stat().st_size:
            raise RuntimeError("jackhmmer did not produce a nonempty alignment.")
        pending.replace(msa)
    if not msa.stat().st_size:
        raise ValueError(f"Empty cached MSA: {msa}")

    if not ss2.is_file():
        if psipred_program is None:
            psipred_program = (
                Path(__file__).resolve().parents[1] / "third_party"
                / "psipred.4.02" / "psipred" / "runpsipred_single"
            )
        program = _tool_path(psipred_program)
        work = Path(tempfile.mkdtemp(prefix="psipred_", dir=str(output_dir)))
        (work / "input.fa").write_text(f">{protein_id}\n{sequence}\n", encoding="utf-8")
        # The stock script refers to ./bin and ./data relative to its cwd.
        for name in ("bin", "data"):
            resource = program.parent / name
            if resource.is_dir():
                (work / name).symlink_to(resource, target_is_directory=True)
        _run_domain_tool([program, "input.fa"], work, work / "psipred.log")
        prediction = work / "input.ss2"
        if not prediction.is_file():
            raise FileNotFoundError(f"PSIPRED output missing: {prediction}")
        _check_ss2(prediction, sequence)
        shutil.copy2(prediction, ss2)
    _check_ss2(ss2, sequence)

    if not dist.is_file():
        owned_model = esmfold_model is None and _model_cache is None
        if esmfold_model is not None:
            model = esmfold_model
        elif _model_cache is not None:
            if _model_cache.get("model") is None:
                _model_cache["model"] = _load_esmfold(device, chunk_size)
            model = _model_cache["model"]
        else:
            model = _load_esmfold(device, chunk_size)
        try:
            with torch.no_grad():
                prediction = model.infer(sequence)
            distance = _distogram_to_distance(prediction["distogram_logits"])
            if distance.shape != (len(sequence), len(sequence)):
                raise ValueError("ESMFold distance dimensions do not match the protein.")
            pending = output_dir / f"{protein_id}.pending.npz"
            np.savez_compressed(pending, dist=distance)
            pending.replace(dist)
        finally:
            if owned_model:
                del model
    _check_distance(dist, len(sequence))
    return {"seq_path": fasta, "msa_path": msa, "ss2_path": ss2, "dist_path": dist}


def make_domain_mask(line, seq_length=None):
    """Parse DomBpred ranges; return (name, zero-based mask list, ranges).

    Discontinuous pieces separated by commas belong to the same domain.
    Every residue must be assigned exactly once, including residues in domain 0.
    """
    parts = line.strip().split()
    if len(parts) != 3:
        raise ValueError("Expected one DomBpred result: name, domain count, ranges.")
    name, count, ranges = parts
    domains = [value for value in ranges.split(";") if value]
    if int(count) < 1 or int(count) != len(domains):
        raise ValueError(f"{name}: domain count does not match the predicted ranges.")
    domain_ranges = []
    for domain in domains:
        segments = []
        for segment in domain.split(","):
            match = re.fullmatch(r"(\d+)-(\d+)", segment)
            if match is None:
                raise ValueError(f"{name}: invalid domain segment {segment!r}")
            start, end = map(int, match.groups())
            if not 1 <= start <= end:
                raise ValueError(f"{name}: invalid domain interval {segment!r}")
            segments.append((start, end))
        domain_ranges.append(segments)
    length = max(end for segments in domain_ranges for _, end in segments)
    if seq_length is not None and length != seq_length:
        raise ValueError(f"{name}: domain ranges end at {length}, expected {seq_length}.")
    mask = np.full(length, -1, dtype=np.int64)
    for domain_id, segments in enumerate(domain_ranges):
        for start, end in segments:
            if (mask[start - 1:end] != -1).any():
                raise ValueError(f"{name}: overlapping domain assignments at {start}-{end}.")
            mask[start - 1:end] = domain_id
    if (mask < 0).any():
        raise ValueError(f"{name}: some residues have no domain assignment.")
    return name, mask.tolist(), domain_ranges


def _checked_domain_tensor(mask, length):
    import torch

    mask = torch.as_tensor(mask, device="cpu")
    if mask.ndim != 1 or mask.numel() != length or not torch.equal(mask, mask.long()):
        raise ValueError("A domain mask must contain one integer ID per protein residue.")
    mask = mask.long()
    ids = torch.unique(mask, sorted=True)
    if not torch.equal(ids, torch.arange(len(ids), dtype=torch.long)):
        raise ValueError("Domain IDs must be consecutive integers starting from 0.")
    return mask


def predict_protein_domains(
    prot_fasta_file, output_dir, protein_id,
    dombpred_dir=None, sdsl_file=None, psipred_program=None,
    jackhmmer_program="jackhmmer", esmfold_model=None,
    device="cuda:0", chunk_size=64, _model_cache=None,
):
    """Prepare missing inputs, run DomBpred.py, cache and return a CPU long mask."""
    import torch

    if not protein_id or protein_id in (".", "..") or any(c in protein_id for c in "/\\"):
        raise ValueError("protein_id must be a filename-safe string.")
    sequence = _protein_sequence(prot_fasta_file)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cached_fasta = output_dir / f"{protein_id}.fasta"
    if cached_fasta.is_file() and _protein_sequence(cached_fasta) != sequence:
        raise ValueError(f"Cached protein sequence differs from the input: {cached_fasta}")
    result_file = output_dir / f"{protein_id}_DomBpred.txt"
    mask_file = output_dir / f"{protein_id}_domain_mask.pt"
    if not result_file.is_file() and mask_file.is_file():
        return _checked_domain_tensor(
            torch.load(mask_file, map_location="cpu", weights_only=True), len(sequence),
        )
    if not result_file.is_file():
        script = _dombpred_script(dombpred_dir)
        inputs = prepare_dombpred_inputs(
            prot_fasta_file, output_dir, protein_id,
            dombpred_dir=script.parent, sdsl_file=sdsl_file,
            psipred_program=psipred_program, jackhmmer_program=jackhmmer_program,
            esmfold_model=esmfold_model, device=device, chunk_size=chunk_size,
            _model_cache=_model_cache,
        )
        # DomBpred appends to -log. A fresh path prevents concatenated predictions.
        work = Path(tempfile.mkdtemp(prefix="dombpred_", dir=str(output_dir)))
        prediction = work / "result.txt"
        command = [sys.executable, script]
        command += [f"-{key}={value}" for key, value in inputs.items()]
        command += [f"-log={prediction}"]
        _run_domain_tool(command, script.parent, work / "dombpred.log")
        make_domain_mask(prediction.read_text(), len(sequence))
        shutil.copy2(prediction, result_file)
    _, mask, _ = make_domain_mask(result_file.read_text(), len(sequence))
    mask = _checked_domain_tensor(mask, len(sequence))
    torch.save(mask, mask_file)
    return mask


def build_domain_label(prot_mask_tensor, interface_dist_df, cutoff=4.0):
    """Optional supervised labels from native interface distances (not inference).

    Label a domain 1 if any of its residues has a distance <= cutoff. Residue
    numbers in the table are one-based; domain IDs in the mask are zero-based.
    This helper is never called by ``prior_init``.
    """
    import torch

    required = {"residue_number", "nucleotide_number", "distance"}
    if not isinstance(interface_dist_df, pd.DataFrame) or not required.issubset(interface_dist_df):
        raise ValueError(f"interface_dist_df must contain columns {sorted(required)}")
    if not np.isfinite(cutoff) or cutoff < 0:
        raise ValueError("cutoff must be a finite nonnegative distance.")
    mask = _checked_domain_tensor(prot_mask_tensor, len(prot_mask_tensor))
    residues = pd.to_numeric(interface_dist_df["residue_number"], errors="raise").to_numpy()
    distances = pd.to_numeric(interface_dist_df["distance"], errors="raise").to_numpy()
    if (not np.isfinite(residues).all() or not np.isfinite(distances).all()
            or (residues != np.floor(residues)).any() or (distances < 0).any()
            or (residues < 1).any() or (residues > len(mask)).any()):
        raise ValueError("Invalid interface distances or one-based protein residue numbers.")
    labels = torch.zeros(len(torch.unique(mask)), dtype=torch.long)
    contacts = residues[distances <= cutoff].astype(np.int64) - 1
    if len(contacts):
        labels[mask[torch.from_numpy(contacts)]] = 1
    return labels


def prior_init(
    RNA_path, prot_path,
    RNA_fa_files=None, prot_fa_files=None, pdbcodes=None,
    rna_site_dir=None, prot_site_dir=None, dataset=None,
    gpsite_dir=None, gpu="0",
    *, domain_dir=None, dombpred_dir=None, sdsl_file=None,
    psipred_program=None, jackhmmer_program="jackhmmer",
    esmfold_model=None, domain_device=None, domain_chunk_size=64,
    predict_domains=True,
):
    """Return RNA/protein binding-site arrays, each with shape (length, 1).

    Existing prediction files are reused. Generating a missing RNA prior uses
    the single-complex graph pair accepted by ``predict_binding_sites``.
    RNA_fa_files and dataset are retained for compatibility with existing calls.

    Supplying prot_fa_files enables domains by default. All domain outputs go
    to prot_site_dir/features/protein/<pdbid>_<protein_chain>/. For example,
    prot_site_dir='/example/8PTZ_X_C/' produces the folder
    '/example/8PTZ_X_C/features/protein/8PTZ_C/'. An explicit domain_dir
    overrides only the protein-feature root. Set predict_domains=False for
    sites only. Cached site-only calls without protein FASTAs remain valid.
    Domain-enabled results also contain ``prot_domain_mask``: a zero-based
    CPU torch.long tensor of shape (protein_length,). ESMFold is loaded once
    only if a distance is missing; pass esmfold_model to reuse a loaded model.
    Default resources are third_party/DomBpred_python39/data/SDSL.txt and
    third_party/psipred.4.02/psipred/runpsipred_single. These paths are resolved
    from the project root, independently of the caller's working directory.
    """
    if pdbcodes is None or len(pdbcodes) == 0:
        raise ValueError("pdbcodes must contain at least one complex ID.")
    if rna_site_dir is None or prot_site_dir is None:
        raise ValueError("rna_site_dir and prot_site_dir are required.")
    if prot_fa_files is not None and len(prot_fa_files) != len(pdbcodes):
        raise ValueError("prot_fa_files and pdbcodes must have the same length.")
    domains_enabled = predict_domains and (prot_fa_files is not None or domain_dir is not None)
    if domains_enabled and prot_fa_files is None:
        raise ValueError("prot_fa_files is required when domain_dir enables DomBpred.")

    rna_site_dir = Path(rna_site_dir).expanduser()
    prot_site_dir = Path(prot_site_dir).expanduser()
    if domains_enabled and domain_dir is None:
        domain_dir = prot_site_dir / "features" / "protein"
    rna_site_dir.mkdir(parents=True, exist_ok=True)
    prot_site_dir.mkdir(parents=True, exist_ok=True)
    result_dict = {}

    for idx, pdbcode in enumerate(tqdm(pdbcodes)):
        if not isinstance(pdbcode, str) or not pdbcode or pdbcode in (".", "..") or any(
            separator in pdbcode for separator in ("/", "\\")
        ):
            raise ValueError("Each complex ID must be a nonempty filename-safe string.")
        print(pdbcode)

        #============================== site prediction ==============================
        rna_file = rna_site_dir / f"{pdbcode}_rna_pred_sites.csv"
        if rna_file.is_file():
            rna_frame = pd.read_csv(rna_file)
        else:
            if len(pdbcodes) != 1:
                raise ValueError("Generate missing RNA priors one complex/graph pair at a time.")
            rna_frame = predict_binding_sites(RNA_path=RNA_path, protein_path=prot_path)
            _site_array(rna_frame, "pred_score", rna_file)
            rna_frame.to_csv(rna_file, index=False)

        prot_file = prot_site_dir / f"{pdbcode}_protein.txt"
        if prot_file.is_file():
            prot_frame = pd.read_csv(prot_file, sep="\t")
        else:
            if prot_fa_files is None:
                raise ValueError(f"A protein FASTA is required to generate {prot_file}")
            print(f"Running GPSite for {pdbcode}...")
            prot_frame = predict_protein_sites(
                prot_fa_files[idx], prot_file, gpsite_dir=gpsite_dir, gpu=gpu,
            )

        result_dict[pdbcode] = {
            "RNA_pred_site": _site_array(rna_frame, "pred_score", rna_file),
            "prot_pred_site": _site_array(prot_frame, "RNA_binding", prot_file),
        }
        
        
    #============================== site prediction ==============================
    # Run domains after all site predictions so a cached ESMFold model does not
    # occupy GPU memory during RNA/GPSite inference for subsequent complexes.
    if domains_enabled:
        import torch

        domain_dir = Path(domain_dir).expanduser().resolve()
        device = domain_device
        if device is None:
            device = f"cuda:{gpu}" if gpu is not None and str(gpu) and torch.cuda.is_available() else "cpu"
        model_cache = {"model": esmfold_model}
        try:
            for idx, pdbcode in enumerate(tqdm(pdbcodes, desc="DomBpred")):
                parts = pdbcode.split("_")
                if len(parts) != 3 or not all(parts):
                    raise ValueError("Domain prediction expects IDs like pdbid_RNAchain_proteinchain.")
                protein_id = f"{parts[0]}_{parts[2]}"
                output_dir = domain_dir / protein_id
                mask = predict_protein_domains(
                    prot_fa_files[idx], output_dir, protein_id,
                    dombpred_dir=dombpred_dir, sdsl_file=sdsl_file,
                    psipred_program=psipred_program, jackhmmer_program=jackhmmer_program,
                    esmfold_model=esmfold_model, device=device, chunk_size=domain_chunk_size,
                    _model_cache=model_cache,
                )
                if len(mask) != len(result_dict[pdbcode]["prot_pred_site"]):
                    raise ValueError(f"{pdbcode}: protein-site and domain-mask lengths differ.")
                result_dict[pdbcode]["prot_domain_mask"] = mask
        finally:
            if esmfold_model is None:
                model_cache.pop("model", None)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    #============================== inter edge feature ==============================
    

    return result_dict
