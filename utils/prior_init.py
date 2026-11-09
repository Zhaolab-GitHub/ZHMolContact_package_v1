"""Load binding-site/domain priors and build RNA-protein pair features.

When protein FASTAs are supplied, ``prior_init`` prepares/predicts domains
under ``prot_site_dir/features/protein/<protein_id>/`` by default. Existing
outputs are reused, and all intermediates remain in that protein's folder.
When interface-distance CSVs are available, domain labels are also built
and saved there; ``dataset_path`` can specify the dataset's root directory.
Supplying both FASTA lists also builds the notebook's four inter-residue
features under ``prot_site_dir/features/inter_residue/``.
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
    """Build domain labels from one-based interface distances, with cutoff 4 A.

    A domain is positive if any residue has distance <= cutoff. Domain IDs
    may start at 0 or 1, or be nonconsecutive; labels follow their sorted
    order. Residue numbers refer to FASTA positions (1 -> mask index 0).
    Return a CPU torch.long tensor. These are supervised targets for the
    specific RNA-protein pair described by the table.
    """
    import torch

    required = {"residue_number", "nucleotide_number", "distance"}
    if not isinstance(interface_dist_df, pd.DataFrame):
        raise TypeError("interface_dist_df must be a pandas DataFrame.")
    if not required.issubset(interface_dist_df.columns):
        raise ValueError(f"interface_dist_df must contain columns {sorted(required)}")
    if not np.isfinite(cutoff) or cutoff < 0:
        raise ValueError("cutoff must be a finite nonnegative distance.")
    mask = torch.as_tensor(prot_mask_tensor, device="cpu")
    if (mask.ndim != 1 or not mask.numel() or mask.is_complex()
            or not torch.equal(mask, mask.long()) or (mask < 0).any()):
        raise ValueError("A domain mask must contain one nonnegative integer ID per residue.")
    _, domain_index = torch.unique(mask.long(), sorted=True, return_inverse=True)
    residues = pd.to_numeric(interface_dist_df["residue_number"], errors="raise").to_numpy(dtype=float)
    nucleotides = pd.to_numeric(interface_dist_df["nucleotide_number"], errors="raise").to_numpy(dtype=float)
    distances = pd.to_numeric(interface_dist_df["distance"], errors="raise").to_numpy(dtype=float)
    if (not np.isfinite(residues).all() or not np.isfinite(distances).all()
            or (residues != np.floor(residues)).any() or (distances < 0).any()
            or (residues < 1).any() or (residues > len(mask)).any()
            or not np.isfinite(nucleotides).all() or (nucleotides < 1).any()
            or (nucleotides != np.floor(nucleotides)).any()):
        raise ValueError("Invalid interface distances or one-based residue/nucleotide numbers.")
    labels = torch.zeros(len(torch.unique(mask)), dtype=torch.long)
    contacts = residues[distances <= cutoff].astype(np.int64) - 1
    if len(contacts):
        labels[domain_index[torch.from_numpy(contacts)]] = 1
    return labels


def _domain_interface_source(pdbcode, prot_fasta_file, prot_site_dir,
                             dataset_path=None, explicit=None):
    """Resolve a supplied distance table or the dataset's revNo CSV."""
    if isinstance(explicit, pd.DataFrame):
        return explicit
    if explicit is not None:
        source = Path(explicit).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"Interface distance CSV not found: {source}")
        return source
    filename = f"{pdbcode}.pdb.distance_revNo.csv"
    if dataset_path is not None:
        root = Path(dataset_path).expanduser()
        candidates = [root / pdbcode / filename, root / filename]
    else:
        candidates = [Path(prot_fasta_file).expanduser().parent / filename,
                      prot_site_dir / filename, prot_site_dir / pdbcode / filename]
    for source in candidates:
        if source.is_file():
            return source
    if dataset_path is not None:
        raise FileNotFoundError(f"Interface distance CSV not found; checked: {candidates}")
    return None


_PROT_DONOR = dict.fromkeys("ACDEFGHIKLMNPQRSTVWYX", 0.0)
_PROT_DONOR.update(R=1.2, K=1.0, H=0.7, W=0.4, N=0.8, Q=0.8,
                   S=0.6, T=0.6, Y=0.5, C=0.2)
_PROT_ACCEPTOR = dict.fromkeys("ACDEFGHIKLMNPQRSTVWYX", 0.0)
_PROT_ACCEPTOR.update(D=1.1, E=1.1, N=0.8, Q=0.8, S=0.7, T=0.7,
                      Y=0.6, H=0.5, C=0.1)
_RNA_DONOR = dict(G=1.1, C=0.9, A=0.8, U=0.8, X=0.8)
_RNA_ACCEPTOR = dict(C=1.1, G=1.0, A=0.9, U=0.8, X=0.8)
_PROT_ASA_MAX = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "E": 223.0, "Q": 225.0, "G": 104.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
    "X": 200.0,
}
_PAIR_CHANNELS = {
    "elec_edge_feat": 3, "hbond_edge_feat": 4,
    "pi_pi_edge_feat": 12, "sasa_edge_feat": 6,
}


def _pair_sequence(fasta_file, rna=False):
    sequence = str(SeqIO.read(str(Path(fasta_file).expanduser()), "fasta").seq).upper()
    if rna:
        sequence = sequence.replace("T", "U")
    allowed = _RNA_DONOR if rna else _PROT_DONOR
    invalid = sorted(set(sequence) - set(allowed))
    if not sequence or invalid:
        raise ValueError(f"Empty or invalid {'RNA' if rna else 'protein'} sequence in {fasta_file}: {invalid}")
    return sequence


def build_elec_edge_feat(rna_seq, prot_seq):
    """Electrostatic one-hot channels: attraction, neutral, repulsion."""
    charges = {aa: 0 for aa in _PROT_DONOR}
    charges.update(K=1, R=1, H=1, D=-1, E=-1)
    interaction = -np.array([charges[aa] for aa in prot_seq])[None, :]
    return np.broadcast_to(
        np.stack([interaction == -1, interaction == 0, interaction == 1], axis=-1),
        (len(rna_seq), len(prot_seq), 3),
    ).astype(np.float32, copy=True)


def get_hbond_propensity_with_water(rna_seq, prot_seq, alpha=0.3, beta=0.2):
    """Keep the notebook's direct, cooperative and gated water-bridge terms."""
    rd = np.array([_RNA_DONOR[x] for x in rna_seq], dtype=float)
    ra = np.array([_RNA_ACCEPTOR[x] for x in rna_seq], dtype=float)
    pdonor = np.array([_PROT_DONOR[x] for x in prot_seq], dtype=float)
    pa = np.array([_PROT_ACCEPTOR[x] for x in prot_seq], dtype=float)
    da, ad = np.outer(rd, pa), np.outer(ra, pdonor)
    direct = da + ad
    cooperative = alpha * np.sqrt(da * ad)
    water = beta * 0.5 * (np.outer(rd, pdonor) + np.outer(ra, pa)) / (1.0 + direct)
    return da, ad, water, np.tanh(direct + cooperative + water)


def read_rna_ss_file(ss_file, seq_length=None):
    """Read dot-bracket notation, including pseudoknots and optional energy."""
    candidates = []
    for line in Path(ss_file).expanduser().read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and set(fields[0]) <= set(".()[]{}<>"):
            candidates.append(fields[0])
    if seq_length is not None:
        candidates = [ss for ss in candidates if len(ss) == seq_length]
    if not candidates:
        raise ValueError(f"No RNA dot-bracket structure of length {seq_length} in {ss_file}")
    return candidates[0]


def _pair_expand(rna_feat, prot_feat):
    """Concatenate per-RNA and per-protein channels on every residue pair."""
    h, w = len(rna_feat), len(prot_feat)
    return np.concatenate([
        np.broadcast_to(rna_feat[:, None, :], (h, w, rna_feat.shape[1])),
        np.broadcast_to(prot_feat[None, :, :], (h, w, prot_feat.shape[1])),
    ], axis=-1)


def _ss_aromatic_feature(rna_ss, prot_seq):
    # Nine bracket symbols + unpaired/paired flags = 11 RNA channels.
    vocab = ".()[]{}<>"
    onehot = np.eye(len(vocab), dtype=np.float32)[[vocab.index(x) for x in rna_ss]]
    unpaired = np.array([x == "." for x in rna_ss], dtype=np.float32)[:, None]
    rna_feat = np.concatenate([onehot, unpaired, 1.0 - unpaired], axis=1)
    aromatic = np.array([aa in "FYWH" for aa in prot_seq], dtype=np.float32)[:, None]
    return _pair_expand(rna_feat, aromatic)  # 12 channels, not the notebook comment's 6.


def calc_sasa_df(pdb_file, seq, skip_unknown=False):
    """Calculate monomer SASA, preserving PDB residue order and checking FASTA."""
    try:
        import freesasa
    except ImportError as exc:
        raise ImportError("Inter-residue SASA features require freesasa==2.2.1.") from exc
    from Bio.SeqUtils import seq1

    structure = freesasa.Structure(str(Path(pdb_file).expanduser()), None,
                                  {"skip-unknown": skip_unknown})
    if not structure.nAtoms():
        raise ValueError(f"No atoms available for SASA in {pdb_file}")
    # Follow first appearance in the actual structure: string sorting would
    # put residue 10 before residue 2 and misalign SASA with the sequence.
    residues = {}
    for i in range(structure.nAtoms()):
        key = (structure.chainLabel(i), structure.residueNumber(i).strip())
        residues.setdefault(key, structure.residueName(i).strip().upper())
    if len(residues) != len(seq):
        raise ValueError(f"{pdb_file}: {len(residues)} PDB residues but {len(seq)} FASTA residues.")
    rna_names = {"A": "A", "C": "C", "G": "G", "U": "U", "T": "U", "X": "X",
                 "ADE": "A", "CYT": "C", "GUA": "G", "URA": "U", "URI": "U"}
    areas = freesasa.calc(structure).residueAreas()
    rows = []
    for i, ((chain, number), name) in enumerate(residues.items()):
        letter = rna_names.get(name, seq1(name, custom_map={"UNK": "X"}))
        if seq[i] != "X" and letter != seq[i]:
            raise ValueError(f"{pdb_file}: PDB/FASTA mismatch at position {i + 1}: {name} vs {seq[i]}.")
        area = areas[chain][number]
        rows.append({"number": i, "res": seq[i], "total_sasa": area.total,
                     "polar_sasa": area.polar, "apolar_sasa": area.apolar})
    return pd.DataFrame(rows)


def _sasa_pair_feature(rna_pdb_file, prot_pdb_file, rna_seq, prot_seq):
    rna = calc_sasa_df(rna_pdb_file, rna_seq)
    prot = calc_sasa_df(prot_pdb_file, prot_seq)
    columns = ["total_sasa", "polar_sasa", "apolar_sasa"]
    # Preserve the trained notebook's scaling: all three RNA channels use
    # the maximum TOTAL SASA of that base type within this RNA, not separate maxima.
    denominator = rna.groupby("res")["total_sasa"].transform("max").to_numpy()[:, None]
    rna_values = rna[columns].to_numpy(dtype=float)
    rna_values = np.divide(rna_values, denominator, out=np.zeros_like(rna_values),
                           where=denominator > 0)
    prot_values = prot[columns].to_numpy(dtype=float) / np.array(
        [_PROT_ASA_MAX[aa] for aa in prot_seq], dtype=float,
    )[:, None]
    return _pair_expand(rna_values, prot_values).astype(np.float32)


def _checked_pair_features(features, rna_length, prot_length):
    if not isinstance(features, dict) or set(features) != set(_PAIR_CHANNELS):
        raise ValueError("Inter-residue cache must contain the four named feature arrays.")
    for name, channels in _PAIR_CHANNELS.items():
        value = features[name]
        if not isinstance(value, np.ndarray) or value.shape != (rna_length, prot_length, channels):
            raise ValueError(f"{name}: expected shape {(rna_length, prot_length, channels)}.")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values.")
    return features


def build_inter_residue_features(
    rna_fasta_file, prot_fasta_file, rna_pdb_file, prot_pdb_file, rna_ss_file,
    output_file=None,
):
    """Return the notebook's four NumPy arrays, optionally caching a .pt file.

    Channels are electrostatics (3), hydrogen-bond propensity (4), RNA
    dot-bracket/aromaticity (12), and monomer SASA (6). No distance labels
    or native RNA-protein contacts are used. FASTAs must match the residues
    present in each PDB, in file order. Cached files are reused after shape
    and finiteness checks; remove a cache to rebuild after changing inputs.
    """
    import torch

    rna_seq = _pair_sequence(rna_fasta_file, rna=True)
    prot_seq = _pair_sequence(prot_fasta_file)
    output_file = Path(output_file).expanduser() if output_file is not None else None
    if output_file is not None and output_file.is_file():
        # These local caches contain NumPy arrays, rather than tensors only.
        return _checked_pair_features(
            torch.load(output_file, map_location="cpu", weights_only=False),
            len(rna_seq), len(prot_seq),
        )
    for source in (rna_pdb_file, prot_pdb_file, rna_ss_file):
        if not Path(source).expanduser().is_file():
            raise FileNotFoundError(f"Inter-residue feature input not found: {source}")
    rna_ss = read_rna_ss_file(rna_ss_file, len(rna_seq))
    features = {
        "elec_edge_feat": build_elec_edge_feat(rna_seq, prot_seq),
        "hbond_edge_feat": np.stack(get_hbond_propensity_with_water(rna_seq, prot_seq), axis=-1),
        "pi_pi_edge_feat": _ss_aromatic_feature(rna_ss, prot_seq),
        "sasa_edge_feat": _sasa_pair_feature(rna_pdb_file, prot_pdb_file, rna_seq, prot_seq),
    }
    _checked_pair_features(features, len(rna_seq), len(prot_seq))
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(features, output_file)
    return features


def prior_init(
    RNA_path, prot_path,
    RNA_fa_files=None, prot_fa_files=None, pdbcodes=None,
    rna_site_dir=None, prot_site_dir=None, dataset=None,
    gpsite_dir=None, gpu="0",
    *, domain_dir=None, dombpred_dir=None, sdsl_file=None,
    psipred_program=None, jackhmmer_program="jackhmmer",
    esmfold_model=None, domain_device=None, domain_chunk_size=64,
    predict_domains=True,
    RNA_pdb_files=None, prot_pdb_files=None, RNA_ss_files=None,
    inter_residue_dir=None, build_inter_residue=True,
    interface_dist_files=None, dataset_path=None, domain_label_cutoff=4.0,
):
    """Return RNA/protein binding-site arrays, each with shape (length, 1).

    Existing prediction files are reused. Generating a missing RNA prior uses
    the single-complex graph pair accepted by ``predict_binding_sites``.
    dataset is retained for compatibility with existing calls.

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

    Domain labels are built when a corresponding interface-distance CSV is
    found. Pass interface_dist_files (paths or DataFrames, one per complex),
    or dataset_path containing <pdbcode>/<pdbcode>.pdb.distance_revNo.csv.
    Otherwise the CSV is searched beside each protein FASTA and under
    prot_site_dir. Explicit missing inputs raise an error; when no source
    was supplied or discovered, domain masks are returned without labels.
    Labels use distance <= domain_label_cutoff (default 4.0 A), are returned
    as prot_domain_lbl and saved beside the mask as <protein_id>_domain_lbl.pt.
    When several complexes share a protein ID in this call, their labels
    instead use <pdbcode>_domain_lbl.pt to keep RNA-specific targets separate.
    Labels are rebuilt from the current CSV and cutoff on each call.

    Supplying both FASTA lists also enables inter-residue features by default.
    Each result then includes elec_edge_feat (H,W,3), hbond_edge_feat (H,W,4),
    pi_pi_edge_feat (H,W,12) and sasa_edge_feat (H,W,6), as NumPy arrays.
    PDB paths default to each FASTA's sibling .pdb file. RNA SS is discovered
    in rna_site_dir/features/RNA/<pdbid>_<RNAchain>/<pdbid>_<RNAchain>_RNA.ss,
    the corresponding prot_site_dir or FASTA-parent features/RNA folder,
    or beside the RNA FASTA with suffix .ss. Supply the three *_files lists
    explicitly when inputs live elsewhere. Cache files are written to
    prot_site_dir/features/inter_residue/<pdbcode>_inter_residue_feat.pt;
    inter_residue_dir overrides this folder. Set build_inter_residue=False
    to disable this stage. Changing inputs requires deleting their cache.
    """
    if pdbcodes is None or len(pdbcodes) == 0:
        raise ValueError("pdbcodes must contain at least one complex ID.")
    if rna_site_dir is None or prot_site_dir is None:
        raise ValueError("rna_site_dir and prot_site_dir are required.")
    if prot_fa_files is not None and len(prot_fa_files) != len(pdbcodes):
        raise ValueError("prot_fa_files and pdbcodes must have the same length.")
    for name, files in (("RNA_fa_files", RNA_fa_files), ("RNA_pdb_files", RNA_pdb_files),
                        ("prot_pdb_files", prot_pdb_files), ("RNA_ss_files", RNA_ss_files),
                        ("interface_dist_files", interface_dist_files)):
        if files is not None and len(files) != len(pdbcodes):
            raise ValueError(f"{name} and pdbcodes must have the same length.")
    inter_enabled = build_inter_residue and (
        (RNA_fa_files is not None and prot_fa_files is not None)
        or any(value is not None for value in
               (RNA_pdb_files, prot_pdb_files, RNA_ss_files, inter_residue_dir))
    )
    if inter_enabled and (RNA_fa_files is None or prot_fa_files is None):
        raise ValueError("RNA_fa_files and prot_fa_files are required for inter-residue features.")
    domains_enabled = predict_domains and (prot_fa_files is not None or domain_dir is not None)
    if domains_enabled and prot_fa_files is None:
        raise ValueError("prot_fa_files is required when domain_dir enables DomBpred.")
    if (interface_dist_files is not None or dataset_path is not None) and not domains_enabled:
        raise ValueError("Domain labels require prot_fa_files and predict_domains=True.")
    if not np.isfinite(domain_label_cutoff) or domain_label_cutoff < 0:
        raise ValueError("domain_label_cutoff must be a finite nonnegative distance.")

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
        
        
    #============================== domain prediction ==============================
    # Run domains after all site predictions so a cached ESMFold model does not
    # occupy GPU memory during RNA/GPSite inference for subsequent complexes.
    if domains_enabled:
        import torch

        domain_dir = Path(domain_dir).expanduser().resolve()
        device = domain_device
        if device is None:
            device = f"cuda:{gpu}" if gpu is not None and str(gpu) and torch.cuda.is_available() else "cpu"
        model_cache = {"model": esmfold_model}
        protein_counts = {}
        for pdbcode in pdbcodes:
            protein_id = f"{pdbcode}_protein"
            protein_counts[protein_id] = protein_counts.get(protein_id, 0) + 1
        try:
            for idx, pdbcode in enumerate(tqdm(pdbcodes, desc="DomBpred")):
                protein_id = f"{pdbcode}_protein"
                output_dir = domain_dir / protein_id
                source = _domain_interface_source(
                    pdbcode, prot_fa_files[idx], prot_site_dir,
                    dataset_path=dataset_path,
                    explicit=interface_dist_files[idx] if interface_dist_files is not None else None,
                )
                interface_dist_df = (source if isinstance(source, pd.DataFrame)
                                     else pd.read_csv(source) if source is not None else None)
                mask = predict_protein_domains(
                    prot_fa_files[idx], output_dir, protein_id,
                    dombpred_dir=dombpred_dir, sdsl_file=sdsl_file,
                    psipred_program=psipred_program, jackhmmer_program=jackhmmer_program,
                    esmfold_model=esmfold_model, device=device, chunk_size=domain_chunk_size,
                    _model_cache=model_cache,
                )
                if len(mask) != len(result_dict[pdbcode]["prot_pred_site"]):
                    raise ValueError(f"{pdbcode}: protein-site and domain-mask lengths differ.")
                result_dict[pdbcode]["protein_domain_mask"] = mask
                if interface_dist_df is not None:
                    labels = build_domain_label(mask, interface_dist_df, cutoff=domain_label_cutoff)
                    if (pd.to_numeric(interface_dist_df["nucleotide_number"], errors="raise")
                            > len(result_dict[pdbcode]["RNA_pred_site"])).any():
                        raise ValueError(f"{pdbcode}: nucleotide numbers exceed the RNA-site length.")
                    # A protein mask is shared across RNAs; supervised labels
                    # belong to each RNA-protein pair and must not overwrite one another.
                    label_id = pdbcode if protein_counts[protein_id] > 1 else protein_id
                    output_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(labels, output_dir / f"{label_id}_domain_lbl.pt")
                    result_dict[pdbcode]["protein_domain_lbl"] = labels
        finally:
            if esmfold_model is None:
                model_cache.pop("model", None)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    #============================== inter edge feature ==============================
    if inter_enabled:
        feature_dir = (Path(inter_residue_dir).expanduser() if inter_residue_dir is not None
                       else prot_site_dir / "features" / "inter_residue")
        for idx, pdbcode in enumerate(tqdm(pdbcodes, desc="Inter-residue features")):
            rna_fasta = Path(RNA_fa_files[idx]).expanduser()
            prot_fasta = Path(prot_fa_files[idx]).expanduser()
            rna_id = f"{pdbcode}_RNA"
            ss_candidates = [
                root / "features" / "RNA" / rna_id / f"{rna_id}.ss"
                for root in (rna_site_dir, prot_site_dir, rna_fasta.parent)
            ] + [rna_fasta.with_suffix(".ss")]
            ss_file = (RNA_ss_files[idx] if RNA_ss_files is not None else
                       next((p for p in ss_candidates if p.is_file()), ss_candidates[0]))
            features = build_inter_residue_features(
                rna_fasta, prot_fasta,
                RNA_pdb_files[idx] if RNA_pdb_files is not None else rna_fasta.with_suffix(".pdb"),
                prot_pdb_files[idx] if prot_pdb_files is not None else prot_fasta.with_suffix(".pdb"),
                ss_file, output_file=feature_dir / f"{pdbcode}_inter_residue_feat.pt",
            )
            if (features["elec_edge_feat"].shape[:2] !=
                    (len(result_dict[pdbcode]["RNA_pred_site"]),
                     len(result_dict[pdbcode]["prot_pred_site"]))):
                raise ValueError(f"{pdbcode}: site-prior and inter-residue feature lengths differ.")
            result_dict[pdbcode].update(features)

    return result_dict
