#!/usr/bin/env bash
# Install GPSite into the active ZHMolContact GPU environment.
set -Eeuo pipefail
trap 'printf "\nGPSite installation failed at line %s. Fix the error above, then rerun this script.\n" "$LINENO" >&2' ERR

usage() {
    cat <<'EOF'
Usage: bash scripts/install_gpsite.sh [options]

Activate zhmolcontact first (Python 3.9 / PyTorch 2.0.0 / CUDA 11.8).

Options:
  --prottrans-path DIR  Reuse an extracted ProtT5 model; do not download it.
  --run-example        Run example/demo.fa after installation.
  --gpu ID             GPU for the example (default: 0).
  -h, --help           Show this help.

CUDA_HOME may be set to the CUDA 11.8 toolkit directory.
Default model directory: third_party/GPSite/pretrained_models/
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
log() { printf '\n[GPSite] %s\n' "$*"; }

PROTTRANS_PATH=""
RUN_EXAMPLE=0
GPU=0
while (($#)); do
    case "$1" in
        --prottrans-path)
            (($# >= 2)) || die "--prottrans-path requires a directory."
            [[ -n "$2" && "$2" != --* ]] || die "Invalid model directory."
            PROTTRANS_PATH="$2"; shift 2 ;;
        --run-example) RUN_EXAMPLE=1; shift ;;
        --gpu)
            (($# >= 2)) || die "--gpu requires an ID."
            [[ "$2" =~ ^[0-9]+$ ]] || die "--gpu must be a nonnegative integer."
            GPU="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "Unknown option: $1" ;;
    esac
done

# Resolve the script location, so all installation paths are independent of cwd.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
GPSITE_DIR="$PROJECT_ROOT/third_party/GPSite"
if [[ -n "$PROTTRANS_PATH" ]]; then
    [[ -d "$PROTTRANS_PATH" ]] || die "Model directory does not exist: $PROTTRANS_PATH"
    PROTTRANS_PATH="$(cd -- "$PROTTRANS_PATH" && pwd -P)"
fi

log "Checking the active Python environment and build tools"
for tool in python git wget; do
    command -v "$tool" >/dev/null || die "Missing command: $tool"
done
command -v "${CXX:-c++}" >/dev/null || die "Install a C++ compiler, or set CXX."
python -m pip --version

log "Preparing GPSite source code"
mkdir -p "$PROJECT_ROOT/third_party"
if [[ -e "$GPSITE_DIR" ]]; then
    [[ -f "$GPSITE_DIR/script/predict.py" ]] || die "Existing GPSite directory is incomplete: $GPSITE_DIR"
    log "Reusing $GPSITE_DIR"
else
    git clone https://github.com/biomed-AI/GPSite.git "$GPSITE_DIR"
fi
cd -- "$GPSITE_DIR"
[[ -d model ]] || die "GPSite model/ directory is missing."
[[ -f script/feature_extraction/mkdssp ]] || die "GPSite mkdssp executable is missing."

log "Recording installed package constraints"
python - <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

required = (
    "torch torch-geometric torch-scatter torch-sparse torch-cluster "
    "numpy scipy pandas biopython"
).split()
optional = "torchvision torchaudio fair-esm huggingface-hub".split()
versions = {}
for name in required + optional:
    try:
        versions[name] = version(name)
    except PackageNotFoundError:
        if name in required:
            raise SystemExit(f"Missing {name}; complete the ZHMolContact GPU installation first.")
if "fair-esm" in versions and versions["fair-esm"] != "2.0.0":
    raise SystemExit("Installed fair-esm must be 2.0.0 for these GPSite dependencies.")
Path("gpsite-constraints.txt").write_text(
    "".join(f"{name}=={value}\n" for name, value in versions.items()), encoding="utf-8"
)
print(Path("gpsite-constraints.txt").resolve())
PY

log "Installing pinned ESMFold and ProtTrans dependencies"
python -m pip install -c gpsite-constraints.txt \
    "setuptools==77.0.1" wheel ninja packaging psutil py-cpuinfo

DS_BUILD_OPS=0 python -m pip install --no-build-isolation -c gpsite-constraints.txt \
    "fair-esm==2.0.0" "deepspeed==0.9.5" \
    "dm-tree==0.1.8" "pytorch-lightning==1.9.5" "torchmetrics==0.11.4" \
    "omegaconf==2.3.0" "ml-collections==0.1.1" "einops==0.6.1" \
    "transformers==4.30.1" "sentencepiece==0.1.99" "modelcif==0.7" \
    "dllogger @ git+https://github.com/NVIDIA/dllogger.git" \
    --extra-index-url https://download.pytorch.org/whl/cu118

python -m pip install --no-build-isolation --no-deps \
    "openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307"

log "Applying the OpenFold / DeepSpeed compatibility patch"
python - <<'PY'
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("openfold")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("Cannot locate installed OpenFold.")
count = 0
for root in spec.submodule_search_locations:
    for path in Path(root).rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        updated = source.replace(
            "deepspeed.utils.is_initialized()", "deepspeed.comm.comm.is_initialized()"
        )
        if updated == source:
            continue
        compile(updated, str(path), "exec")
        backup = path.with_name(path.name + ".gpsite.bak")
        if not backup.exists():
            backup.write_text(source, encoding="utf-8")
        path.write_text(updated, encoding="utf-8")
        count += 1
        print("Updated:", path)
print(f"OpenFold patch complete ({count} files updated; existing patches are reused).")
PY

log "Checking dependencies and imports before model downloads"
python -m pip check
python - <<'PY'
from importlib.metadata import version
from pathlib import Path

for line in Path("gpsite-constraints.txt").read_text().splitlines():
    name, expected = line.split("==", 1)
    if version(name) != expected:
        raise SystemExit(f"Unexpected package version change: {name}")
import torch_geometric
import torch_scatter
import torch_sparse
import torch_cluster
import dllogger
import modelcif
import sentencepiece
from transformers import T5EncoderModel, T5Tokenizer
from esm.esmfold.v1.esmfold import ESMFold
from openfold.utils.kernel.attention_core import attention_core
print("Package versions and GPSite dependency imports passed.")
PY

log "Preparing ProtT5 weights and configuring GPSite"
python - "$GPSITE_DIR" "$PROTTRANS_PATH" <<'PY'
import ast
import hashlib
from pathlib import Path
import subprocess
import sys
import zipfile

gpsite = Path(sys.argv[1])
custom = sys.argv[2]
pretrained = gpsite / "pretrained_models"
extracting = pretrained / ".gpsite-extracting"

def complete_model(path):
    if not all((path / name).is_file() and (path / name).stat().st_size > 0
               for name in ("config.json", "spiece.model")):
        return False
    return any(p.is_file() and p.stat().st_size > 0 for pattern in (
        "pytorch_model*.bin", "model*.safetensors"
    ) for p in path.glob(pattern))

def locate_model():
    default = pretrained / "prot_t5_xl_uniref50"
    if complete_model(default):
        return default
    candidates = sorted({p.parent for p in pretrained.rglob("config.json")
                         if complete_model(p.parent)}) if pretrained.exists() else []
    if len(candidates) > 1:
        raise SystemExit("Multiple model directories found; select one with --prottrans-path.")
    return candidates[0] if candidates else None

if custom:
    model = Path(custom)
    if not complete_model(model):
        raise SystemExit("--prottrans-path must contain config.json, spiece.model and model weights.")
else:
    # An interrupted extraction must be repeated, even if a partial weight
    # file is already present.
    model = None if extracting.exists() else locate_model()
    if model is None:
        pretrained.mkdir(parents=True, exist_ok=True)
        archive = pretrained / "prot_t5_xl_uniref50.zip"
        subprocess.run([
            "wget", "-c",
            "https://zenodo.org/records/4644188/files/prot_t5_xl_uniref50.zip?download=1",
            "-O", str(archive),
        ], check=True)
        checksum = hashlib.md5()
        with archive.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                checksum.update(block)
        if checksum.hexdigest() != "ab11a7eddfbaff5784effd41380b482a":
            raise SystemExit(f"MD5 mismatch: {archive}. Move/remove the damaged ZIP and rerun.")
        print("ProtT5 checksum passed; extracting...", flush=True)
        extracting.touch()
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(pretrained)
        model = locate_model()
        if model is None:
            raise SystemExit("No complete ProtT5 model found after extraction.")
        extracting.unlink()

predict = gpsite / "script/predict.py"
source = predict.read_bytes()
tree = ast.parse(source, filename=str(predict))
assignments = [node for node in ast.walk(tree)
               if isinstance(node, (ast.Assign, ast.AnnAssign))
               and any(isinstance(t, ast.Name) and t.id == "ProtTrans_path"
                       for t in (node.targets if isinstance(node, ast.Assign) else [node.target]))]
if len(assignments) != 1 or assignments[0].value is None:
    raise SystemExit("Expected exactly one ProtTrans_path assignment in GPSite script/predict.py.")
value = assignments[0].value
lines = source.splitlines(keepends=True)
start = sum(map(len, lines[:value.lineno - 1])) + value.col_offset
end = sum(map(len, lines[:value.end_lineno - 1])) + value.end_col_offset
updated = source[:start] + repr(str(model.resolve())).encode("utf-8") + source[end:]
compile(updated, str(predict), "exec")
if updated != source:
    backup = predict.with_name(predict.name + ".gpsite.bak")
    if not backup.exists():
        backup.write_bytes(source)
    predict.write_bytes(updated)
print("ProtTrans_path:", model.resolve())
PY

chmod +x script/feature_extraction/mkdssp
if ((RUN_EXAMPLE)); then
    log "Running example/demo.fa on GPU $GPU (first run downloads ESM weights)"
    python script/predict.py -i example/demo.fa -o example/ --gpu "$GPU"
    log "Example finished: $GPSITE_DIR/example/demo/pred/"
fi
log "GPSite installation complete: $GPSITE_DIR"
if ((!RUN_EXAMPLE)); then
    printf 'To verify prediction, run from the project root:\n'
    printf '(cd third_party/GPSite && python script/predict.py -i example/demo.fa -o example/ --gpu %s)\n' "$GPU"
fi
