#!/usr/bin/env bash
# Install native dependencies for the bundled Python 3.9 DomBpred code.
set -Eeuo pipefail
trap 'printf "\nDomBpred installation failed at line %s. Fix the error above, then rerun this script.\n" "$LINENO" >&2' ERR

usage() {
    cat <<'EOF'
Usage: bash scripts/install_dombpred.sh [--skip-apt]

Install HMMER, tcsh and build tools, build PSIPRED 4.02, and download SDSL.
The project must already contain third_party/DomBpred_python39/DomBpred.py.
Existing PSIPRED sources, valid source archives and nonempty SDSL are reused.

Options:
  --skip-apt  Skip apt-get when all required system tools are already installed.
  -h, --help  Show this help.
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
log() { printf '\n[DomBpred] %s\n' "$*"; }

SKIP_APT=0
while (($#)); do
    case "$1" in
        --skip-apt) SKIP_APT=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "Unknown option: $1" ;;
    esac
done

# Resolve paths from the script, so they do not depend on the caller's cwd.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
THIRD_PARTY="$PROJECT_ROOT/third_party"
DOMBPRED_DIR="$THIRD_PARTY/DomBpred_python39"
PSIPRED_ROOT="$THIRD_PARTY/psipred.4.02"
PSIPRED_DIR="$PSIPRED_ROOT/psipred"
PSIPRED_ARCHIVE="$THIRD_PARTY/psipred.4.02.tar.gz"
SDSL_FILE="$DOMBPRED_DIR/data/SDSL.txt"

[[ -f "$DOMBPRED_DIR/DomBpred.py" ]] || die \
    "Missing $DOMBPRED_DIR/DomBpred.py. This installer requires the bundled DomBpred code."

if ((!SKIP_APT)); then
    command -v apt-get >/dev/null || die "apt-get is required; this installer targets Ubuntu."
    APT=(apt-get)
    if ((EUID != 0)); then
        command -v sudo >/dev/null || die "sudo is required, or install system tools separately and use --skip-apt."
        APT=(sudo apt-get)
    fi
    log "Installing HMMER, tcsh and build tools (sudo may request your password)"
    "${APT[@]}" update
    "${APT[@]}" install -y hmmer tcsh build-essential wget
fi

log "Checking required system tools"
for tool in wget tar make cc c++ tcsh hmmsearch hmmscan hmmpress; do
    command -v "$tool" >/dev/null || die \
        "Missing command: $tool. Install the system dependencies before using --skip-apt."
done
mkdir -p "$PSIPRED_ROOT" "$DOMBPRED_DIR/data"

if [[ ! -f "$PSIPRED_DIR/runpsipred_single" || ! -f "$PSIPRED_DIR/src/Makefile" ]]; then
    if [[ ! -s "$PSIPRED_ARCHIVE" ]] || ! tar -tzf "$PSIPRED_ARCHIVE" >/dev/null 2>&1; then
        log "Downloading PSIPRED 4.02 (resuming an existing .part download)"
        wget -c https://bioinf.cs.ucl.ac.uk/downloads/psipred/psipred.4.02.tar.gz \
            -O "$PSIPRED_ARCHIVE.part"
        tar -tzf "$PSIPRED_ARCHIVE.part" >/dev/null 2>&1 || die \
            "Invalid PSIPRED archive: $PSIPRED_ARCHIVE.part. Move/remove it and rerun."
        mv -- "$PSIPRED_ARCHIVE.part" "$PSIPRED_ARCHIVE"
    else
        log "Reusing the PSIPRED source archive"
    fi
    log "Extracting PSIPRED"
    tar -xzf "$PSIPRED_ARCHIVE" -C "$PSIPRED_ROOT"
else
    log "Reusing PSIPRED sources: $PSIPRED_DIR"
fi
[[ -f "$PSIPRED_DIR/runpsipred_single" && -f "$PSIPRED_DIR/src/Makefile" ]] || die \
    "PSIPRED extraction did not create the expected psipred/runpsipred_single and psipred/src/Makefile."

log "Building and installing PSIPRED"
make -C "$PSIPRED_DIR/src"
make -C "$PSIPRED_DIR/src" install
for executable in \
    "$PSIPRED_DIR/runpsipred_single" \
    "$PSIPRED_DIR/bin/seq2mtx" \
    "$PSIPRED_DIR/bin/psipred" \
    "$PSIPRED_DIR/bin/psipass2"; do
    [[ -s "$executable" ]] || die "Missing or empty PSIPRED executable: $executable"
    chmod +x "$executable"
    [[ -x "$executable" ]] || die "Cannot execute $executable"
done

if [[ -s "$SDSL_FILE" ]]; then
    log "Reusing SDSL: $SDSL_FILE"
else
    log "Downloading SDSL (resuming an existing .part download)"
    wget -c https://zhanglab-bioinf.com/DomBpred/materials/SDSL/SDSL.txt \
        -O "$SDSL_FILE.part"
    [[ -s "$SDSL_FILE.part" ]] || die "Downloaded SDSL file is empty: $SDSL_FILE.part"
    mv -- "$SDSL_FILE.part" "$SDSL_FILE"
fi

log "DomBpred dependencies are ready."
printf 'PSIPRED: %s\nSDSL: %s\n' "$PSIPRED_DIR" "$SDSL_FILE"
