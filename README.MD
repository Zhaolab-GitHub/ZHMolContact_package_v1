# ZHMolContact

## Introduction

Accurately predicting RNA-protein interfacial contacts is essential for understanding molecular recognition and modeling complex structures. Yet this task is challenging because native contacts are sparse among possible RNA-protein residue pairs, especially when reliable intermolecular coevolutionary signals and complex templates are unavailable. Here, we present ZHMolContact, a deep learning framework that combines geometric learning with hierarchical priors to predict RNA-protein contacts without explicit multiple sequence alignments or complex-template inputs. We evaluated ZHMolContact on the public FoldBench benchmark. Across the benchmark, ZHMolContact outperformed AI-based methods, including AlphaFold3, as well as statistical physics-based direct coupling analysis baselines. Incorporating the predicted contacts into structural modeling workflows further improved complex-model accuracy. These findings highlight the value of combining hierarchical priors with geometric learning to predict RNA-protein contacts and guide complex modeling.

## Installation

### System Requirements

ZHMolContact currently supports Ubuntu Linux with an NVIDIA GPU and an NVIDIA driver compatible with CUDA 11.8 (recommended). Anaconda or Miniconda is recommended for environment management. ZHMolContact was tested on Ubuntu 20.04 with a 128GB-cpu or an NVIDIA GeForce RTX 4090 GPU.

### Installation with NIVDIA GPU (CUDA 11.8 was used in the authors’ environment)

Download or clone this repository, navigate to the project root directory, and create a dedicated Conda environment:

```bash
git clone https://github.com/Zhaolab-GitHub/ZHMolContact_package_v1.git
cd ZHMolContact_package_v1
conda create -n zhmolcontact python=3.9 -y
conda activate zhmolcontact
```

Install ZHMolContact and its CUDA 11.8-compatible dependencies:

```bash
python -m pip install ".[notebook, cu118]" \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  --find-links https://data.pyg.org/whl/torch-2.0.0+cu118.html \
  --find-links ./wheels \
  --prefer-binary
```

### Verify the Installation

Check the installed PyTorch version, CUDA, and selected device:

```bash
python - <<'PY'
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Selected device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

For the CUDA 11.8 installation, the expected output includes:

```text
CUDA version: 11.8
CUDA available: True
Selected device: cuda:0
GPU: NVIDIA GeForce RTX 4090
```

### GPSite Installation

Use the existing `zhmolcontact` GPU environment (Python 3.9, PyTorch 2.0.0,
CUDA 11.8). Git, a C++ compiler, `wget`, and the CUDA 11.8 toolkit with `nvcc`
are required. Run from the project root:

```bash
conda activate zhmolcontact
bash scripts/install_gpsite.sh
```

The script installs GPSite in `third_party/GPSite`, preserves installed core
package versions, applies the OpenFold compatibility patch, downloads and
checks the ProtT5 weights, and configures `ProtTrans_path` automatically.
Existing source code and extracted weights are reused. If needed, set
`CUDA_HOME` to your CUDA 11.8 toolkit directory before running the script.

To reuse an existing extracted ProtT5 model instead of downloading it:

```bash
bash scripts/install_gpsite.sh --prottrans-path /absolute/path/to/prot_t5_xl_uniref50
```

To install and run the example on GPU 0:

```bash
bash scripts/install_gpsite.sh --run-example --gpu 0
```

Example results are saved under `third_party/GPSite/example/demo/pred/`.
ESM2/ESMFold weights download on the first prediction; allow approximately
13 GB of model downloads, plus space for extraction. Run
`bash scripts/install_gpsite.sh --help` for available options. Rerun the
installer if OpenFold is reinstalled to reapply its compatibility patch.

### DomBpred Installation

On Ubuntu, run from the **project root directory**:

```bash
bash scripts/install_dombpred.sh
```

The script requires the bundled `third_party/DomBpred_python39/DomBpred.py`.
It installs HMMER, tcsh and build tools through `apt-get` (using `sudo` when
needed), builds PSIPRED 4.02, and downloads SDSL. Existing PSIPRED sources and
nonempty SDSL files are reused; interrupted downloads can be resumed. If the
system dependencies are already installed, use `--skip-apt`.

`prior_init.py` automatically uses these paths. ESMFold shares the GPSite
environment installed above. Domain outputs are saved to
`prot_site_dir/features/protein/<pdbid>_<protein_chain>/`.


## Code and Data

### Data Files

The following directories contain the datasets, examples, and trained model required to evaluate and run ZHMolContact:

- `Data/`: Data required to reproduce the evaluation results on the FoldBench test set.
- `trained_model/`: Trained ZHMolContact model weights.

### Evaluation Notebooks

The following Jupyter notebooks are provided for evaluating ZHMolContact on the benchmark datasets:

- `FoldBench_evaluate.ipynb`: Evaluates ZHMolContact on the FoldBench test set.

## Predicting Protein-Binding Nucleotides

ZHMolContact predicts RNA-protein residue-nucleotide contacts using RNA and protein sequences together with their predicted structures.

### Prediction with Input Sequences and Structures

Run the prediction script from the root directory of the ZHMolSite project and specify a configuration file using `--config`:

```bash
# Example using an RNA-protein complex
python ZHMolContact_predict.py --config example/8PTZ_X_C/config.json

# Example using an aminoacyl-tRNA synthetase (aaRS)-tRNA complex
python ZHMolContact_predict.py --config example/aaRS_tRNA/config.json
```


## Contact

For questions, comments, or suggestions regarding ZHMolContact, please contact:

- Yunjie Zhao: [yjzhaowh@ccnu.edu.cn](mailto:yjzhaowh@ccnu.edu.cn)
- Haoquan Liu: [liuhaoquan@mails.ccnu.edu.cn](mailto:liuhaoquan@mails.ccnu.edu.cn)

