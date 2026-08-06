# MULTIPINN

MULTIPINN is a Python framework for **Physics-Informed Neural Networks (PINNs)** built on top of **PyTorch**.  
The repository contains reusable components for defining PDE-based problems, generating collocation points, configuring experiments with **Hydra**, training PINN models, and visualizing the results.

## Canonical user tutorial

The **canonical getting-started tutorial** for installation and the first run is:

- [`docs/guide/getting-started.ipynb`](docs/guide/getting-started.ipynb)

This README and the notebook above are intentionally synchronized.  
For all user-facing instructions, use the following common baseline:

- **Supported Python versions:** `3.8` – `3.11`
- **Recommended Python version for a new environment:** `3.10`
- **Canonical installation command:** `pip install -e .`
- `make install` is only a thin wrapper around the same command

## What MULTIPINN provides

- Fully-connected, residual, Fourier-feature, dense, factorized and other neural-network architectures
- PINN conditions for PDE residuals, boundary conditions and extra data
- Static and adaptive point generators
- Hydra-based experiment configuration
- Training loops for standard, gradual and multi-GPU workflows
- Built-in callbacks for logging, checkpoints and visualization
- Support for mesh- and graph-based workflows

## Repository structure

```text
multipinn/                     
examples/<example_name>/       
examples/<example_name>/configs/config.yaml
examples/<example_name>/problem.py
examples/<example_name>/run_train.py
docs/guide/getting-started.ipynb
tests/
```

The main files a user usually edits in an example are:

- `configs/config.yaml` — experiment parameters
- `problem.py` — equations, geometry and conditions
- `run_train.py` — training entry point

## System requirements

### Minimum hardware requirements

MULTIPINN is used for numerical experiments of very different computational cost,
so hardware requirements are listed separately for three scenarios:

- installing the library and running the unit tests
- reproducing lightweight 2D experiments
- reproducing computationally heavy 3D and multi-GPU experiments

| Component | Minimum (install + unit tests) | Recommended (2D experiments) | Recommended (3D / multi-GPU experiments) |
|---|---|---|---|
| OS | Linux x86_64 / Windows 10+ | Linux x86_64 | Linux x86_64 |
| CPU | 4 cores | 8 cores | 16+ cores |
| RAM | 16 GB | 32 GB | 128 GB or more |
| Disk | 20 GB SSD | 100 GB SSD | 200+ GB SSD |
| GPU | Not required | NVIDIA GPU with CUDA support | NVIDIA A100 or a computationally comparable accelerator |
| VRAM | Not required | 8–12 GB or more | 40 GB or more |
| CUDA | Not required | Version compatible with PyTorch | Version compatible with PyTorch |

The heaviest experiments reported for MULTIPINN were reproduced on a compute node
based on an AMD EPYC 7313 CPU (16 cores, about 0.28 TFLOPS FP64) with NVIDIA A100
class accelerators or comparable analogues. For some 3D formulations, increasing
the number of collocation points leads to a substantial growth of VRAM
consumption — take this into account when choosing a configuration.

### Software requirements

| Component | Requirement / note |
|---|---|
| Python | `>=3.8, <3.12` |
| OS | Linux x86_64; Windows 10/11 with a compatible Python environment; Windows 10/11 with WSL2 |
| PyTorch | A version compatible with your CUDA installation |
| CUDA | NVIDIA driver and CUDA runtime are required for GPU runs; not needed for CPU-only usage |
| Other Python dependencies | See `requirements.txt` |
| Browser for Plotly PNG export | Google Chrome / Chromium may be required by `kaleido` for static PNG export |
| Extra graph/mesh dependencies | Some GNN / mesh examples may require compatible PyG binary wheels, see below |

## Installation

All commands below assume that you run them **from the repository root**.

### 1. Clone the repository

```bash
git clone https://github.com/multipinn/multipinn.git
cd multipinn
```

### 2. Create and activate a virtual environment

#### Recommended: `venv`

**Linux / WSL2**
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

If `python3.10` is not available, use any supported version from `3.8` to `3.11`.

**Windows (PowerShell)**
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Alternative: `conda`

```bash
conda create -n multipinn python=3.10
conda activate multipinn
```

### 3. Install the package

```bash
python -m pip install --upgrade pip
pip install -e .
```

Equivalent shorthand from the Makefile:

```bash
make install
```

## Verifying the installation

After installation, the following command should work:

```bash
python -c "import multipinn, torch; print('MULTIPINN import OK'); print('Torch:', torch.__version__)"
```

## Quick smoke test

For a short sanity check, it is convenient to run a lightweight example with reduced settings.  
The command below runs the `regression_1D` example for a few epochs and saves interactive HTML artifacts instead of PNG files:

```bash
python -m examples.regression_1D.run_train \
  trainer.num_epochs=5 \
  generator.domain_points=256 \
  visualization.grid_plot_points=256 \
  visualization.save_period=1 \
  visualization.save_mode=html \
  paths.save_dir=./artifacts_smoke/regression_1D
```
After the smoke test completes successfully, the `artifacts_smoke/regression_1D` directory should contain not only `Used_config.yaml` but also the HTML visualization, loss plot functions, and model checkpoints. If you only need a minimal import check without generating artifacts, use the command from the "Installation Verification" section.


## First full example

A more representative PDE example is `poisson_2D_1C`:

```bash
python -m examples.poisson_2D_1C.run_train
```

If you want a shorter local run, override the defaults from Hydra:

```bash
python -m examples.poisson_2D_1C.run_train \
  trainer.num_epochs=100 \
  generator.domain_points=2000 \
  generator.bound_points=400 \
  visualization.grid_plot_points=1000 \
  visualization.save_period=100 \
  visualization.save_mode=html
```

## How to configure an experiment

Typical workflow:

1. Choose an example under `examples/`.
2. Adjust the parameters in `configs/config.yaml`.
3. Edit `problem.py` if you need to change equations, geometry or conditions.
4. Run the experiment via `python -m examples.<example_name>.run_train`.

Common Hydra sections in example configs:

- `problem` — problem-specific parameters
- `model` — neural-network architecture and its parameters
- `regularization` — loss balancing strategy
- `generator` — number of domain and boundary points
- `trainer` — epochs, seeds, update frequency
- `visualization` — plot density, save period, output format
- `optimizer`, `scheduler` — optimizer and learning-rate schedule
- `paths` — output directories

## Results and artifacts

Results are written to the directory defined by `paths.save_dir`.

Typical artifacts include:

- `used_config.yaml` — exact configuration used for the run
- model checkpoints (`.pth`)
- loss curves
- Plotly visualizations in `html` and/or `png`
- additional callback outputs depending on the example

If you do not need static PNG files, prefer:

```yaml
visualization:
  save_mode: html
```

This avoids extra browser requirements for Plotly image export.

## Notes for GNN and mesh examples

The base installation already includes `torch_geometric` from `requirements.txt`.  
However, some graph- and mesh-based workflows can additionally depend on [PyG binary packages](https://data.pyg.org/whl/) such as:

- `pyg_lib`
- `torch_scatter`
- `torch_sparse`
- `torch_cluster`
- `torch_spline_conv`

If an example fails with errors related to `torch_scatter`, `torch_cluster`, `knn_graph` or `radius_graph`, install the matching PyG packages for your **PyTorch** and **CUDA** combination, then rerun the example.

To check your current Torch and CUDA versions:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

## Developer setup

For development, install the package together with dev dependencies:

```bash
pip install -e ".[dev]"
```

Equivalent shorthand:

```bash
make install.dev
```

Install pre-commit hooks:

```bash
make install-pre-commit
```

Run the test suite:

```bash
make test
```

## Contribution workflow

1. Fork the repository.
2. Create a feature branch.
3. Make a small, focused change.
4. Run tests locally.
5. Open a pull request with a concise description of the change.

When you add or change user-facing behavior, update the corresponding documentation as well — at minimum this README and `docs/guide/getting-started.ipynb`.

## Documentation policy for installation

To avoid contradictory instructions across the repository, use the following rules in all user-facing documents:

- keep the installation command as `pip install -e .`
- keep the supported Python range as `3.8–3.11`
- keep the supported operating systems as Linux x86_64, Windows 10/11 and Windows 10/11 with WSL2
- keep `docs/guide/getting-started.ipynb` as the main tutorial
- treat `make install` only as a shorthand for `pip install -e .`
- do not introduce alternative installation paths as the primary workflow unless they are synchronized here first
- keep the minimum hardware requirements in the [Minimum hardware requirements](#minimum-hardware-requirements) section as the single source of truth, and mirror any change to it in `docs/index.md` and `docs/guide/getting-started.ipynb`

## Full guide for users and developers

[Read full guide for users and developers](https://docs.google.com/document/d/1UiArho_iPbQvwLAPaCg7ojdMlovd1ELGTU3VrW-pTp8/edit?tab=t.0)
