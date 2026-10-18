# Installation and demonstration

## System and dependencies

The numerical utilities support Python 3.11+ on Linux and Windows. They require NumPy and ordinary CPU hardware. The report renderer additionally uses Matplotlib. No special GPU, EEG files, pretrained models or private service is needed for the reference demo.

The historical GPU sources have different requirements and external inputs; see [REPRODUCIBILITY.md](REPRODUCIBILITY.md). The commands below do not recreate that environment.

## Install

Create and activate a Python virtual environment using the procedure appropriate to your OS, then:

```bash
git clone https://github.com/Botwwt/EEG-Diffusion.git
cd EEG-Diffusion
python -m pip install -e '.[dev,report]'
```

For a review environment with the tested top-level versions:

```bash
python -m pip install -r requirements/review.txt
python -m pip install -e .
```

Keep pip's default build isolation, which installs the declared build backend. A host setuptools 65.5 build without isolation produced a broken editable installation under a Chinese Windows path during local checking; reinstalling with the declared isolated backend worked. Do not bypass the backend requirements when reproducing the review environment.

## Demo and expected result

```bash
eeg-diffusion reference-demo --out outputs/demo.json
eeg-diffusion verify
python -m pytest -q
```

The demo conditions a scalar Gaussian variable on two distinct correlated observations. The expected posterior mean is 0.4 and covariance 0.5, up to floating-point precision; the output marks the data as synthetic and reports two unique trials. `verify` should check 72 frozen source/assets and three aggregate evidence files. The current test suite has 32 tests.

Measured once on the recorded Windows/Python 3.11.9 host:

| Command | Wall time | Scope |
|---|---:|---|
| Editable install with `--no-deps` | 16.226 s | Runtime dependencies already installed; includes isolated backend preparation |
| `reference-demo` | 0.631 s | Includes interpreter and NumPy startup |
| `verify` | 0.401 s | Includes interpreter, file hashes and aggregate arithmetic |

These timings do not include a fresh dependency download and are not performance benchmarks for EEG training or image generation. Details are recorded in [review_environment.json](../provenance/review_environment.json).

## Checks and rebuilding

```bash
python -m ruff check src scripts tests
python -m ruff format --check src scripts tests
python -m pytest --cov=eeg_diffusion --cov-report=term
python scripts/build_report.py --check
python scripts/build_figures.py --check
```

Run the builders without `--check` to regenerate their artifacts. Figure regeneration records the Matplotlib and font environment; committed figure hashes are verified independently of installed renderer versions. Open `docs/index.html` to read the report offline.
