# patrec_time_series
Time-series Pattern Recognition

## Project layout

- `patrec/generation/` — synthetic time-series generators, with YAML presets in `base_configs/` (trend, wave, shift, noise)
- `patrec/utils/` — data loaders, preprocessing, visualisation and logging (including MLflow) helpers
- `.reqs/` — dependency lists for Python 3.10 and 3.12

## Setup

Create a virtual environment (the names below are already ignored by git) and install the requirements for your Python version.

Python 3.12:

```bash
python -m venv .py312_venv
source .py312_venv/bin/activate   # Windows: .py312_venv\Scripts\activate
pip install -r .reqs/py_312.txt
```

Python 3.10:

```bash
python -m venv .py310_venv
source .py310_venv/bin/activate   # Windows: .py310_venv\Scripts\activate
pip install -r .reqs/py310.txt
```
