# COPOD Reproduction

This project runs COPOD on 14 benchmark datasets and compares the resulting
ROC-AUC and average precision (AP) scores with the values reported in the
COPOD paper. This project was made in VisualStudioCode, using Python 3.14.7.

## Files and data

Keep `COPOD_reproduction.py` and the `COPOD` folder together. The script looks
for these MATLAB `.mat` files in `COPOD/data`:

`arrhythmia`, `breastw`, `cardio`, `ionosphere`, `lympho`, `mammography`,
`optdigits`, `pima`, `satellite`, `satimage-2`, `shuttle`, `speech`, `wbc`,
and `wine`.

Each file must contain a feature matrix `X` and binary labels `y`. The script
skips a dataset if its file is missing, so keep all 14 files in place to run the full comparison.

## Setup and run

Use Python 3. Create a virtual environment in the project folder and install
the required packages. Choose the instructions for your operating system.

### Windows (PowerShell)

For a project folder in Downloads:

```powershell
cd "$HOME\Downloads\DW_Assignment_2"
py -3 --version
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn pyod
python -m py_compile COPOD_reproduction.py
python COPOD_reproduction.py
```


### macOS or Linux

For a project folder in Downloads:

```bash
cd ~/Downloads/DW_Assignment_2
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn pyod
python -m py_compile COPOD_reproduction.py
python COPOD_reproduction.py
```

The virtual environment keeps the required packages separate from other
Python installations. After the first setup, reopen a terminal, go to the
project folder, activate the environment, and run the script:

**Windows (PowerShell):**

```powershell
cd "$HOME\Downloads\DW_Assignment_2"
.\.venv\Scripts\Activate.ps1
python COPOD_reproduction.py
```

**macOS or Linux:**

```bash
cd ~/Downloads/DW_Assignment_2
source .venv/bin/activate
python COPOD_reproduction.py
```

The script expects the data at `COPOD/data`, relative to its own location.
Run it from the project folder so the results file is also saved there.

## Experiment settings

The settings are defined near the top of `COPOD_reproduction.py`:

- Datasets: the 14 files listed above.
- Trials per dataset: 10.
- Split: 60% training and 40% testing, with trial random states 0–9.
- Maximum dataset size: 10,000 rows. Larger datasets are subsampled without
  replacement using random seed 0.
- Scaling: `StandardScaler`, fitted on the training split only.
- COPOD contamination: the fraction of outlier labels in the dataset.
- Evaluation: ROC-AUC and average precision calculated from COPOD's continuous
  test decision scores, then averaged over the 10 trials.