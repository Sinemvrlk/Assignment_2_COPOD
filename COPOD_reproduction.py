
# Loading the libraries

import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pyod
import scipy
import sklearn
from pyod.models.copod import COPOD
from scipy.io import loadmat
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# Creating a list of the MAT files that are available in the COPOD paper's benchmark.
Implementation_file = [
    "arrhythmia.mat",
    "breastw.mat",
    "cardio.mat",
    "ionosphere.mat",
    "lympho.mat",
    "mammography.mat",
    "optdigits.mat",
    "pima.mat",
    "satellite.mat",
    "satimage-2.mat",
    "shuttle.mat",
    "speech.mat",
    "wbc.mat",
    "wine.mat",
]

# Settings for the reproduction of the COPOD paper's results.
DATA_DIR = Path(__file__).resolve().parent / "COPOD" / "data"
OUTPUT_FILE = Path("copod_results.csv")
N_ITE = 10
MAX_ROWS = 10000
SUBSAMPLE_SEED = 0

# Storing the ROC-AUC and average precision (AP), values, respectively, as reported in the COPOD paper for each dataset.
PAPER = {
    "arrhythmia": (0.8021, 0.4727),
    "breastw": (0.9936, 0.9877),
    "cardio": (0.8974, 0.5793),
    "ionosphere": (0.8307, 0.7194),
    "lympho": (0.9935, 0.8936),
    "mammography": (0.8942, 0.4290),
    "optdigits": (0.7328, 0.0532),
    "pima": (0.6638, 0.5407),
    "satellite": (0.6612, 0.5854),
    "satimage-2": (0.9852, 0.8599),
    "shuttle": (0.9980, 0.9807),
    "speech": (0.4845, 0.0195),
    "wbc": (0.9747, 0.7827),
    "wine": (0.9490, 0.6082),
}

# Running the COPOD model on a single dataset, with multiple trials and subsampling for large datasets.
def run_dataset(path, trials, max_rows, subsample_seed):

    # Storing each MAT file's feature matrix X and binary labels y.
    data = loadmat(path)
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y"]).ravel().astype(int)

    if X.ndim != 2 or len(X) != len(y) or not np.isfinite(X).all():
        raise ValueError(f"{path}: invalid features, labels, or non-finite values")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"{path}: labels must be binary 0/1")

    if len(X) > max_rows:
        # Using a fixed seed so that large-dataset subsampling is repeatable.
        indices = np.random.RandomState(subsample_seed).choice(
            len(X), max_rows, replace=False
        )
        X, y = X[indices], y[indices]

    # Calculating the contamination parameter for the COPOD model, which is the fraction of outliers in the dataset.
    contamination = float(np.mean(y))
    scores_by_trial = []

    for trial in range(trials):
        # Splitting the dataset into training and test sets, with a random state for reproducibility.
        X_train, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size = 0.4,
            random_state = np.random.RandomState(trial),
        )
        # Fitting the scaler only on training data, then transforming both training and test sets.
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Fitting the COPOD model to learn anomaly scores from the training features.
        model = COPOD(contamination = contamination)
        model.fit(X_train)
        # Ranking metrics with continuous anomaly scores.
        scores = np.nan_to_num(model.decision_function(X_test))

        if len(np.unique(y_test)) < 2:
            raise ValueError(f"{path}: test split {trial} contains only one class")

        # Measuring how well the scores rank outliers above normal test examples.
        scores_by_trial.append(
            [
                roc_auc_score(y_test, scores),
                average_precision_score(y_test, scores),
            ]
        )

    return contamination, np.asarray(scores_by_trial)


def main():

    # Compare each available dataset against the values reported in the paper.
    rows = []
    for mat_file in Implementation_file:
        name = Path(mat_file).stem
        paper_roc, paper_ap = PAPER[name]
        path = DATA_DIR / mat_file
        if not path.exists():
            print(f"SKIP {name}: missing {path}")
            continue

        contamination, scores = run_dataset(
            path, N_ITE, MAX_ROWS, SUBSAMPLE_SEED
        )
        roc, ap = scores.mean(axis=0)
        rows.append(
            {
                "dataset": name,
                "contamination": contamination,
                "trials": N_ITE,
                "roc_paper": paper_roc,
                "roc_implementation": roc,
                "roc_difference": roc - paper_roc,
                "ap_paper": paper_ap,
                "ap_implementation": ap,
                "ap_difference": ap - paper_ap,
                "roc_std": scores[:, 0].std(ddof=1) if N_ITE > 1 else 0,
                "ap_std": scores[:, 1].std(ddof=1) if N_ITE > 1 else 0,
            }
        )
        print(
            f"{name:15s} ROC {roc:.4f} (paper {paper_roc:.4f}) | "
            f"AP {ap:.4f} (paper {paper_ap:.4f})"
        )

    if not rows:
        raise SystemExit(f"No datasets were processed. Check {DATA_DIR}.")

    results = pd.DataFrame(rows)
    # Saving the paper values, implementation values, differences, and trial spread to a csv file.
    results.to_csv(OUTPUT_FILE, index = False)
    print(f"\nSaved: {OUTPUT_FILE} | {len(results)} of {len(Implementation_file)} datasets")

    for metric, paper_col, implementation_col in (
        ("ROC", "roc_paper", "roc_implementation"),
        ("AP", "ap_paper", "ap_implementation"),
    ):
        # Using absolute relative differences to rank change size, so negative values do not cancel positive values.
        relative_change = (
            (results[implementation_col] - results[paper_col]) / results[paper_col]
        ).abs()
        largest = relative_change.idxmax()
        smallest = relative_change.idxmin()
        print(
            f"{metric} relative change: largest {relative_change[largest]:.2%} "
            f"({results.loc[largest, 'dataset']}), smallest "
            f"{relative_change[smallest]:.2%} "
            f"({results.loc[smallest, 'dataset']})"
        )

    # Print the settings needed to describe and reproduce this run.
    print("\nSettings for reproduction of the implementation:")
    print(f"  Trials: {N_ITE}; split: 60/40; split seeds: 0 to {N_ITE - 1}")
    print(f"  Maximum rows: {MAX_ROWS}; subsampling seed: "
          f"{SUBSAMPLE_SEED} (only used above the maximum)"
    )

    print("  Scaling: StandardScaler fitted on each training split")
    print("  COPOD contamination: outlier-label fraction in each full dataset")
    print("  Metrics: mean ROC-AUC and average precision from test decision scores")
    print("  NOTE: This implementation covers the listed .mat datasets, not the paper's full benchmark.")


if __name__ == "__main__":
    main()
