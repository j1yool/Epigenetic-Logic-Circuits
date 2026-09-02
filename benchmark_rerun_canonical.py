import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

MARK_COLS = ["H3K4me3", "H3K27ac", "H3K4me1", "H3K27me3", "H3K9me3"]
RANDOM_STATE = 42
N_SPLITS = 5

# Confirm these two subpaths match your actual layout before running.
CELL_LINE_FILES = {
    "GM12878": {
        "gates": DATA_DIR / "gate_assignments_named.tsv",
        "marks": DATA_DIR / "binary_matrix.csv",
    },
    "K562": {
        "gates": DATA_DIR / "k562" / "k562_gate_assignments_named.tsv",
        "marks": DATA_DIR / "k562" / "k562_binary_matrix.csv",  
    },
}

def load_cell_line(files):
    gates = pd.read_csv(files["gates"], sep="\t")
    marks = pd.read_csv(files["marks"])
    marks = marks.rename(columns={"Unnamed: 0": "gene_id"})
    df = gates.merge(marks, on="gene_id", how="inner")
    before = len(df)
    df = df.dropna(subset=["gate_type", "expressed"])
    print(f"  dropped {before - len(df)} unassigned-gate/unexpressed rows ({before} -> {len(df)})")
    return df

def fit_and_score(X, y, model_label, cell_line, results):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    clf = LogisticRegression(solver="liblinear", max_iter=1000, random_state=RANDOM_STATE)
    probs = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, probs)
    results.append({"cell_line": cell_line, "model": model_label, "auc": round(auc, 4)})
    print(f"{cell_line:10s} | {model_label:15s} | AUC = {auc:.4f}")

def run_benchmark(df, cell_line, results):
    y = df["expressed"].values
    gate_dummies = pd.get_dummies(df["gate_type"], prefix="gate", drop_first=True)
    fit_and_score(gate_dummies.values, y, "gate-type", cell_line, results)

    marks_X = df[MARK_COLS].values
    fit_and_score(marks_X, y, "marks-only", cell_line, results)

    complex_X = df[MARK_COLS + ["complexity_score"]].values
    fit_and_score(complex_X, y, "X_complex", cell_line, results)

def main():
    results = []
    for cell_line, files in CELL_LINE_FILES.items():
        print(f"\nLoading {cell_line} from {files['gates'].parent} ...")
        df = load_cell_line(files)
        run_benchmark(df, cell_line, results)

    out = pd.DataFrame(results)
    out.to_csv("benchmark_rerun_results.csv", index=False)
    print("\nSaved benchmark_rerun_results.csv")

    locked = {
        ("GM12878", "gate-type"): 0.9864,
        ("K562", "gate-type"): 0.9986,
        ("GM12878", "marks-only"): 0.7974,
        ("K562", "marks-only"): 0.7535,
    }
    print("\n--- Reconciliation check vs locked benchmark ---")
    for (cell_line, model), locked_auc in locked.items():
        row = out[(out["cell_line"] == cell_line) & (out["model"] == model)]
        if row.empty:
            continue
        new_auc = row["auc"].values[0]
        diff = abs(new_auc - locked_auc)
        status = "MATCH" if diff <= 0.001 else "DIVERGENT — LOG, DO NOT ITERATE"
        print(f"{cell_line:10s} | {model:12s} | locked={locked_auc:.4f} new={new_auc:.4f} diff={diff:.4f} -> {status}")

if __name__ == "__main__":
    main()