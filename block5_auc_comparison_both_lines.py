"""
Block 5 (extended) — Expression Prediction AUC Comparison, GM12878 + K562
Mirrors the Day-6 GM12878 script exactly, run twice (once per cell line),
then renders both ROC panels in one figure for direct visual comparison.

Models per cell line:
  Model 1: Individual histone marks only   (5 binary features)
  Model 2: Gate type only                  (one-hot encoded categorical)
  Model 3: Complexity score alone          (continuous)

Run from: C:\\Users\\jamoo\\Downloads\\LOGIC CIRCUITS\\
Inputs:
  - data/gate_assignments.tsv          (GM12878)
  - data/binary_matrix.csv             (GM12878)
  - data/k562_gate_assignments_named.tsv  (K562)
  - data/k562_binary_matrix.csv           (K562)
Outputs:
  - figures/auc_comparison_gm12878_vs_k562_day15.png
  - block5_auc_results_day15.txt
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"

import sys
import traceback
import faulthandler

_FAULT_LOG_PATH = r"C:\Users\jamoo\Downloads\LOGIC CIRCUITS\block5_day15_faulthandler_log.txt"
_fault_fh = open(_FAULT_LOG_PATH, "w")
faulthandler.enable(file=_fault_fh, all_threads=True)

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy  as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = r"C:\Users\jamoo\Downloads\LOGIC CIRCUITS"

LINES = {
    "GM12878": {
        "gate_file":   os.path.join(BASE_DIR, "data", "gate_assignments.tsv"),
        "matrix_file": os.path.join(BASE_DIR, "data", "binary_matrix.csv"),
    },
    "K562": {
        "gate_file":   os.path.join(BASE_DIR, "data", "k562", "k562_gate_assignments_named.tsv"),
        "matrix_file": os.path.join(BASE_DIR, "data", "k562", "k562_binary_matrix.csv"),
    },
}

FIG_DIR  = os.path.join(BASE_DIR, "figures")
FIG_FILE = os.path.join(FIG_DIR,  "auc_comparison_gm12878_vs_k562_day15.png")
OUT_FILE = os.path.join(BASE_DIR, "block5_auc_results_day15.txt")

os.makedirs(FIG_DIR, exist_ok=True)

MARK_COLS = ["h3k4me3", "h3k27ac", "h3k4me1", "h3k27me3", "h3k9me3"]
REQUIRED_MATRIX_COLS = MARK_COLS + ["expressed"]


def resolve_column(columns, candidates, label, file_label):
    for c in candidates:
        if c in columns:
            return c
    sys.exit(f"ERROR: No {label} column found in {file_label}.\nColumns: {list(columns)}")


def load_and_merge(gate_file, matrix_file, line_label):
    for path, label in [(gate_file, os.path.basename(gate_file)),
                         (matrix_file, os.path.basename(matrix_file))]:
        if not os.path.isfile(path):
            sys.exit(f"ERROR [{line_label}]: Cannot find {label} at:\n  {path}")

    gates = pd.read_csv(gate_file, sep="\t")
    gates.columns = [c.strip().lower() for c in gates.columns]

    gene_col = None
    for candidate in ["gene_id", "gene", "gene_name", "symbol"]:
        if candidate in gates.columns:
            gene_col = candidate
            break
    if gene_col is None:
        first_col = gates.columns[0]
        if gates[first_col].astype(str).str.startswith("ENSG").any():
            gene_col = first_col
        else:
            sys.exit(f"ERROR [{line_label}]: No gene name column found.\nColumns: {list(gates.columns)}")

    gate_col = resolve_column(gates.columns, ["gate", "gate_type", "gate_call"],
                               "gate", f"{line_label} gate file")
    complexity_col = resolve_column(gates.columns, ["complexity", "complexity_score", "complexity score"],
                                     "complexity", f"{line_label} gate file")

    mat = pd.read_csv(matrix_file, index_col=0)
    mat.columns = [c.strip().lower() for c in mat.columns]

    missing_cols = [c for c in REQUIRED_MATRIX_COLS if c not in mat.columns]
    if missing_cols:
        sys.exit(
            f"ERROR [{line_label}]: binary_matrix is missing columns: {missing_cols}\n"
            f"Columns found: {list(mat.columns)}"
        )

    mat = mat[REQUIRED_MATRIX_COLS].copy()
    mat.index.name = "ensembl_id"
    mat = mat.reset_index()

    pre_merge = len(gates)
    df = gates.merge(mat, left_on=gene_col, right_on="ensembl_id", how="inner")
    post_merge = len(df)

    print(f"[{line_label}] Gate assignments : {pre_merge:,} genes")
    print(f"[{line_label}] Binary matrix    : {len(mat):,} genes")
    print(f"[{line_label}] After inner join : {post_merge:,} genes")

    if post_merge == 0:
        sys.exit(
            f"ERROR [{line_label}]: Zero rows after merge. Gene IDs do not match between files.\n"
            f"  gate file sample IDs   : {gates[gene_col].head(3).tolist()}\n"
            f"  matrix file sample IDs : {mat['ensembl_id'].head(3).tolist()}"
        )
    if post_merge < 1000:
        print(f"WARNING [{line_label}]: Only {post_merge:,} genes matched. Check ID format mismatches.")

    required = MARK_COLS + [gate_col, complexity_col, "expressed"]
    df_clean = df[required].dropna()
    dropped = post_merge - len(df_clean)
    if dropped:
        print(f"[{line_label}] Dropped {dropped:,} rows with NaN. Proceeding with {len(df_clean):,} genes.")
    if len(df_clean) < 100:
        sys.exit(f"ERROR [{line_label}]: Fewer than 100 clean rows after dropna.")

    return df_clean, gate_col, complexity_col, pre_merge, post_merge, mat


def fit_eval(X, idx_tr, idx_te, y):
    lr = LogisticRegression(max_iter=1000, random_state=42, solver="liblinear", n_jobs=1)
    lr.fit(X[idx_tr], y[idx_tr])
    prob = lr.predict_proba(X[idx_te])[:, 1]
    auc = roc_auc_score(y[idx_te], prob)
    fpr, tpr, _ = roc_curve(y[idx_te], prob)
    return auc, fpr, tpr


def run_line(line_label, gate_file, matrix_file):
    df_clean, gate_col, complexity_col, pre_merge, post_merge, mat = load_and_merge(
        gate_file, matrix_file, line_label
    )

    X_marks = df_clean[MARK_COLS].values.astype(float)
    X_gate  = pd.get_dummies(df_clean[gate_col].astype(str), prefix="gate").values.astype(float)
    X_compl = df_clean[complexity_col].values.reshape(-1, 1).astype(float)
    y       = df_clean["expressed"].values.astype(int)

    print(f"[{line_label}] Expressed: {y.sum():,} / {len(y):,} genes ({y.mean()*100:.1f}%)")

    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx, test_size=0.20, random_state=42, stratify=y)

    print(f"[{line_label}] Train: {len(idx_tr):,}  |  Test: {len(idx_te):,}  |  "
          f"Expressed in test: {y[idx_te].mean()*100:.1f}%")

    auc_marks, fpr_m, tpr_m = fit_eval(X_marks, idx_tr, idx_te, y)
    auc_gate,  fpr_g, tpr_g = fit_eval(X_gate,  idx_tr, idx_te, y)
    auc_compl, fpr_c, tpr_c = fit_eval(X_compl, idx_tr, idx_te, y)

    print(f"[{line_label}] Model 1 (individual marks)  AUC = {auc_marks:.4f}")
    print(f"[{line_label}] Model 2 (gate type)         AUC = {auc_gate:.4f}")
    print(f"[{line_label}] Model 3 (complexity score)  AUC = {auc_compl:.4f}")

    return {
        "line": line_label,
        "pre_merge": pre_merge, "post_merge": post_merge,
        "n_train": len(idx_tr), "n_test": len(idx_te),
        "expr_test_pct": y[idx_te].mean() * 100,
        "auc_marks": auc_marks, "fpr_marks": fpr_m, "tpr_marks": tpr_m,
        "auc_gate":  auc_gate,  "fpr_gate":  fpr_g, "tpr_gate":  tpr_g,
        "auc_compl": auc_compl, "fpr_compl": fpr_c, "tpr_compl": tpr_c,
    }


CRASH_LOG = os.path.join(BASE_DIR, "block5_day15_crash_log.txt")

def main():
    # ── run both cell lines ─────────────────────────────────────────────────
    results = {}
    for line_label, paths in LINES.items():
        print(f"\n>>> Starting {line_label}...")
        results[line_label] = run_line(line_label, paths["gate_file"], paths["matrix_file"])
        print(f">>> Finished {line_label}.")
    return results

try:
    results = main()
except Exception:
    tb_text = traceback.format_exc()
    print("\n" + "=" * 70)
    print("SCRIPT CRASHED — full traceback below and saved to:")
    print(f"  {CRASH_LOG}")
    print("=" * 70)
    print(tb_text)
    with open(CRASH_LOG, "w", encoding="utf-8") as fh:
        fh.write(tb_text)
    sys.exit(1)

# ── combined figure, 2 panels side by side ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

panel_colors = {"marks": "#2c7bb6", "gate": "#d7191c", "compl": "#1a9641"}

for ax, line_label in zip(axes, ["GM12878", "K562"]):
    r = results[line_label]
    ax.plot(r["fpr_marks"], r["tpr_marks"], color=panel_colors["marks"], lw=2,
            label=f"Individual marks  (AUC = {r['auc_marks']:.3f})")
    ax.plot(r["fpr_gate"], r["tpr_gate"], color=panel_colors["gate"], lw=2,
            label=f"Gate type          (AUC = {r['auc_gate']:.3f})")
    ax.plot(r["fpr_compl"], r["tpr_compl"], color=panel_colors["compl"], lw=2, linestyle="--",
            label=f"Complexity score   (AUC = {r['auc_compl']:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle=":")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"{line_label}\nPredicting Gene Expression (TPM > 1)", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

fig.suptitle("Epigenetic Logic Gate Framework — ROC Comparison Across Cell Lines",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIG_FILE, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigure saved: {FIG_FILE}")

# ── combined text report ────────────────────────────────────────────────────
lines_out = []
lines_out.append("=" * 70)
lines_out.append("DAY 15 — AUC COMPARISON RESULTS, GM12878 vs K562")
lines_out.append("=" * 70)

for line_label in ["GM12878", "K562"]:
    r = results[line_label]
    delta = r["auc_gate"] - r["auc_marks"]
    gate_wins = r["auc_gate"] > r["auc_marks"]

    lines_out.append("")
    lines_out.append(f"--- {line_label} ---")
    lines_out.append(f"  Merged : {r['post_merge']:,} / {r['pre_merge']:,} genes")
    lines_out.append(f"  Train / Test : {r['n_train']:,} / {r['n_test']:,}")
    lines_out.append(f"  Expressed (test) : {r['expr_test_pct']:.1f}%")
    lines_out.append(f"  Model 1 (individual marks)   AUC = {r['auc_marks']:.4f}")
    lines_out.append(f"  Model 2 (gate type)          AUC = {r['auc_gate']:.4f}")
    lines_out.append(f"  Model 3 (complexity score)   AUC = {r['auc_compl']:.4f}")
    lines_out.append(f"  Gate AUC - Marks AUC = {delta:+.4f}")
    if gate_wins:
        lines_out.append("  RESULT: Gate type model outperforms individual marks.")
    else:
        lines_out.append("  RESULT: Individual marks model matches or outperforms gate type.")

lines_out.append("")
lines_out.append("--- Cross-line comparison ---")
gm_delta = results["GM12878"]["auc_gate"] - results["GM12878"]["auc_marks"]
k5_delta = results["K562"]["auc_gate"] - results["K562"]["auc_marks"]
lines_out.append(f"  GM12878 gate-model advantage : {gm_delta:+.4f}")
lines_out.append(f"  K562    gate-model advantage : {k5_delta:+.4f}")
if k5_delta < gm_delta:
    lines_out.append("  → Gate framework's predictive advantage over individual marks")
    lines_out.append("    is SMALLER in K562 than GM12878 — consistent with the")
    lines_out.append("    INCONSISTENT-class expansion (23.0% → 44.4%) reducing how much")
    lines_out.append("    structured logic the framework can extract in the malignant line.")
else:
    lines_out.append("  → Gate framework's predictive advantage over individual marks")
    lines_out.append("    is preserved or larger in K562 relative to GM12878.")
lines_out.append("")
lines_out.append("=" * 70)

output = "\n".join(lines_out)
print(output)

with open(OUT_FILE, "w", encoding="utf-8") as fh:
    fh.write(output)
print(f"Report saved: {OUT_FILE}")