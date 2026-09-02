"""
switching_gene_expression_prediction.py

Tests whether the gate-type model's AUC advantage over marks-only is
concentrated in switching genes (Day 32 output: switching_genes_annotated_v1.tsv),
vs. uniform genome-wide.

BUILT DIRECTLY FROM UPLOADED FILES:
  - gate_assignments_named.tsv   (GM12878; gene_id, gate_type, complexity_score, gene_name)
  - k562_gate_assignments_named.tsv (K562; gene_id, gate_type, complexity_score -- NO gene_name)
  - binary_matrix.csv            (GM12878; index=gene_id, 5 marks + expressed)
  - switching_genes_annotated_v1.tsv (keyed by SYMBOL, not gene_id -- see below)
  - k562_binary_matrix.csv       NOT UPLOADED. Assumed same schema as binary_matrix.csv,
    at the path the Day 14 script wrote/read from. If this fails, the real file's
    columns don't match binary_matrix.csv and need to be pasted.

TWO REAL ISSUES FOUND IN THE DATA (not hypothetical -- confirmed by inspection):

1. switching_genes_annotated_v1.tsv is keyed by gene SYMBOL ('gene' column),
   while gate_assignments_named.tsv and binary_matrix.csv are keyed by Ensembl
   gene_id. A direct join on 'gene_id' returns zero rows. This script joins via
   symbol -> gene_id using gate_assignments_named.tsv's gene_name column as the
   crosswalk (K562's gate assignment file has no gene_name column at all, so its
   crosswalk is inherited from the GM12878 file, assuming the same Ensembl gene
   set -- true under the locked shared-GRCh38/Ensembl-109-GTF decision).

2. gate_type contains the literal string 'NULL' for silent genes (from
   call_gate() in the Day 5 / Day 14 scripts). pandas' default CSV parser
   treats the text "NULL" as a missing value on read, silently converting
   ~34k-38k rows to NaN. This script reads with keep_default_na=False so
   'NULL' survives as a real category -- confirmed necessary, not precautionary.

3. gene_name/gene symbol is NOT a unique key (69 Ensembl IDs in GM12878 share
   a duplicated symbol, mostly Y_RNA/snoRNA paralog families; 10 of the 7,474
   switching genes hit this). This script explicitly excludes ambiguous
   symbols from the subset rather than silently duplicating rows into the
   AUC calculation, and prints exactly which genes were dropped and why.

Output: switching_gene_auc_comparison.csv
  columns: cell_line, model_type, subset_auc, genome_wide_auc, subset_n
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# =============================================================================
# CONFIG
# =============================================================================
BASE = r"C:\Users\jamoo\Downloads\LOGIC CIRCUITS\data"

GM_GATE_PATH = fr"{BASE}\gate_assignments_named.tsv"
GM_BINARY_PATH = fr"{BASE}\binary_matrix.csv"
K562_GATE_PATH = fr"{BASE}\k562\k562_gate_assignments_named.tsv"
K562_BINARY_PATH = fr"{BASE}\k562\k562_binary_matrix.csv"   # NOT verified -- see docstring
SWITCHING_GENES_PATH = "switching_genes_annotated_v1.tsv"

OUTPUT_PATH = "switching_gene_auc_comparison.csv"

MARKS_COLS = ["H3K4me3", "H3K27ac", "H3K4me1", "H3K27me3", "H3K9me3"]
COMPLEXITY_COL = "complexity_score"
GATE_TYPE_COL = "gate_type"
EXPRESSION_COL = "expressed"

# Locked genome-wide benchmark AUCs (MASTER_STATUS.md) -- for comparison only,
# NOT recomputed here. If this script's own genome-wide AUC (printed at runtime)
# doesn't roughly match these, the feature construction here diverges from
# whatever produced the locked numbers, and that mismatch needs to be found
# before trusting the subset comparison.
GENOME_WIDE_AUC = {
    ("GM12878", "gate_type"): 0.9864,
    ("K562", "gate_type"): 0.9986,
    ("GM12878", "marks_only"): 0.7974,
    ("K562", "marks_only"): 0.7535,
}

RANDOM_STATE = 42


# =============================================================================
# STEP 1 -- Load gate assignments (GM12878 has gene_name; K562 does not)
# =============================================================================
def load_gate_assignments():
    gm_gates = pd.read_csv(
        GM_GATE_PATH, sep="\t", keep_default_na=False, na_values=[""]
    )
    k562_gates = pd.read_csv(
        K562_GATE_PATH, sep="\t", keep_default_na=False, na_values=[""]
    )

    for name, df, needs_name in [("GM12878 gates", gm_gates, True),
                                   ("K562 gates", k562_gates, False)]:
        required = ["gene_id", GATE_TYPE_COL, COMPLEXITY_COL] + (["gene_name"] if needs_name else [])
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{name} missing expected columns: {missing}. Found: {list(df.columns)}")

    # Build the symbol -> gene_id crosswalk from GM12878 (only file with gene_name).
    # Flag ambiguous symbols (>1 gene_id per symbol) rather than silently
    # duplicating rows downstream.
    symbol_counts = gm_gates["gene_name"].value_counts()
    ambiguous_symbols = set(symbol_counts[symbol_counts > 1].index)
    if ambiguous_symbols:
        print(f"[crosswalk] {len(ambiguous_symbols)} gene symbols map to >1 Ensembl ID "
              f"in GM12878 gate assignments (e.g. small-RNA paralog families like Y_RNA, "
              f"snoRNAs). These will be excluded from the switching-gene subset wherever "
              f"they appear, to avoid duplicate-row inflation of the AUC calculation.")

    crosswalk = gm_gates[~gm_gates["gene_name"].isin(ambiguous_symbols)][["gene_id", "gene_name"]].copy()
    crosswalk = crosswalk.rename(columns={"gene_name": "gene_symbol"})

    # Attach symbol to K562 gate table via gene_id (assumes shared Ensembl gene set,
    # true under the locked shared-GRCh38 decision).
    k562_gates = k562_gates.merge(crosswalk, on="gene_id", how="left")

    gm_gates = gm_gates.rename(columns={"gene_name": "gene_symbol"})

    return gm_gates, k562_gates, crosswalk, ambiguous_symbols


# =============================================================================
# STEP 2 -- Load binary matrices (marks + expressed), reset index to gene_id
# =============================================================================
def load_binary_matrix(path, cell_line_label):
    df = pd.read_csv(path, index_col=0)
    df.index.name = "gene_id"
    df = df.reset_index()
    missing = [c for c in MARKS_COLS + [EXPRESSION_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"{cell_line_label} binary matrix at {path} is missing columns {missing}. "
            f"Found: {list(df.columns)}. If this is k562_binary_matrix.csv and it "
            f"wasn't actually verified against this schema, paste its real header row."
        )
    return df


# =============================================================================
# STEP 3 -- Identify switching-gene subset per cell line, with loud diagnostics
# =============================================================================
def build_subset_mask(gates_df, switching_df, ambiguous_symbols, cell_line_label):
    switching_df = switching_df.copy()
    switching_df["gene"] = switching_df["gene"].str.strip()

    # 967 switching genes have no resolved symbol -- their 'gene' column IS
    # already an Ensembl ID (gene_name is NaN for these rows). Match those
    # directly on gene_id. Match everything else on symbol, as before.
    is_ensembl_id = switching_df["gene"].str.match(r"^ENSG\d+$")
    direct_id_genes = set(switching_df.loc[is_ensembl_id, "gene"])
    symbol_genes = set(switching_df.loc[~is_ensembl_id, "gene"]) - ambiguous_symbols

    n_ambiguous_in_switching = len(
        set(switching_df.loc[~is_ensembl_id, "gene"]) & ambiguous_symbols
    )
    if n_ambiguous_in_switching:
        print(f"[{cell_line_label}] {n_ambiguous_in_switching} switching genes have an "
              f"ambiguous symbol and are excluded from this subset check.")

    mask_by_id = gates_df["gene_id"].isin(direct_id_genes)
    mask_by_symbol = gates_df["gene_symbol"].isin(symbol_genes)
    subset_mask = mask_by_id | mask_by_symbol

    subset_n = int(subset_mask.sum())
    expected_n = len(direct_id_genes) + len(symbol_genes)

    if subset_n == 0:
        raise ValueError(f"[{cell_line_label}] subset_n = 0 -- crosswalk failed entirely.")
    if subset_n < 0.9 * expected_n:
        print(f"[{cell_line_label}] WARNING: subset_n ({subset_n}) below 90% of "
              f"expected ({expected_n}). Some switching genes may genuinely be "
              f"absent from this cell line's gate assignment table.")

    return subset_mask, subset_n


# =============================================================================
# STEP 4 -- Build locked feature sets
# =============================================================================
def build_feature_matrices(merged):
    # Gate-type model: full one-hot, ALL categories including 'NULL' and
    # 'INCONSISTENT' (no drop_first) -- consistent with the known collinearity
    # source that made liblinear the required solver (per MASTER_STATUS.md).
    X_gate = pd.get_dummies(merged[GATE_TYPE_COL])

    # Marks-only model: X_complex convention -- marks + complexity_score ONLY,
    # no gate dummies, so the two models stay genuinely non-nested.
    X_marks = merged[MARKS_COLS].copy()

    return X_gate, X_marks


# =============================================================================
# STEP 5 -- Fit, evaluate genome-wide, evaluate subset
# =============================================================================
def evaluate_cell_line(gates_df, binary_df, switching_df, ambiguous_symbols, cell_line_label):
    merged = gates_df.merge(binary_df, on="gene_id", how="inner")

    # Drop rows with undefined expression status before fitting -- NaN here
    # means "expression not resolved," not "unexpressed." Confirmed present
    # in K562 (4,436 rows, ~7.1%) via direct inspection of k562_binary_matrix.csv;
    # applying the same drop to GM12878 defensively in case it has any too.
    n_before = len(merged)
    merged = merged[merged[EXPRESSION_COL].notna()].copy()
    n_dropped = n_before - len(merged)
    if n_dropped > 0:
        print(f"[{cell_line_label}] Dropped {n_dropped} genes ({n_dropped/n_before:.1%}) "
              f"with undefined 'expressed' status before fitting. Not zero-filled.")

    y = merged[EXPRESSION_COL].astype(int)

    subset_mask, subset_n = build_subset_mask(merged, switching_df, ambiguous_symbols, cell_line_label)

    results = []
    for model_type, X in zip(
        ["gate_type", "marks_only"],
        build_feature_matrices(merged),
    ):
        model = LogisticRegression(solver="liblinear", max_iter=1000, random_state=RANDOM_STATE)
        model.fit(X, y)
        probs_all = model.predict_proba(X)[:, 1]

        genome_wide_auc_here = roc_auc_score(y, probs_all)
        expected = GENOME_WIDE_AUC[(cell_line_label, model_type)]
        if abs(genome_wide_auc_here - expected) > 0.01:
            print(f"[{cell_line_label}/{model_type}] NOTE: this script's own genome-wide "
                  f"AUC ({genome_wide_auc_here:.4f}) differs from the locked benchmark "
                  f"({expected:.4f}) by >0.01. Feature construction here likely diverges "
                  f"from whatever produced the locked number -- flag this rather than "
                  f"trusting the subset comparison below at face value.")

        y_subset = y[subset_mask]
        probs_subset = probs_all[subset_mask.values]
        if y_subset.nunique() < 2:
            raise ValueError(
                f"[{cell_line_label}/{model_type}] Subset (n={subset_n}) has only one "
                f"class in y -- AUC undefined. This is a finding to report, not a bug to fix."
            )
        subset_auc = roc_auc_score(y_subset, probs_subset)

        results.append({
            "cell_line": cell_line_label,
            "model_type": model_type,
            "subset_auc": round(subset_auc, 4),
            "genome_wide_auc": expected,
            "subset_n": subset_n,
        })

    return results


# =============================================================================
# MAIN
# =============================================================================
def main():
    gm_gates, k562_gates, crosswalk, ambiguous_symbols = load_gate_assignments()
    gm_binary = load_binary_matrix(GM_BINARY_PATH, "GM12878")
    k562_binary = load_binary_matrix(K562_BINARY_PATH, "K562")
    switching = pd.read_csv(SWITCHING_GENES_PATH, sep="\t")

    all_results = []
    all_results.extend(evaluate_cell_line(gm_gates, gm_binary, switching, ambiguous_symbols, "GM12878"))
    all_results.extend(evaluate_cell_line(k562_gates, k562_binary, switching, ambiguous_symbols, "K562"))

    out_df = pd.DataFrame(all_results)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nWrote {OUTPUT_PATH}")
    print(out_df.to_string(index=False))

    for cell_line in ["GM12878", "K562"]:
        gt = out_df[(out_df.cell_line == cell_line) & (out_df.model_type == "gate_type")].iloc[0]
        mo = out_df[(out_df.cell_line == cell_line) & (out_df.model_type == "marks_only")].iloc[0]
        subset_delta = gt.subset_auc - mo.subset_auc
        genome_delta = gt.genome_wide_auc - mo.genome_wide_auc
        print(f"\n{cell_line}: subset ΔAUC = {subset_delta:.4f} vs genome-wide ΔAUC = "
              f"{genome_delta:.4f} ({'>=' if subset_delta >= genome_delta else '<'} genome-wide)")


if __name__ == "__main__":
    main()