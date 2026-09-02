"""
logic_switching_gene_finder.py

Purpose: Draft candidate list of genes whose assigned logic-gate type differs
between GM12878 (baseline) and K562 (CML comparator). Groundwork only, per Day
30 Block 4 -- no pathway enrichment or biological interpretation here. That's
Days 91-120 territory once the AML extension data is in.

Input:  two SEPARATE per-cell-line files (each one row per gene):
    data/gate_assignments_named.tsv            (GM12878)
    data/k562/k562_gate_assignments_named.tsv  (K562)
Output: switching_genes_candidate_list.tsv

CONFIRMED SCHEMA (from actual files, via check_gate_types.py):
    GM12878 file: gene_id, gene_name, gate_type, complexity_score
    K562 file:    gene_id,            gate_type, complexity_score
                  (no gene_name column in the K562 file)

CONFIRMED gate_type VOCABULARY (both files):
    SIMPLE_AND, SIMPLE_OR                  -> simple
    BIVALENT, COMPLEX, INCONSISTENT, REPRESSED -> complex/other

CONFIRMED: gate_type has substantial missingness --
    GM12878: 34,163 rows with missing gate_type
    K562:    37,763 rows with missing gate_type
A gene with a missing gate_type call in EITHER cell line is excluded from
the switching comparison (same data-completeness principle already applied
to genes missing entirely from one file) rather than being forced into a
switch_category based on an undefined value.

Run this script from the project root (LOGIC CIRCUITS/) so the relative
paths below resolve correctly.

Windows environment constraint: threading env vars set before numeric imports.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

GM12878_PATH = os.path.join("data", "gate_assignments_named.tsv")
K562_PATH = os.path.join("data", "k562", "k562_gate_assignments_named.tsv")

OUTPUT_PATH = "switching_genes_candidate_list.tsv"
DROPPED_MISSING_GATE_TYPE_PATH = "switching_genes_dropped_missing_gate_type.tsv"

GENE_ID_COL = "gene_id"
GATE_TYPE_COL = "gate_type"
GENE_NAME_COL = "gene_name"  # optional -- present in GM12878 file only

BASELINE_LABEL = "GM12878"
COMPARATOR_LABEL = "K562"

# Confirmed against actual data (check_gate_types.py output).
SIMPLE_GATE_TYPES = {"SIMPLE_AND", "SIMPLE_OR"}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_one_cell_line(path, label):
    if not os.path.exists(path):
        print(f"ERROR: {path} not found (expected {label} gate assignments).", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path, sep="\t")

    required_cols = (GENE_ID_COL, GATE_TYPE_COL)
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"Expected REQUIRED columns {missing_required} not found in {path} "
            f"for {label}. Actual columns: {list(df.columns)}. Fix CONFIG block."
        )

    n_dupes = df[GENE_ID_COL].duplicated().sum()
    if n_dupes > 0:
        print(
            f"WARNING: {n_dupes} duplicate {GENE_ID_COL} entries in {path} "
            f"(keeping first occurrence).",
            file=sys.stderr,
        )
        df = df.drop_duplicates(subset=GENE_ID_COL, keep="first")

    n_missing_gate = df[GATE_TYPE_COL].isna().sum()
    print(f"{label}: {n_missing_gate} of {len(df)} rows have missing gate_type.", file=sys.stderr)

    keep_cols = [GENE_ID_COL, GATE_TYPE_COL]
    has_gene_name = GENE_NAME_COL in df.columns
    if has_gene_name:
        keep_cols.append(GENE_NAME_COL)

    out = df[keep_cols].rename(columns={GATE_TYPE_COL: f"{label}_gate_type"})
    if has_gene_name:
        out = out.rename(columns={GENE_NAME_COL: f"{label}_gene_name"})
    return out


def build_merged(gm_df, k562_df):
    merged = pd.merge(gm_df, k562_df, on=GENE_ID_COL, how="inner")

    n_gm_only = len(gm_df) - len(merged)
    n_k562_only = len(k562_df) - len(merged)
    if n_gm_only > 0 or n_k562_only > 0:
        print(
            f"NOTE: {n_gm_only} gene_ids present only in GM12878 file, "
            f"{n_k562_only} gene_ids present only in K562 file -- dropped "
            f"(missing entirely from one file, not a switching candidate). "
            f"{len(merged)} gene_ids retained with an entry in both files.",
            file=sys.stderr,
        )

    gm_name_col = f"{BASELINE_LABEL}_gene_name"
    k562_name_col = f"{COMPARATOR_LABEL}_gene_name"
    if gm_name_col in merged.columns:
        merged["gene_name"] = merged[gm_name_col]
    elif k562_name_col in merged.columns:
        merged["gene_name"] = merged[k562_name_col]
    else:
        merged["gene_name"] = merged[GENE_ID_COL]

    return merged


def split_on_gate_type_completeness(merged_df):
    """
    Separate genes with a gate_type call in BOTH cell lines (usable for the
    switching comparison) from genes missing a call in either (excluded --
    data-completeness gap, not a switch).
    """
    gm_col = f"{BASELINE_LABEL}_gate_type"
    k562_col = f"{COMPARATOR_LABEL}_gate_type"

    has_both = merged_df[gm_col].notna() & merged_df[k562_col].notna()
    usable = merged_df[has_both].copy()
    dropped = merged_df[~has_both].copy()

    print(
        f"NOTE: of {len(merged_df)} genes with an entry in both files, "
        f"{len(usable)} have a gate_type call in both cell lines and are "
        f"usable for switching comparison; {len(dropped)} are missing "
        f"gate_type in at least one cell line and are excluded.",
        file=sys.stderr,
    )

    return usable, dropped


def classify_switch(gm_type, k562_type):
    """
    Returns one of:
      "no change"
      "{A}->{B}"          - both simple, different type
      "simple->complex"
      "complex->simple"
      "complex->complex"  - both non-simple, distinct labels (or same label
                             -- caught earlier by "no change" if identical)
    """
    if gm_type == k562_type:
        return "no change"

    gm_simple = gm_type in SIMPLE_GATE_TYPES
    k562_simple = k562_type in SIMPLE_GATE_TYPES

    if gm_simple and k562_simple:
        return f"{gm_type}->{k562_type}"
    elif gm_simple and not k562_simple:
        return "simple->complex"
    elif not gm_simple and k562_simple:
        return "complex->simple"
    else:
        return "complex->complex"


def build_candidate_list(usable_df):
    gm_col = f"{BASELINE_LABEL}_gate_type"
    k562_col = f"{COMPARATOR_LABEL}_gate_type"

    usable_df["switch_category"] = usable_df.apply(
        lambda row: classify_switch(row[gm_col], row[k562_col]), axis=1
    )

    return usable_df[[GENE_ID_COL, "gene_name", gm_col, k562_col, "switch_category"]]


def sanity_check(out_df):
    counts = out_df["switch_category"].value_counts()
    total = len(out_df)
    no_change = counts.get("no change", 0)

    print(f"\nTotal genes usable for switching comparison: {total}")
    print("Switch category distribution:")
    for category, count in counts.items():
        pct = 100 * count / total if total else 0
        print(f"    {category:20s} {count:6d}  ({pct:.1f}%)")

    n_categories = len(counts)
    if total > 0 and no_change == total:
        print(
            "\nWARNING: 100% 'no change'. Check merge logic before trusting this list.",
            file=sys.stderr,
        )
    elif n_categories <= 2:
        print(
            f"\nWARNING: only {n_categories} distinct switch_category values "
            f"appeared. If you expect finer-grained simple-type transitions "
            f"(e.g. SIMPLE_AND->SIMPLE_OR) and don't see them, check that "
            f"SIMPLE_GATE_TYPES in CONFIG still matches your actual labels.",
            file=sys.stderr,
        )


def main():
    gm_df = load_one_cell_line(GM12878_PATH, BASELINE_LABEL)
    k562_df = load_one_cell_line(K562_PATH, COMPARATOR_LABEL)

    merged = build_merged(gm_df, k562_df)
    usable, dropped_missing = split_on_gate_type_completeness(merged)

    out_df = build_candidate_list(usable)
    out_df.to_csv(OUTPUT_PATH, sep="\t", index=False)
    dropped_missing.to_csv(DROPPED_MISSING_GATE_TYPE_PATH, sep="\t", index=False)

    print(f"\nWrote {len(out_df)} genes to {OUTPUT_PATH}")
    print(f"Wrote {len(dropped_missing)} excluded (missing gate_type) genes to {DROPPED_MISSING_GATE_TYPE_PATH}")
    sanity_check(out_df)


if __name__ == "__main__":
    main()