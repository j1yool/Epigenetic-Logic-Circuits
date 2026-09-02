"""
reshape_encode_rnaseq_v1.py

Day 55 (late). Converts the 4 raw ENCODE RSEM gene-quantification TSVs
(2 replicates x 2 cell lines, ENCODE4 v1.2.1 GRCh38 V29 pipeline) into
the wide gene_id/rep1/rep2 format switching_gene_expression_variability_v1.py
expects at GM12878_RNASEQ_PATH / K562_RNASEQ_PATH.

SOURCE FILES (confirmed real, schema-inspected before writing this):
  GM12878: ENCFF935TRK.tsv (isogenic rep 2), ENCFF700JWD.tsv (isogenic rep 1)
           -- experiment ENCSR843RJV
  K562:    ENCFF628SMT.tsv (isogenic rep 2), ENCFF472HFI.tsv (isogenic rep 1)
           -- experiment ENCSR000AEP
Each is a standard RSEM gene-quantification TSV: 59,526 rows, 17 columns
(gene_id, transcript_id(s), length, effective_length, expected_count,
TPM, FPKM, ...). Confirmed real TPM values (sums to ~1e6 per file).

CLEANING STEPS (each named, each checked against real data before being
applied -- not assumed):
  1. 649 rows have non-ENSG numeric gene_id values (a small non-standard
     annotation set bundled into the V29 GTF, unrelated to the protein-
     coding/lncRNA gene set the gate-calling pipeline uses). Dropped.
  2. 746 rows are ERCC spike-in controls (gene_id starting
     "gSpikein_ERCC-") plus one phiX control row. These are synthetic
     RNA standards used for pipeline QC, not real genes. Dropped.
  3. Real gene_id values carry a GENCODE version suffix
     (ENSG00000000003.14) that gate_assignments_named.tsv /
     k562_gate_assignments_named.tsv do not use (ENSG00000000003).
     Version suffix stripped for merge compatibility.
  4. 45 genes are pseudoautosomal-region (PAR) genes, double-annotated
     in the GTF once on X and once as a "..._PAR_Y" duplicate on Y.
     Confirmed: every "_PAR_Y" row has TPM=0 in every file checked --
     this is the standard RSEM/GENCODE PAR convention (reads assigned
     to the X copy only). The "_PAR_Y" row is dropped BEFORE version-
     stripping, not averaged or summed with its X counterpart, so no
     real signal is altered.

Output columns are named rep1/rep2 (TPM values) to match REP_COL_PREFIX
in switching_gene_expression_variability_v1.py -- gene_id is
UNversioned to match the gate-calling files it will be merged against.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import pandas as pd

GENE_COL = "gene_id"
TPM_COL = "TPM"
REQUIRED_COLS = [GENE_COL, TPM_COL]

GM12878_REP_PATHS = ["data/ENCFF935TRK.tsv", "data/ENCFF700JWD.tsv"]
K562_REP_PATHS = ["data/ENCFF628SMT.tsv", "data/ENCFF472HFI.tsv"]

GM12878_OUT_PATH = "data/encode_rnaseq_gm12878_replicates_v1.csv"
K562_OUT_PATH = "data/encode_rnaseq_k562_replicates_v1.csv"


def load_and_validate(path, required_cols):
    if not os.path.exists(path):
        sys.exit(f"FATAL: input file not found at {path}. Not proceeding.")
    df = pd.read_csv(path, sep="\t")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        sys.exit(
            f"FATAL: expected column(s) {missing} not found in {path}.\n"
            f"Actual columns present: {list(df.columns)}\n"
            f"Not proceeding with a guessed schema."
        )
    return df


def clean_gene_quant(df, source_label):
    """Applies the four named cleaning steps in the module docstring, in
    order, and prints a count at each step so drops are auditable."""
    n0 = len(df)

    ensg_mask = df[GENE_COL].astype(str).str.startswith("ENSG")
    df = df[ensg_mask].copy()
    n1 = len(df)
    print(f"  [{source_label}] dropped {n0 - n1} non-ENSG rows -> {n1} remain")

    par_y_mask = df[GENE_COL].astype(str).str.endswith("_PAR_Y")
    dropped_par = df[par_y_mask]
    if len(dropped_par) > 0 and not (dropped_par[TPM_COL] == 0).all():
        nonzero = dropped_par[dropped_par[TPM_COL] != 0]
        sys.exit(
            f"FATAL [{source_label}]: found {len(nonzero)} '_PAR_Y' row(s) with "
            f"NONZERO TPM -- this violates the assumption this cleaning step "
            f"relies on (that PAR_Y duplicates always carry zero signal). "
            f"Example: {nonzero[[GENE_COL, TPM_COL]].head(3).to_dict('records')}. "
            f"Not proceeding -- this needs a decision, not a silent drop."
        )
    df = df[~par_y_mask].copy()
    n2 = len(df)
    print(f"  [{source_label}] dropped {n1 - n2} '_PAR_Y' duplicate rows "
          f"(all confirmed TPM=0) -> {n2} remain")

    df[GENE_COL] = df[GENE_COL].str.split(".").str[0]
    n_dup = df[GENE_COL].duplicated().sum()
    if n_dup > 0:
        sys.exit(
            f"FATAL [{source_label}]: {n_dup} duplicate gene_id(s) remain after "
            f"version-stripping and PAR_Y removal -- the known PAR_Y cause does "
            f"not fully explain this. Not proceeding with an ambiguous merge key."
        )
    print(f"  [{source_label}] version-stripped, {df[GENE_COL].nunique()} unique gene_id, no duplicates\n")

    return df[[GENE_COL, TPM_COL]]


def build_wide_replicate_table(rep_paths, cell_line_label):
    print(f"Processing {cell_line_label}:")
    cleaned = []
    for i, path in enumerate(rep_paths, start=1):
        raw = load_and_validate(path, REQUIRED_COLS)
        c = clean_gene_quant(raw, f"{cell_line_label} rep{i} ({os.path.basename(path)})")
        c = c.rename(columns={TPM_COL: f"rep{i}"})
        cleaned.append(c)

    wide = cleaned[0]
    for c in cleaned[1:]:
        wide = wide.merge(c, on=GENE_COL, how="inner")

    if wide[GENE_COL].duplicated().sum() > 0:
        sys.exit(f"FATAL: duplicate gene_id in final {cell_line_label} wide table. Not proceeding.")

    print(f"{cell_line_label} wide table: {len(wide)} genes x {len(rep_paths)} replicates\n")
    return wide


def run():
    gm_wide = build_wide_replicate_table(GM12878_REP_PATHS, "GM12878")
    gm_wide.to_csv(GM12878_OUT_PATH, index=False)
    print(f"Wrote {GM12878_OUT_PATH}\n")

    k562_wide = build_wide_replicate_table(K562_REP_PATHS, "K562")
    k562_wide.to_csv(K562_OUT_PATH, index=False)
    print(f"Wrote {K562_OUT_PATH}\n")

    print("Reshape complete. Both files are now in the wide gene_id/rep1/rep2 "
          "format switching_gene_expression_variability_v1.py expects at "
          "GM12878_RNASEQ_PATH / K562_RNASEQ_PATH. Re-run that script now.")


if __name__ == "__main__":
    run()