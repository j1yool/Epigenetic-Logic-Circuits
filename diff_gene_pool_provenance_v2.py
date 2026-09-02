"""
diff_gene_pool_provenance_v2.py

SUPERSEDES v1. v1 assumed placeholder column names and a single flat
symbol+biotype diff. Real data changed both:

  1. BioMart columns are: 'Gene stable ID', 'Gene type', 'Gene start (bp)',
     'Gene end (bp)', 'Chromosome/scaffold name', 'Strand', 'Gene name'
     -- not the generic candidates v1 guessed.

  2. Both sources include hundreds of alt-haplotype patch/scaffold contigs
     (e.g. HG1012_PATCH, GL000008.2) in addition to the 25 standard
     chromosomes. These are alternate sequence representations of the same
     genomic regions, not additional distinct genes -- included unfiltered,
     they inflate any gene-count comparison meaninglessly.

  3. Non-protein-coding biotypes (Y_RNA, U6, 5S_rRNA, snoRNAs, etc.) are
     legitimately duplicated hundreds of times across the genome under the
     identical symbol (e.g. 845 rows named "Y_RNA" in BioMart alone). A
     symbol-keyed row-level diff is meaningless for these -- pandas would
     either error or silently pick one arbitrary row per symbol. These are
     instead compared in aggregate, by biotype, not gene-by-gene.

APPROACH
--------
PRIMARY DIFF (gene-level, symbol-keyed): protein_coding genes only, on
standard chromosomes only. This is the scope where HGNC symbols are
~unique and a 1:1 diff is meaningful -- and it's almost certainly the
scope the original "~600-gene delta" in MASTER_STATUS.md was computed on,
since the existing full_genome_gene_pool_v1.csv (20,093 genes, no biotype
column) is consistent with a protein-coding-only pool.

SECONDARY COMPARISON (aggregate, biotype-level): every other biotype,
counted per source, to check whether biotype definitions disagree
structurally beyond the protein-coding set.

Output:
  - gene_pool_diff_buckets_v1.csv     (protein-coding delta, bucketed)
  - gene_pool_diff_detail_v1.csv      (protein-coding delta, per-gene)
  - gene_pool_biotype_counts_v1.csv   (aggregate biotype counts, both sources)
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import glob
import pandas as pd

BIOMART_PATH = "full_genome_gene_pool_biomart_v1_csv.txt"
GTF_PATH = "full_genome_gene_pool_gtf_v1.csv"  # output of extract_genes_from_gtf_v1.py

OUT_BUCKETS = "gene_pool_diff_buckets_v1.csv"
OUT_DETAIL = "gene_pool_diff_detail_v1.csv"
OUT_BIOTYPE_COUNTS = "gene_pool_biotype_counts_v1.csv"

STANDARD_CHROMS = set([str(i) for i in range(1, 23)] + ["X", "Y", "MT"])


def resolve_biomart_path(configured_path: str) -> str:
    """
    Auto-detect the BioMart file in the current working directory if the
    hardcoded name doesn't match, instead of hard-failing on a filename
    guess. Looks for any file with 'biomart' in the name.
    """
    if os.path.exists(configured_path):
        return configured_path
    candidates = [f for f in glob.glob("*") if "biomart" in f.lower() and os.path.isfile(f)]
    if len(candidates) == 1:
        print(f"[AUTO-DETECTED] {configured_path} not found; using {candidates[0]} instead.")
        return candidates[0]
    if len(candidates) > 1:
        sys.exit(
            f"[AMBIGUOUS] {configured_path} not found, and multiple candidate files matched 'biomart': "
            f"{candidates}\nSet BIOMART_PATH explicitly at the top of this script to the correct one."
        )
    sys.exit(
        f"[FILE NOT FOUND] No file named '{configured_path}' and no file containing 'biomart' "
        f"found in the current directory: {os.getcwd()}\n"
        f"Files present: {[f for f in glob.glob('*') if os.path.isfile(f)]}\n"
        f"Set BIOMART_PATH at the top of this script to the exact filename of your BioMart export."
    )

BIOMART_COLUMNS = {
    "symbol": "Gene name",
    "biotype": "Gene type",
    "chr": "Chromosome/scaffold name",
    "start": "Gene start (bp)",
    "end": "Gene end (bp)",
    "strand": "Strand",
    "gene_id": "Gene stable ID",
}
GTF_COLUMNS = {
    "symbol": "gene_symbol",
    "biotype": "biotype",
    "chr": "chr",
    "start": "start",
    "end": "end",
    "strand": "strand",
    "gene_id": "gene_id",
}


def _require_columns(df: pd.DataFrame, colmap: dict, source_label: str):
    missing = [v for v in colmap.values() if v not in df.columns]
    if missing:
        sys.exit(
            f"[SCHEMA MISMATCH] {source_label} is missing expected column(s): {missing}\n"
            f"Actual columns present: {list(df.columns)}\n"
            f"Update BIOMART_COLUMNS / GTF_COLUMNS at the top of this script to match."
        )


def load_and_normalize(path: str, colmap: dict, source_label: str, sep: str = ",") -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"[FILE NOT FOUND] {source_label} path does not exist: {path}")
    df = pd.read_csv(path, sep=sep, dtype=str)
    _require_columns(df, colmap, source_label)

    out = pd.DataFrame({
        "gene_symbol": df[colmap["symbol"]].fillna("").str.strip(),
        "biotype": df[colmap["biotype"]].fillna("").str.strip().str.lower(),
        "chr": df[colmap["chr"]].fillna("").str.strip(),
    })
    return out


def restrict_to_standard_chroms(df: pd.DataFrame) -> tuple:
    """Returns (filtered_df, n_excluded)."""
    mask = df["chr"].isin(STANDARD_CHROMS)
    return df[mask].copy(), int((~mask).sum())


def classify_delta(biomart_pc: pd.DataFrame, gtf_pc: pd.DataFrame) -> pd.DataFrame:
    """
    Core symbol-level diff logic for the protein-coding, standard-chrom,
    named, deduplicated subsets. Factored out so synthetic_self_test can
    exercise it against known toy fixtures.

    Expects both inputs already restricted to: biotype == protein_coding,
    gene_symbol != "", and one row per gene_symbol (deduplicated upstream,
    with the dedup count logged by the caller).
    """
    bm = biomart_pc.set_index("gene_symbol")
    gt = gtf_pc.set_index("gene_symbol")

    bm_only = bm.index.difference(gt.index)
    gt_only = gt.index.difference(bm.index)
    shared = bm.index.intersection(gt.index)

    rows = []
    for sym in bm_only:
        rows.append({"gene_symbol": sym, "biomart_biotype": bm.loc[sym, "biotype"],
                     "gtf_biotype": None, "bucket": "biomart_exclusive_no_gtf_match"})
    for sym in gt_only:
        rows.append({"gene_symbol": sym, "biomart_biotype": None,
                     "gtf_biotype": gt.loc[sym, "biotype"], "bucket": "gtf_exclusive_no_biomart_match"})
    for sym in shared:
        bm_bt, gt_bt = bm.loc[sym, "biotype"], gt.loc[sym, "biotype"]
        if bm_bt == gt_bt:
            continue
        rows.append({"gene_symbol": sym, "biomart_biotype": bm_bt, "gtf_biotype": gt_bt,
                     "bucket": "biotype_mismatch_despite_shared_symbol"})

    return pd.DataFrame(rows, columns=["gene_symbol", "biomart_biotype", "gtf_biotype", "bucket"])


def synthetic_self_test():
    """
    Toy fixture: 8 clean protein-coding matches, 2 exclusive-to-BioMart,
    2 exclusive-to-GTF, 1 biotype-mismatch-despite-shared-symbol case.
    Asserts exact expected bucket counts.
    """
    bm_rows = [{"gene_symbol": f"GENE{i}", "biotype": "protein_coding"} for i in range(1, 9)]
    gt_rows = [{"gene_symbol": f"GENE{i}", "biotype": "protein_coding"} for i in range(1, 9)]

    bm_rows += [{"gene_symbol": "BMONLY1", "biotype": "protein_coding"},
                {"gene_symbol": "BMONLY2", "biotype": "protein_coding"}]
    gt_rows += [{"gene_symbol": "GTONLY1", "biotype": "protein_coding"},
                {"gene_symbol": "GTONLY2", "biotype": "protein_coding"}]

    bm_rows.append({"gene_symbol": "MISMATCH1", "biotype": "protein_coding"})
    gt_rows.append({"gene_symbol": "MISMATCH1", "biotype": "polymorphic_pseudogene"})

    bm_df = pd.DataFrame(bm_rows)
    gt_df = pd.DataFrame(gt_rows)

    delta = classify_delta(bm_df, gt_df)
    counts = delta["bucket"].value_counts().to_dict()

    expected = {
        "biomart_exclusive_no_gtf_match": 2,
        "gtf_exclusive_no_biomart_match": 2,
        "biotype_mismatch_despite_shared_symbol": 1,
    }
    assert counts == expected, f"[SELF-TEST FAILED]\nExpected: {expected}\nGot: {counts}"
    assert len(delta) == 5, f"[SELF-TEST FAILED] Expected 5 delta rows, got {len(delta)}"

    print("[SELF-TEST PASSED] classify_delta correctly bucketed synthetic protein-coding fixture.")
    print(f"  Bucket counts: {counts}")


def main():
    synthetic_self_test()

    print("\nLoading and normalizing both sources...")
    biomart_path_resolved = resolve_biomart_path(BIOMART_PATH)
    biomart_all = load_and_normalize(biomart_path_resolved, BIOMART_COLUMNS, "BioMart pool", sep=",")
    gtf_all = load_and_normalize(GTF_PATH, GTF_COLUMNS, "GTF pool", sep=",")

    print(f"  BioMart: {len(biomart_all)} total rows (all biotypes, all chroms/scaffolds)")
    print(f"  GTF:     {len(gtf_all)} total rows (all biotypes, all chroms/scaffolds)")

    # --- Secondary comparison: aggregate biotype counts across BOTH sources, unrestricted ---
    bm_biotype_counts = biomart_all["biotype"].value_counts().rename("biomart_count")
    gt_biotype_counts = gtf_all["biotype"].value_counts().rename("gtf_count")
    biotype_compare = pd.concat([bm_biotype_counts, gt_biotype_counts], axis=1).fillna(0).astype(int)
    biotype_compare["abs_diff"] = (biotype_compare["biomart_count"] - biotype_compare["gtf_count"]).abs()
    biotype_compare = biotype_compare.sort_values("abs_diff", ascending=False)
    biotype_compare.to_csv(OUT_BIOTYPE_COUNTS)
    print(f"\nWrote {OUT_BIOTYPE_COUNTS} (all biotypes, unrestricted -- for structural biotype-definition review)")
    print("\nTop 10 biotypes by absolute count disagreement between sources:")
    print(biotype_compare.head(10).to_string())

    # --- Primary diff: protein_coding, standard chromosomes only ---
    print("\n--- Primary diff scope: protein_coding, standard chromosomes 1-22/X/Y/MT ---")

    biomart_std, bm_excluded = restrict_to_standard_chroms(biomart_all)
    gtf_std, gt_excluded = restrict_to_standard_chroms(gtf_all)
    print(f"BioMart: excluded {bm_excluded} rows on patches/scaffolds, {len(biomart_std)} remain")
    print(f"GTF:     excluded {gt_excluded} rows on patches/scaffolds, {len(gtf_std)} remain")

    biomart_pc = biomart_std[biomart_std["biotype"] == "protein_coding"].copy()
    gtf_pc = gtf_std[gtf_std["biotype"] == "protein_coding"].copy()
    print(f"BioMart protein_coding, std-chrom: {len(biomart_pc)} rows")
    print(f"GTF protein_coding, std-chrom:     {len(gtf_pc)} rows")

    bm_missing_symbol = int((biomart_pc["gene_symbol"] == "").sum())
    gt_missing_symbol = int((gtf_pc["gene_symbol"] == "").sum())
    print(f"  BioMart protein_coding rows with EMPTY gene symbol: {bm_missing_symbol}")
    print(f"  GTF protein_coding rows with EMPTY gene symbol:     {gt_missing_symbol}")

    biomart_pc = biomart_pc[biomart_pc["gene_symbol"] != ""]
    gtf_pc = gtf_pc[gtf_pc["gene_symbol"] != ""]

    bm_dupe_count = int(biomart_pc["gene_symbol"].duplicated().sum())
    gt_dupe_count = int(gtf_pc["gene_symbol"].duplicated().sum())
    print(f"  BioMart duplicate protein-coding symbols (kept first, dropped rest): {bm_dupe_count}")
    print(f"  GTF duplicate protein-coding symbols (kept first, dropped rest):     {gt_dupe_count}")
    if bm_dupe_count > 0:
        dupes = biomart_pc[biomart_pc["gene_symbol"].duplicated(keep=False)]["gene_symbol"].unique()
        print(f"    Sample BioMart dupe symbols (check if PAR-region genes, expected): {list(dupes[:10])}")

    biomart_pc = biomart_pc.drop_duplicates(subset="gene_symbol")
    gtf_pc = gtf_pc.drop_duplicates(subset="gene_symbol")

    delta = classify_delta(biomart_pc, gtf_pc)

    if delta.empty:
        print("\nNo delta found in the protein_coding/standard-chrom/named/deduplicated scope. "
              "This contradicts the ~600-gene figure in MASTER_STATUS.md -- re-check paths/filters.")
        sys.exit(1)

    bucket_summary = (
        delta["bucket"].value_counts().rename_axis("bucket").reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    print(f"\nTotal primary-scope delta size: {len(delta)} genes")
    print("\nBucket breakdown:")
    print(bucket_summary.to_string(index=False))

    bucket_summary.to_csv(OUT_BUCKETS, index=False)
    delta.to_csv(OUT_DETAIL, index=False)
    print(f"\nWrote {OUT_BUCKETS}")
    print(f"Wrote {OUT_DETAIL}")

    print("\n--- Interpretation guide ---")
    print(f"BioMart empty-symbol protein-coding rows: {bm_missing_symbol}")
    print("If this number is close to the ~600-gene delta already logged in MASTER_STATUS.md, "
          "the delta is substantially explained by BioMart protein-coding genes lacking an "
          "assigned Gene name (these genes exist and are correctly typed, they just have no "
          "symbol to match on) -- a data-completeness issue in the BioMart pull, not a "
          "biotype-definition disagreement between the two sources. If so, the GTF source "
          "(which carries gene_name directly from the GENCODE annotation for nearly all "
          "protein-coding genes) is the stronger choice on BOTH reproducibility grounds "
          "(static, versioned file) AND symbol-completeness grounds -- confirm this against "
          "the printed gt_missing_symbol count above before locking the decision in Block 2.")


if __name__ == "__main__":
    main()