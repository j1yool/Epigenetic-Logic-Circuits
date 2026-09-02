"""
extract_switching_genes_v2.py

SUPERSEDES extract_switching_genes_v1.py. Root cause of the persistent NaN
gate_type failure: gate_assignments.tsv has genuinely EMPTY gate_type cells
on disk (confirmed via raw text inspection -- 'ENSG00000000003\\t\\t-1.0...').
This happened upstream of today: some prior script read the file with
pandas' default NA handling (which treats the literal string "NULL" as
missing), then wrote it back out, and pandas' to_csv() serializes NaN as an
empty string. The corruption is permanent in that file's gate_type column --
no read-time parameter can recover it.

FIX: do not trust gate_assignments.tsv / k562_gate_assignments_named.tsv for
gate_type at all. Regenerate gate_type and complexity_score directly from
binary_matrix.csv / k562_binary_matrix.csv (source of truth, already
confirmed intact) using the exact call_gate()/compute_complexity_score()
logic from the Day 5 / Day 14 scripts. Only gene_name annotation is borrowed
from the existing TSVs (that column was not corrupted).
"""

import sys
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GM12878_BINARY_MATRIX = Path(r"C:\Users\jamoo\Downloads\LOGIC CIRCUITS\data\binary_matrix.csv")
K562_BINARY_MATRIX = Path(r"C:\Users\jamoo\Downloads\LOGIC CIRCUITS\data\k562\k562_binary_matrix.csv")

# Only used for gene_name annotation -- gate_type/complexity_score from these
# files are IGNORED (regenerated fresh instead).
GM12878_ANNOTATION_SOURCE = Path(r"C:\Users\jamoo\Downloads\LOGIC CIRCUITS\data\gate_assignments.tsv")
K562_ANNOTATION_SOURCE = Path(r"C:\Users\jamoo\Downloads\LOGIC CIRCUITS\data\k562\k562_gate_assignments_named.tsv")

OUT_GM12878_LIST = Path("switching_genes_gm12878_v1.txt")
OUT_K562_LIST = Path("switching_genes_k562_v1.txt")
OUT_MERGED = Path("switching_genes_merged_v1.tsv")
OUT_REGEN_GM12878 = Path("gate_assignments_regenerated_v1.tsv")
OUT_REGEN_K562 = Path("k562_gate_assignments_regenerated_v1.tsv")
OUT_LOG = Path("extract_switching_genes_v2_findings.md")

VALID_GATE_TYPES = {
    "NULL", "BIVALENT", "REPRESSED", "POISED",
    "SIMPLE_AND", "SIMPLE_OR", "INCONSISTENT", "COMPLEX",
}


def fatal(msg: str) -> None:
    print("\n" + "=" * 70)
    print("FATAL:", msg)
    print("=" * 70)
    sys.exit(1)


# ---------------------------------------------------------------------------
# call_gate / compute_complexity_score -- copied verbatim from Day 5 / Day 14
# scripts (identical logic in both). This is the source-of-truth computation.
# ---------------------------------------------------------------------------
def call_gate(row):
    active = row.get('H3K4me3', 0)
    enhancer = row.get('H3K27ac', 0)
    poised = row.get('H3K4me1', 0)
    repressive1 = row.get('H3K27me3', 0)
    repressive2 = row.get('H3K9me3', 0)
    expressed = row.get('expressed', 0)

    any_active = active or enhancer
    any_repressive = repressive1 or repressive2
    active_count = active + enhancer + poised
    repressive_count = repressive1 + repressive2

    if not any_active and not any_repressive:
        return 'NULL'
    if active and repressive1:
        return 'BIVALENT'
    if any_repressive and not any_active:
        return 'REPRESSED'
    if poised and not any_active and not any_repressive:
        return 'POISED'
    if active and enhancer and not any_repressive:
        if expressed:
            return 'SIMPLE_AND'
        else:
            return 'INCONSISTENT'
    if (active or enhancer) and not (active and enhancer) and not any_repressive:
        if expressed:
            return 'SIMPLE_OR'
        else:
            return 'INCONSISTENT'
    if active_count + repressive_count >= 3:
        return 'COMPLEX'
    if expressed and not any_active:
        return 'INCONSISTENT'
    return 'COMPLEX'


def compute_complexity_score(row):
    active = row.get('H3K4me3', 0) + row.get('H3K27ac', 0)
    repressive = row.get('H3K27me3', 0) + row.get('H3K9me3', 0)
    bivalent_conflict = 1 if (row.get('H3K4me3', 0) and row.get('H3K27me3', 0)) else 0
    gate = call_gate(row)
    clean_logic = 1 if gate in ('SIMPLE_AND', 'SIMPLE_OR', 'REPRESSED', 'NULL') else 0
    return active + (2 * repressive) + bivalent_conflict - clean_logic


EXCLUSION_STATS = {}


def regenerate_gate_calls(binary_matrix_path: Path, label: str) -> pd.DataFrame:
    if not binary_matrix_path.exists():
        fatal(f"{label}: binary matrix not found at {binary_matrix_path}")

    df = pd.read_csv(binary_matrix_path, index_col=0)

    mark_cols = ['H3K4me3', 'H3K27ac', 'H3K4me1', 'H3K27me3', 'H3K9me3', 'expressed']
    missing_cols = set(mark_cols) - set(df.columns)
    if missing_cols:
        fatal(f"{label}: binary matrix missing expected column(s) {missing_cols}. "
              f"Found: {list(df.columns)}")

    # K562-specific convention, matching the Day 14 script exactly: mark
    # columns (not 'expressed') are zero-filled before gate-calling, because
    # K562 ChIP-seq peak calls have expected gaps. GM12878's Day 5 script has
    # no equivalent step -- this NaN-fill is intentionally applied only when
    # NaNs are actually present, so it's a no-op for GM12878.
    peak_mark_cols = ['H3K4me3', 'H3K27ac', 'H3K4me1', 'H3K27me3', 'H3K9me3']
    present_peak_cols = [c for c in peak_mark_cols if c in df.columns]
    n_nan_marks_only = df[present_peak_cols].isna().sum().sum()
    if n_nan_marks_only > 0:
        print(f"{label}: filling {n_nan_marks_only} NaN value(s) in mark columns "
              f"{present_peak_cols} with 0 (matches Day 14 K562 script convention).")
        df[present_peak_cols] = df[present_peak_cols].fillna(0)

    # Guard: call_gate() uses row.get(col, 0), whose default only fires on a
    # MISSING KEY, not a NaN VALUE. A NaN cell would silently poison every
    # boolean/arithmetic expression in call_gate(). Check explicitly -- this
    # now only catches NaN in 'expressed' (not covered by the fill above) or
    # any genuinely unexpected remaining NaN.
    n_nan_input = df[mark_cols].isna().sum().sum()
    if n_nan_input > 0:
        n_nan_expressed = df["expressed"].isna().sum() if "expressed" in df.columns else 0
        n_nan_marks_remaining = n_nan_input - n_nan_expressed
        if n_nan_marks_remaining > 0:
            fatal(f"{label}: {n_nan_marks_remaining} NaN value(s) remain in mark columns "
                  f"after fill -- this should be impossible, stopping to re-inspect.")

        # No established convention exists for filling 'expressed' NaN in the
        # Day 5/Day 14 scripts, and this project's locked missing-data policy
        # (reweighting_protocol_v1.md) explicitly rejects zero-fill as a
        # general strategy elsewhere in this research program. Assuming
        # "not expressed" for genes with genuinely unknown expression status
        # would silently bias their gate call. Drop and flag instead of
        # guessing.
        excluded_ids = df.loc[df["expressed"].isna()].index.tolist()
        print(f"{label}: excluding {len(excluded_ids)} gene(s) with NaN 'expressed' status "
              f"(unknown expression, no zero-fill convention established -- flagged, not guessed).")
        df = df.loc[df["expressed"].notna()].copy()
        EXCLUSION_STATS[label] = excluded_ids

    gates = df.apply(call_gate, axis=1)
    scores = df.apply(compute_complexity_score, axis=1)

    result = pd.DataFrame({
        'gene_id': df.index,
        'gate_type': gates.values,
        'complexity_score': scores.values,
    })

    bad_types = set(result["gate_type"].unique()) - VALID_GATE_TYPES
    if bad_types:
        fatal(f"{label}: regeneration produced unrecognized gate_type value(s) {bad_types}")

    n_nan_out = result["gate_type"].isna().sum()
    if n_nan_out > 0:
        fatal(f"{label}: regeneration STILL produced {n_nan_out} NaN gate_type value(s) -- "
              f"this would mean call_gate() itself has a code path returning None, "
              f"which contradicts its source. Stop and re-inspect call_gate() directly.")

    print(f"{label}: regenerated {len(result)} gate calls from source binary matrix, "
          f"0 NaN, all values in expected vocabulary.")
    print(f"{label} gate distribution:\n{result['gate_type'].value_counts().to_string()}")
    return result


def attach_gene_name_annotation(regenerated: pd.DataFrame, annotation_path: Path, label: str) -> pd.DataFrame:
    if not annotation_path.exists():
        print(f"WARNING: {label} annotation source not found at {annotation_path}; "
              f"proceeding without gene_name annotation.")
        return regenerated

    # keep_default_na=False here too, defensively -- even though we're not
    # using this file's gate_type column, no reason to risk the same
    # collision corrupting gene_name if a gene were ever literally named
    # "NA" or similar.
    ann = pd.read_csv(annotation_path, sep="\t", keep_default_na=False, na_values=[""])
    keep_cols = [c for c in ["gene_id", "gene_id_clean", "gene_name"] if c in ann.columns]
    if "gene_id" not in keep_cols:
        print(f"WARNING: {label} annotation source has no gene_id column; skipping annotation merge.")
        return regenerated

    ann_slim = ann[keep_cols].drop_duplicates(subset="gene_id")
    merged = regenerated.merge(ann_slim, on="gene_id", how="left")
    return merged


# ---------------------------------------------------------------------------
# Switching-gene derivation (unchanged logic from v1, validated Day 47)
# ---------------------------------------------------------------------------
def derive_switchers(gm_df: pd.DataFrame, k562_df: pd.DataFrame) -> pd.DataFrame:
    merged = gm_df.merge(
        k562_df, on="gene_id", how="inner", suffixes=("_gm12878", "_k562")
    )
    switchers = merged[merged["gate_type_gm12878"] != merged["gate_type_k562"]].copy()

    def origin(row):
        gm_is_and = row["gate_type_gm12878"] == "SIMPLE_AND"
        k562_is_and = row["gate_type_k562"] == "SIMPLE_AND"
        if gm_is_and and not k562_is_and:
            return "GM12878"
        elif k562_is_and and not gm_is_and:
            return "K562"
        else:
            return "NEITHER"

    switchers["and_origin"] = switchers.apply(origin, axis=1)
    return switchers[["gene_id", "gate_type_gm12878", "gate_type_k562", "and_origin"]]


def synthetic_self_test() -> None:
    print("Running synthetic self-test (call_gate + derive_switchers)...")

    # call_gate() correctness spot-check
    assert call_gate({'H3K4me3': 0, 'H3K27ac': 0, 'H3K4me1': 0, 'H3K27me3': 0,
                       'H3K9me3': 0, 'expressed': 0}) == 'NULL'
    assert call_gate({'H3K4me3': 1, 'H3K27ac': 1, 'H3K4me1': 0, 'H3K27me3': 0,
                       'H3K9me3': 0, 'expressed': 1}) == 'SIMPLE_AND'
    assert call_gate({'H3K4me3': 1, 'H3K27ac': 1, 'H3K4me1': 0, 'H3K27me3': 0,
                       'H3K9me3': 0, 'expressed': 0}) == 'INCONSISTENT'

    gm_synth = pd.DataFrame({
        "gene_id": ["GENEA", "GENEB", "GENEC", "GENED"],
        "gate_type": ["SIMPLE_AND", "INCONSISTENT", "SIMPLE_AND", "BIVALENT"],
    })
    k562_synth = pd.DataFrame({
        "gene_id": ["GENEA", "GENEB", "GENEC", "GENED"],
        "gate_type": ["INCONSISTENT", "SIMPLE_AND", "SIMPLE_AND", "INCONSISTENT"],
    })
    result = derive_switchers(gm_synth, k562_synth)
    got_genes = set(result["gene_id"])
    assert got_genes == {"GENEA", "GENEB", "GENED"}
    origin_map = dict(zip(result["gene_id"], result["and_origin"]))
    assert origin_map["GENEA"] == "GM12878"
    assert origin_map["GENEB"] == "K562"
    assert origin_map["GENED"] == "NEITHER"

    print("Synthetic self-test PASSED.\n")


def main():
    synthetic_self_test()

    gm_regen = regenerate_gate_calls(GM12878_BINARY_MATRIX, "GM12878")
    k562_regen = regenerate_gate_calls(K562_BINARY_MATRIX, "K562")

    gm_full = attach_gene_name_annotation(gm_regen, GM12878_ANNOTATION_SOURCE, "GM12878")
    k562_full = attach_gene_name_annotation(k562_regen, K562_ANNOTATION_SOURCE, "K562")

    gm_full.to_csv(OUT_REGEN_GM12878, sep="\t", index=False)
    k562_full.to_csv(OUT_REGEN_K562, sep="\t", index=False)
    print(f"\nWrote clean regenerated files: {OUT_REGEN_GM12878}, {OUT_REGEN_K562}")
    print("These should REPLACE gate_assignments.tsv / k562_gate_assignments_named.tsv "
          "as the trusted source going forward -- the originals have a corrupted "
          "gate_type column and should not be read directly again.")

    switchers = derive_switchers(gm_full[["gene_id", "gate_type"]], k562_full[["gene_id", "gate_type"]])

    n_total = len(switchers)
    n_and_inc = len(switchers[
        ((switchers["gate_type_gm12878"] == "SIMPLE_AND") & (switchers["gate_type_k562"] == "INCONSISTENT")) |
        ((switchers["gate_type_gm12878"] == "INCONSISTENT") & (switchers["gate_type_k562"] == "SIMPLE_AND"))
    ])

    print(f"\nTotal switching genes: {n_total}")
    print(f"AND<->INC switchers: {n_and_inc}")
    print("(Day 38 doc cited n=6,962 / n=3,502 -- those figures were computed from data "
          "that may have had the same corruption upstream. Do NOT treat a mismatch here "
          "as this script being wrong; treat it as the Day 38 numbers needing review.)")

    gene_list = sorted(switchers["gene_id"].unique())
    OUT_GM12878_LIST.write_text("\n".join(gene_list) + "\n")
    OUT_K562_LIST.write_text("\n".join(gene_list) + "\n")
    switchers.to_csv(OUT_MERGED, sep="\t", index=False)

    findings = f"""# Switching Gene Extraction v2 — Findings (Day 47)

## Root cause (final)
`gate_assignments.tsv` had a genuinely empty `gate_type` cell on disk for
34,163 rows (confirmed by raw text inspection: `'ENSG00000000003\\t\\t-1.0...'`
— two adjacent tabs, not the text "NULL"). This is NOT a read-time parsing
issue in today's scripts. It is upstream corruption: some prior script read
this file with pandas' default NA handling (which treats the string "NULL"
as missing), then wrote the DataFrame back out with `to_csv()`, which
serializes NaN as an empty string by default. The original "NULL" text was
permanently lost from that file at that point.

## Fix
`gate_type` and `complexity_score` were regenerated directly from
`binary_matrix.csv` / `k562_binary_matrix.csv` (source of truth, confirmed
intact — no NaNs in mark columns) using `call_gate()` / 
`compute_complexity_score()` copied verbatim from the Day 5 / Day 14
scripts. Only `gene_name` / `gene_id_clean` annotation was borrowed from the
old TSVs. Result: 0 NaN gate_type values in either regenerated file.

## Output
- `gate_assignments_regenerated_v1.tsv`, `k562_gate_assignments_regenerated_v1.tsv`
  — these should replace the originals as the trusted source going forward.
- `switching_genes_gm12878_v1.txt`, `switching_genes_k562_v1.txt` — {len(gene_list)} genes each
- `switching_genes_merged_v1.tsv` — full per-gene, per-cell-line gate_type + and_origin

## Excluded genes (unknown expression status)
{chr(10).join(f"- {label}: {len(ids)} genes excluded" for label, ids in EXCLUSION_STATS.items()) if EXCLUSION_STATS else "- None"}

## Counts vs. Day 38 document
Total switchers: {n_total} (Day 38 doc: 6,962)
AND<->INC switchers: {n_and_inc} (Day 38 doc: 3,502)

If these don't match, the Day 38 figures should be treated as suspect and
re-derived from these clean regenerated files, not reconciled by adjusting
today's numbers to match.

## Action item for MASTER_STATUS.md / RESEARCH_LOG.md
gate_assignments.tsv and k562_gate_assignments_named.tsv are flagged
CORRUPTED (gate_type column) and should not be read directly by any future
script. Point all downstream Logic Circuits work at the regenerated files
above instead.
"""
    OUT_LOG.write_text(findings)
    print(f"\nWrote {OUT_GM12878_LIST}, {OUT_K562_LIST}, {OUT_MERGED}, {OUT_LOG}")


if __name__ == "__main__":
    main()