"""
extract_switching_genes_v1.py

Day 47, Block 1 — Logic Circuits.

Purpose: produce switching_genes_gm12878_v1.txt and switching_genes_k562_v1.txt
for today's pathway enrichment run (Block 2).

WHY THIS SCRIPT LOOKS THE WAY IT DOES:
switching_gene_validation_interpretation_v1.md (Day 38 revision) explicitly
flags the per-cell-line split of the AND->INC category as UNRESOLVED. The two
numbers in that doc (GM12878 delta +0.0552, K562 delta -0.1439) are aggregate
marks-only-model AUC deltas for the WHOLE 6,962-gene switching subset
evaluated against each cell line's model -- they are not a per-gene label and
cannot be used to sort individual genes into a "GM12878 list" vs "K562 list".

What CAN produce a genuine per-gene, per-cell-line split is the two raw
gate-assignment files (Day 5 script -> GM12878; Day 14 script -> K562), each
of which independently calls a gate_type per gene per cell line:

    GM12878: C:\\Users\\jamoo\\Downloads\\LOGIC CIRCUITS\\data\\gate_assignments.tsv
    K562:    C:\\Users\\jamoo\\Downloads\\LOGIC CIRCUITS\\data\\k562\\k562_gate_assignments_named.tsv

Both have columns: gene_id, gate_type, complexity_score.
Vocabulary confirmed from the call_gate() source (identical in both scripts):
    NULL, BIVALENT, REPRESSED, POISED, SIMPLE_AND, SIMPLE_OR, INCONSISTENT, COMPLEX

The switching-gene doc's shorthand "AND", "INC", "BIV" map to
SIMPLE_AND, INCONSISTENT, BIVALENT respectively.

This script merges the two files on gene_id, finds genes whose gate_type
differs between cell lines, and -- for the AND<->INC switchers specifically
(the dominant category, 3,502 / 6,962 = 50.3% per the Day 38 doc) -- labels
each switcher by which cell line has the SIMPLE_AND call. That label is the
"origin" cell line for enrichment purposes: it is the cell line in which the
gene is under simple, clean AND-gate control, switching to inconsistent
logic in the other.

Both output lists (per cell line) are written for Block 2, since a switcher
is relevant to enrichment in both cell lines it appears in -- the origin
label is recorded in the output file for downstream interpretation, not used
to exclude a gene from one list.

STANDING RULES APPLIED:
- Schema confirmed from actual files at runtime; FATAL on mismatch, no
  silent coercion.
- Synthetic self-test before real data.
- Gene counts sanity-checked against the Day 38 doc's cited numbers
  (n=6,962 total switchers, n=3,502 AND->INC) before Block 2 proceeds.
"""

import sys
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GM12878_GATE_FILE = Path(r"C:\Users\jamoo\Downloads\LOGIC CIRCUITS\data\gate_assignments.tsv")
K562_GATE_FILE = Path(r"C:\Users\jamoo\Downloads\LOGIC CIRCUITS\data\k562\k562_gate_assignments_named.tsv")

OUT_GM12878 = Path("switching_genes_gm12878_v1.txt")
OUT_K562 = Path("switching_genes_k562_v1.txt")
OUT_ALL_SWITCHERS = Path("switching_genes_merged_v1.tsv")
OUT_LOG = Path("extract_switching_genes_v1_findings.md")

EXPECTED_TOTAL_SWITCHERS = 6962   # switching_gene_validation_interpretation_v1.md, Day 38
EXPECTED_AND_TO_INC = 3502        # same doc

REQUIRED_COLS = {"gene_id", "gate_type", "complexity_score"}
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
# Core derivation logic (shared by synthetic test and real run)
# ---------------------------------------------------------------------------
def validate_schema(df: pd.DataFrame, label: str) -> None:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        fatal(f"{label}: missing required column(s) {missing}. Found: {list(df.columns)}")
    if df["gene_id"].duplicated().any():
        n_dup = df["gene_id"].duplicated().sum()
        fatal(f"{label}: {n_dup} duplicate gene_id value(s) found. "
              f"Cannot merge on gene_id with duplicates present.")
    bad_types = set(df["gate_type"].unique()) - VALID_GATE_TYPES
    if bad_types:
        fatal(f"{label}: unrecognized gate_type value(s) {bad_types}. "
              f"Expected one of {sorted(VALID_GATE_TYPES)}.")


def derive_switchers(gm_df: pd.DataFrame, k562_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge two per-cell-line gate-assignment tables on gene_id and return
    only genes whose gate_type differs between cell lines, with an
    'and_origin' column set to 'GM12878', 'K562', or 'NEITHER'
    (for switches not involving SIMPLE_AND at all, e.g. BIVALENT<->INCONSISTENT).
    """
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


# ---------------------------------------------------------------------------
# Synthetic self-test
# ---------------------------------------------------------------------------
def synthetic_self_test() -> None:
    print("Running synthetic self-test...")

    gm_synth = pd.DataFrame({
        "gene_id": ["GENEA", "GENEB", "GENEC", "GENED"],
        "gate_type": ["SIMPLE_AND", "INCONSISTENT", "SIMPLE_AND", "BIVALENT"],
        "complexity_score": [0.5, 2.0, 0.4, 3.0],
    })
    k562_synth = pd.DataFrame({
        "gene_id": ["GENEA", "GENEB", "GENEC", "GENED"],
        "gate_type": ["INCONSISTENT", "SIMPLE_AND", "SIMPLE_AND", "INCONSISTENT"],
        "complexity_score": [2.1, 0.3, 0.4, 1.8],
    })
    # GENEA: AND(GM)->INC(K562)   -> origin GM12878
    # GENEB: INC(GM)->AND(K562)   -> origin K562
    # GENEC: AND(GM)->AND(K562)   -> not a switcher (excluded entirely)
    # GENED: BIV(GM)->INC(K562)   -> switcher, origin NEITHER

    validate_schema(gm_synth, "synthetic GM12878")
    validate_schema(k562_synth, "synthetic K562")
    result = derive_switchers(gm_synth, k562_synth)

    got_genes = set(result["gene_id"])
    assert got_genes == {"GENEA", "GENEB", "GENED"}, (
        f"Self-test FAILED: expected switcher set "
        f"{{'GENEA','GENEB','GENED'}}, got {got_genes}"
    )
    assert "GENEC" not in got_genes, "Self-test FAILED: non-switcher GENEC leaked into result"

    origin_map = dict(zip(result["gene_id"], result["and_origin"]))
    assert origin_map["GENEA"] == "GM12878", f"Self-test FAILED: GENEA origin = {origin_map['GENEA']}"
    assert origin_map["GENEB"] == "K562", f"Self-test FAILED: GENEB origin = {origin_map['GENEB']}"
    assert origin_map["GENED"] == "NEITHER", f"Self-test FAILED: GENED origin = {origin_map['GENED']}"

    print("Synthetic self-test PASSED: merge, switcher-detection, and "
          "and_origin labeling all behave correctly on known input.\n")


def synthetic_read_csv_self_test() -> None:
    """
    Regression test for the Day 47 diagnosed bug: pandas' default na_values
    treats the literal string "NULL" as missing. The in-memory
    synthetic_self_test() above never exercises pd.read_csv(), so it could
    not have caught this -- this test writes a real TSV to disk (including a
    NULL gate_type row and a genuinely empty gene_name cell) and confirms
    the read parameters used in main() parse both correctly.
    """
    import tempfile, os
    print("Running synthetic read_csv self-test (NULL-string regression check)...")

    tsv_content = (
        "gene_id\tgate_type\tcomplexity_score\tgene_id_clean\tgene_name\n"
        "GENEX\tNULL\t-1.0\tGENEX\tSOMEGENE\n"
        "GENEY\tSIMPLE_AND\t0.5\tGENEY\t\n"  # genuinely empty gene_name -- SHOULD be NaN
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        f.write(tsv_content)
        tmp_path = f.name

    try:
        df = pd.read_csv(tmp_path, sep="\t", keep_default_na=False, na_values=[""])
        null_row = df[df["gene_id"] == "GENEX"].iloc[0]
        assert null_row["gate_type"] == "NULL", (
            f"Self-test FAILED: literal 'NULL' gate_type was misparsed as "
            f"{null_row['gate_type']!r} (type {type(null_row['gate_type'])}). "
            f"The na_values fix did not work as expected."
        )
        empty_name_row = df[df["gene_id"] == "GENEY"].iloc[0]
        assert pd.isna(empty_name_row["gene_name"]), (
            "Self-test FAILED: genuinely empty gene_name cell was not parsed as NaN -- "
            "na_values=[''] override is not behaving as expected."
        )
    finally:
        os.unlink(tmp_path)

    print("Synthetic read_csv self-test PASSED: literal 'NULL' gate_type string "
          "survives the read intact, and genuinely empty cells still parse as NaN.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    synthetic_self_test()
    synthetic_read_csv_self_test()

    for f in (GM12878_GATE_FILE, K562_GATE_FILE):
        if not f.exists():
            fatal(f"Expected gate-assignment file not found: {f}\n"
                  f"Cannot derive a genuine per-cell-line switching split without it. "
                  f"Block 1 cannot proceed to Block 2 until this file's location is confirmed.")

    # NOTE (Day 47 diagnosis): plain pd.read_csv() treats the literal string
    # "NULL" as a missing value by default (it's in pandas' default
    # na_values list). 'NULL' is a legitimate gate_type value in this
    # codebase's vocabulary (call_gate() explicitly returns 'NULL' for genes
    # with no marks and no expression). Reading with default NA handling
    # silently converts every NULL-gated gene's gate_type to NaN on read --
    # confirmed via diagnose_nan_gate_types_v1.py: all 34,163 NaN rows in
    # gate_assignments.tsv have complexity_score == -1.0 exactly, matching
    # the hand-computed score for a genuine NULL-gate, all-zero-marks gene.
    # keep_default_na=False + na_values=[''] disables the "NULL"-as-missing
    # behavior while still treating genuinely empty cells (e.g. missing
    # gene_name) as NaN.
    gm_df = pd.read_csv(GM12878_GATE_FILE, sep="\t", keep_default_na=False, na_values=[""])
    k562_df = pd.read_csv(K562_GATE_FILE, sep="\t", keep_default_na=False, na_values=[""])

    validate_schema(gm_df, "GM12878 gate_assignments.tsv")
    validate_schema(k562_df, "K562 k562_gate_assignments_named.tsv")

    print(f"GM12878 genes loaded: {len(gm_df)}")
    print(f"K562 genes loaded: {len(k562_df)}")

    switchers = derive_switchers(gm_df, k562_df)

    n_total = len(switchers)
    n_and_inc = len(switchers[
        ((switchers["gate_type_gm12878"] == "SIMPLE_AND") & (switchers["gate_type_k562"] == "INCONSISTENT")) |
        ((switchers["gate_type_gm12878"] == "INCONSISTENT") & (switchers["gate_type_k562"] == "SIMPLE_AND"))
    ])

    print(f"\nTotal switching genes (any gate_type change between cell lines): {n_total}")
    print(f"AND<->INC switchers specifically: {n_and_inc}")
    print(f"Expected (Day 38 doc): total={EXPECTED_TOTAL_SWITCHERS}, AND->INC={EXPECTED_AND_TO_INC}")

    total_match = (n_total == EXPECTED_TOTAL_SWITCHERS)
    and_inc_match = (n_and_inc == EXPECTED_AND_TO_INC)

    if not (total_match and and_inc_match):
        print("\nWARNING: recomputed counts do NOT match the Day 38 document's cited figures.")
        print(f"  Total switchers: recomputed={n_total} vs documented={EXPECTED_TOTAL_SWITCHERS} "
              f"({'MATCH' if total_match else 'MISMATCH'})")
        print(f"  AND->INC:        recomputed={n_and_inc} vs documented={EXPECTED_AND_TO_INC} "
              f"({'MATCH' if and_inc_match else 'MISMATCH'})")
        print("\nPer standing rule (no silent proceeding on mismatched counts), this run STOPS "
              "here rather than writing output lists. Two non-exclusive causes to check before "
              "re-running:")
        print("  1. switching_genes_v1.tsv (the Day 38 doc's source) may have been built from a "
              "different join logic than an inner merge on gene_id used here (e.g. outer join, "
              "or a different gene universe / filtering step upstream).")
        print("  2. The two raw gate-assignment files loaded here may not be the exact files "
              "switching_genes_v1.tsv was originally derived from (file may have been "
              "regenerated/updated since).")
        switchers.to_csv(OUT_ALL_SWITCHERS, sep="\t", index=False)
        print(f"\nRecomputed switcher table written to {OUT_ALL_SWITCHERS} for diagnosis "
              f"(not yet the trusted enrichment input).")
        sys.exit(1)

    print("\nCounts MATCH Day 38 document. Proceeding to write per-cell-line lists.")

    # Write per-cell-line gene lists for Block 2. A switcher is written to a
    # cell line's list if it is a switcher involving that cell line at all
    # (both lists together cover the full switcher set; and_origin in the
    # merged table records directionality for interpretation).
    gm_list = sorted(switchers["gene_id"].unique())
    k562_list = sorted(switchers["gene_id"].unique())
    # NOTE: both cell lines' switching gene SET is identical here because the
    # merge is on the same gene_id universe and switching is inherently a
    # two-cell-line comparison -- what differs per cell line is not which
    # genes are in the list, but each gene's gate_type role (AND-origin vs
    # INC-in-that-line). This is recorded in and_origin, not by exclusion.
    # Flagging explicitly rather than silently writing two identical files
    # without explanation.

    OUT_GM12878.write_text("\n".join(gm_list) + "\n")
    OUT_K562.write_text("\n".join(k562_list) + "\n")
    switchers.to_csv(OUT_ALL_SWITCHERS, sep="\t", index=False)

    findings = f"""# Switching Gene Extraction — Findings (Day 47, Block 1)

## Verdict
Counts MATCH the Day 38 `switching_gene_validation_interpretation_v1.md` figures exactly:
- Total switchers: {n_total} (expected {EXPECTED_TOTAL_SWITCHERS})
- AND<->INC switchers: {n_and_inc} (expected {EXPECTED_AND_TO_INC})

## What was actually derivable
The Day 38 doc's per-cell-line ΔAUC values (GM12878 +0.0552, K562 -0.1439) are
aggregate marks-only-model deltas over the whole switcher subset — NOT a
per-gene cell-line label. A genuine per-gene split was derived instead by
merging the two raw gate-assignment files (Day 5 GM12878 script output, Day 14
K562 script output) on `gene_id` and comparing `gate_type` directly.

## Resolution of the Day 38 UNRESOLVED flag
The per-cell-line AND-origin split (previously flagged unresolved) is now
computed and stored in `switching_genes_merged_v1.tsv`'s `and_origin` column:
- `GM12878`: gene is SIMPLE_AND in GM12878, INCONSISTENT (or other) in K562
- `K562`: gene is SIMPLE_AND in K562, INCONSISTENT (or other) in GM12878
- `NEITHER`: switch does not involve SIMPLE_AND in either line (e.g. BIVALENT<->INCONSISTENT)

AND-origin breakdown:
{switchers['and_origin'].value_counts().to_string()}

## Output files
- `switching_genes_gm12878_v1.txt` — {len(gm_list)} gene IDs, one per line
- `switching_genes_k562_v1.txt` — {len(k562_list)} gene IDs, one per line
- `switching_genes_merged_v1.tsv` — full merged table with per-cell-line gate_type and and_origin, for Block 2 interpretation and any downstream directionality analysis

## Note on identical list contents
Both .txt lists contain the same {len(gm_list)} gene IDs. This is expected: a
switching gene is by definition a gene evaluated in both cell lines whose
call differs between them, so it is relevant to enrichment analysis run
against either cell line's context. Per-gene directionality (which cell line
holds the AND gate) is preserved in `switching_genes_merged_v1.tsv` rather
than encoded by exclusion from one list.
"""
    OUT_LOG.write_text(findings)
    print(f"\nWrote {OUT_GM12878} ({len(gm_list)} genes)")
    print(f"Wrote {OUT_K562} ({len(k562_list)} genes)")
    print(f"Wrote {OUT_ALL_SWITCHERS}")
    print(f"Wrote {OUT_LOG}")


if __name__ == "__main__":
    main()