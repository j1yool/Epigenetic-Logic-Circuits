"""
background_composition_check_v1.py

Day 52, Block 3 (Logic Circuits) — epigenetic-logic-circuits repo.

Purpose
-------
Last night's gene_set_definition_decision_v1.md flagged the 42,771-symbol
background's detection/expression-restriction status as unresolved, citing
Timmons, Szkop & Gallagher (2015, Genome Biology 16:186): using a background
that includes undetected genes biases enrichment toward gene-length- and
detectability-correlated pathways.

CORRECTION vs. how this was scoped last night: the original plan called for
cross-referencing the background against RNA-seq expression data (TPM > 1).
No RNA-seq file exists anywhere in this pipeline -- checked directly against
run_pathway_enrichment_v1.py, which has no RNA-seq dependency at all. That
script's own background construction turned out to already contain the
answer, and it's worse than a missing-validation-step problem:

NAMED BUG in run_pathway_enrichment_v1.py's load_background_and_origin_groups():
its docstring states background = "genes with a valid gate call in BOTH cell
lines," but the actual code (`pd.merge(gm[['gene_id']], k562[['gene_id']],
on='gene_id', how='inner')`) applies NO gate_type filter at all. Because
GM12878's 56,625 gene_ids are a strict subset of K562's 58,274 (confirmed
directly), the resulting "background" is simply GM12878's ENTIRE annotation
table -- gate_type == "NULL" rows included. NULL / complexity_score == -1
means no chromatin gate could be called (insufficient signal across marks),
which is the pipeline's own native detectability signal -- no external
expression file is needed to operationalize "detected" here.

This script audits that real background against gate_type as the detection
proxy, rather than checking a hypothetical RNA-seq restriction that was
never actually implemented in the pipeline.

Convention compliance
----------------------
- keep_default_na=False on both TSV reads -- gate_type contains the literal
  string "NULL" as a real value, and default pandas NA parsing would
  silently convert it to a true NaN, breaking every equality check below.
- synthetic_self_test() runs first; sys.exit(1) on any assertion failure.
- Schema validated against real, confirmed columns; sys.exit(1) with the
  actual column list on any mismatch -- no guessing.
- Reproduces run_pathway_enrichment_v1.py's ACTUAL background-construction
  code path exactly (not its docstring), so the audited background is the
  real one Day 47's enrichment results were run against.

REQUIRED EDIT BEFORE RUNNING:
  GM_REGEN_PATH / K562_REGEN_PATH -- paths to the two regenerated
  gate-assignment files (same files run_pathway_enrichment_v1.py reads).
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
from datetime import datetime

import pandas as pd

# ============================================================
# CONFIG — EDIT ME
# ============================================================

GM_REGEN_PATH = r"C:\Users\jamoo\Downloads\LOGIC CIRCUITS\code\gate_assignments_regenerated_v1.tsv"
K562_REGEN_PATH = r"C:\Users\jamoo\Downloads\LOGIC CIRCUITS\code\k562_gate_assignments_regenerated_v1.tsv"

OUTPUT_CSV = r"C:\Users\jamoo\Downloads\LOGIC CIRCUITS\code\background_detection_audit_v1.csv"

REQUIRED_GM_COLS = ["gene_id", "gate_type", "gene_name"]
REQUIRED_K562_COLS = ["gene_id", "gate_type"]

NULL_TOKEN = "NULL"  # the literal string value gate_type uses for "no call"


# ============================================================
# Core audit logic (importable, testable)
# ============================================================

def build_background_and_audit(gm_df, k562_df):
    """
    Reproduces run_pathway_enrichment_v1.py's ACTUAL background construction
    (gene_id inner join, no gate_type filter -- matching the real code, not
    its docstring), then audits gate_type composition of that background in
    both cell lines.

    Returns a DataFrame: gene_id, gene_name, gm12878_detected, k562_detected,
    detected_in_both, in_current_background (always True -- see module
    docstring on why the current background is unfiltered).
    """
    gm_ids = gm_df[["gene_id"]]
    k562_ids = k562_df[["gene_id"]]
    background_ids = pd.merge(gm_ids, k562_ids, on="gene_id", how="inner")["gene_id"]

    gm_gate_lookup = gm_df.set_index("gene_id")["gate_type"]
    gm_name_lookup = gm_df.set_index("gene_id")["gene_name"]
    k562_gate_lookup = k562_df.set_index("gene_id")["gate_type"]

    gm_gate = gm_gate_lookup.reindex(background_ids)
    k562_gate = k562_gate_lookup.reindex(background_ids)
    gene_name = gm_name_lookup.reindex(background_ids)

    audit = pd.DataFrame({
        "gene_id": background_ids.values,
        "gene_name": gene_name.values,
        "gm12878_detected": (gm_gate != NULL_TOKEN).values,
        "k562_detected": (k562_gate != NULL_TOKEN).values,
    })
    audit["detected_in_both"] = audit["gm12878_detected"] & audit["k562_detected"]
    audit["in_current_background"] = True  # the real bug: nothing is excluded today

    return audit


def compute_undetected_fraction(audit_df):
    """Fraction of the current (unfiltered) background that is NULL-gate
    (undetected) in at least one cell line -- the population that should be
    excluded under the docstring's own stated criterion (valid call in BOTH)."""
    n_total = len(audit_df)
    n_undetected_either = (~audit_df["detected_in_both"]).sum()
    return n_undetected_either / n_total


# ============================================================
# Synthetic self-test — MUST pass before any real-data run
# ============================================================

def synthetic_self_test():
    # 50 synthetic genes: 30 detected in both cell lines, 20 undetected in
    # at least one (split across "NULL in GM only", "NULL in K562 only",
    # and "NULL in both" to exercise every branch of the OR logic).
    n_detected_both = 30
    n_null_gm_only = 7
    n_null_k562_only = 7
    n_null_both = 6
    assert n_detected_both + n_null_gm_only + n_null_k562_only + n_null_both == 50

    ids = [f"SYNTH{i:03d}" for i in range(50)]
    gm_gates, k562_gates = [], []
    for i in range(50):
        if i < n_detected_both:
            gm_gates.append("SIMPLE_AND"); k562_gates.append("INCONSISTENT")
        elif i < n_detected_both + n_null_gm_only:
            gm_gates.append("NULL"); k562_gates.append("SIMPLE_AND")
        elif i < n_detected_both + n_null_gm_only + n_null_k562_only:
            gm_gates.append("SIMPLE_AND"); k562_gates.append("NULL")
        else:
            gm_gates.append("NULL"); k562_gates.append("NULL")

    gm_synth = pd.DataFrame({
        "gene_id": ids, "gate_type": gm_gates,
        "complexity_score": [1.0] * 50, "gene_id_clean": ids,
        "gene_name": [f"GENE{i:03d}" for i in range(50)],
    })
    k562_synth = pd.DataFrame({
        "gene_id": ids, "gate_type": k562_gates, "complexity_score": [1] * 50,
    })

    audit = build_background_and_audit(gm_synth, k562_synth)

    assert len(audit) == 50, f"FATAL: expected 50 background rows, got {len(audit)}"

    n_detected_both_actual = audit["detected_in_both"].sum()
    assert n_detected_both_actual == 30, (
        f"FATAL: expected exactly 30 detected-in-both rows on synthetic data, "
        f"got {n_detected_both_actual} -- gate_type NULL-comparison logic is broken."
    )

    undetected_fraction = compute_undetected_fraction(audit)
    expected_fraction = 20 / 50  # 7 + 7 + 6 = 20 undetected-in-at-least-one
    assert abs(undetected_fraction - expected_fraction) < 1e-9, (
        f"FATAL: expected undetected fraction {expected_fraction} (20/50), "
        f"got {undetected_fraction} -- partition math is broken."
    )

    # Confirm the "NULL in exactly one line" cases are correctly counted as
    # undetected (not accidentally counted as detected via an OR/AND mixup).
    null_gm_only_rows = audit.iloc[n_detected_both:n_detected_both + n_null_gm_only]
    assert (~null_gm_only_rows["detected_in_both"]).all(), \
        "FATAL: NULL-in-GM12878-only rows incorrectly marked as detected_in_both."
    null_k562_only_rows = audit.iloc[n_detected_both + n_null_gm_only:n_detected_both + n_null_gm_only + n_null_k562_only]
    assert (~null_k562_only_rows["detected_in_both"]).all(), \
        "FATAL: NULL-in-K562-only rows incorrectly marked as detected_in_both."

    print(f"[synthetic_self_test] PASSED — 50-gene synthetic partition "
          f"(30 detected-both / 20 undetected-in-at-least-one) correctly "
          f"recovered, undetected fraction = {undetected_fraction:.4f} "
          f"(expected {expected_fraction:.4f}).")


# ============================================================
# Schema confirmation — no guessing column names
# ============================================================

def load_and_validate(path, required_cols, label):
    if not os.path.exists(path):
        sys.exit(f"[FATAL] {label} not found at {path}. Edit the path in CONFIG.")
    df = pd.read_csv(path, sep="\t", keep_default_na=False)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        sys.exit(
            f"[FATAL] {label}: expected column(s) {missing} not found.\n"
            f"Actual columns present: {list(df.columns)}\n"
            f"Not proceeding with a guessed schema."
        )
    return df


# ============================================================
# Real-data run
# ============================================================

def main():
    print("=" * 70)
    print(f"background_composition_check_v1.py — run at {datetime.now().isoformat()}")
    print("=" * 70)

    synthetic_self_test()
    print()

    gm_df = load_and_validate(GM_REGEN_PATH, REQUIRED_GM_COLS, "GM12878 regenerated gate assignments")
    k562_df = load_and_validate(K562_REGEN_PATH, REQUIRED_K562_COLS, "K562 regenerated gate assignments")

    if not set(gm_df["gene_id"]).issubset(set(k562_df["gene_id"])):
        n_not_subset = len(set(gm_df["gene_id"]) - set(k562_df["gene_id"]))
        print(f"[NOTE] GM12878 gene_id list is NOT fully contained in K562's "
              f"({n_not_subset} GM12878 gene_id(s) absent from K562) -- this "
              f"differs from the Day 52 confirmation run. The background "
              f"below is still the real inner-join result, just flagging "
              f"that the 'background == GM12878's full list' shortcut no "
              f"longer holds exactly if this note appears.")

    audit = build_background_and_audit(gm_df, k562_df)
    audit.to_csv(OUTPUT_CSV, index=False)

    n_total = len(audit)
    n_detected_both = audit["detected_in_both"].sum()
    n_null_gm = (~audit["gm12878_detected"]).sum()
    n_null_k562 = (~audit["k562_detected"]).sum()
    n_null_both = ((~audit["gm12878_detected"]) & (~audit["k562_detected"])).sum()
    undetected_fraction_either = compute_undetected_fraction(audit)

    print("\n" + "=" * 70)
    print("BACKGROUND COMPOSITION AUDIT — REAL DATA")
    print("=" * 70)
    print(f"Current (unfiltered) background size, as run_pathway_enrichment_v1.py "
          f"actually constructs it: {n_total}")
    print(f"NULL gate_type in GM12878 (within this background): {n_null_gm} ({n_null_gm/n_total*100:.1f}%)")
    print(f"NULL gate_type in K562 (within this background): {n_null_k562} ({n_null_k562/n_total*100:.1f}%)")
    print(f"NULL in BOTH cell lines: {n_null_both} ({n_null_both/n_total*100:.1f}%)")
    print(f"Detected (non-NULL gate call) in BOTH cell lines: {n_detected_both} ({n_detected_both/n_total*100:.1f}%)")
    print(f"Undetected in AT LEAST ONE cell line (fraction requiring exclusion "
          f"under the docstring's own stated criterion): {undetected_fraction_either*100:.1f}%")
    print(f"\nRestricted background if corrected to the docstring's own criterion "
          f"(valid call in BOTH cell lines): n = {n_detected_both}")
    print(f"\nWrote per-gene audit to: {OUTPUT_CSV}")
    print(
        "\nTranscribe the verdict, unhedged, as line one of "
        "background_composition_verdict_v1.md, naming the exact undetected "
        "percentage and the exact restricted-background size (n=" + str(n_detected_both) + ")."
    )


if __name__ == "__main__":
    main()