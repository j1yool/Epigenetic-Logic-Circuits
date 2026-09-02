"""
build_k562_stratified_rebinning_v1.py

Day 61 | Epigenetic Logic Circuits

Purpose
-------
Day 57's compare_decile_structure_v1.py showed K562's pooled decile
boundaries 1 and 2 both landing at exactly 0.0 (a collapsed, zero-width
decile 1 with 0 switching / 0 pool genes), while GM12878's decile 1 is
cleanly populated. This script tests the zero/near-zero-TPM-mass
hypothesis: that K562 has a materially larger fraction of zero or
near-zero-TPM genes than GM12878, and that this mass is what breaks
flat decile-edge fitting at the low end for K562 specifically.

Fix: carve zero/near-zero-TPM genes into their own explicit stratum
BEFORE calling assign_deciles() on the remainder, for both cell lines
(computed for both so the asymmetry -- or lack of one -- is visible,
not assumed).

This is NOT a re-implementation of matched_nonswitching_v1.py's or
compare_decile_structure_v1.py's binning/merge logic. It imports:
  - matcher.assign_deciles()          (via matched_nonswitching_v1.py)
  - matcher._load_gate_module()       (via matched_nonswitching_v1.py)
  - diag.build_switching_and_pool_tpm()  (via compare_decile_structure_v1.py)
all via importlib.util.spec_from_file_location, per project convention.
Recomputing switching/pool TPM assembly here independently would risk
silently diverging from the Day 57 merge path this diagnostic depends on.

CONFIRMED SCHEMA (checked against real files before writing this script):
  data/encode_rnaseq_gm12878_replicates_v1.csv
      columns: gene_id, rep1, rep2 (gene_id = unversioned ENSG)
  data/encode_rnaseq_k562_replicates_v1.csv
      columns: gene_id, rep1, rep2
  data/gate_assignments_named.tsv        (GM12878 gate calls)
  data/k562_gate_assignments_named.tsv   (K562 gate calls)
  mean_tpm is computed as mean(rep1, rep2), matching
  compare_decile_structure_v1.py's build_switching_and_pool_tpm() exactly.

Inputs (real, on disk)
-----------------------
  code/matched_nonswitching_v1.py         (imported)
  compare_decile_structure_v1.py          (imported)
  data/gate_assignments_named.tsv
  data/k562_gate_assignments_named.tsv
  data/encode_rnaseq_gm12878_replicates_v1.csv
  data/encode_rnaseq_k562_replicates_v1.csv

Outputs
-------
  data/k562_zero_stratified_decile_comparison_v1.tsv
      per-decile switching/pool counts for K562, computed on the
      nonzero remainder only, plus the zero-stratum counts reported
      as a separate row (decile = "ZERO_STRATUM").
  Same structure printed for GM12878 as a comparison-only reference
  (not written to file unless the asymmetry turns out to matter).

Usage
-----
    python build_k562_stratified_rebinning_v1.py --self-test
    python build_k562_stratified_rebinning_v1.py
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ---- all numeric / data imports go below this line, never above ----
import sys
import argparse
import importlib.util
import numpy as np
import pandas as pd

MATCHER_SCRIPT_PATH = "code/matched_nonswitching_v1.py"
DIAG_SCRIPT_PATH = "compare_decile_structure_v1.py"

GM12878_GATE_PATH = "data/gate_assignments_named.tsv"
K562_GATE_PATH = "data/k562_gate_assignments_named.tsv"
GM12878_EXPR_PATH = "data/encode_rnaseq_gm12878_replicates_v1.csv"
K562_EXPR_PATH = "data/encode_rnaseq_k562_replicates_v1.csv"

# Zero-mass threshold (Block 1 design decision): mean_tpm <= 0.1 is
# treated as zero/near-zero. Rationale: K562's collapsed boundary sits
# at exactly 0.0, so the true-zero mass is tested first; 0.1 gives a
# small buffer for near-zero noise without absorbing real low-expression
# signal. If this needs to change, it is the ONLY constant to edit.
ZERO_THRESHOLD = 0.1

N_DECILES = 10
OUT_PATH = "data/k562_zero_stratified_decile_comparison_v1.tsv"


def _load_matcher_module():
    if not os.path.exists(MATCHER_SCRIPT_PATH):
        sys.exit(
            f"FATAL: cannot find {MATCHER_SCRIPT_PATH} to import assign_deciles / "
            f"_load_gate_module from. Not proceeding with a re-implemented copy."
        )
    spec = importlib.util.spec_from_file_location("matcher", MATCHER_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_diag_module():
    if not os.path.exists(DIAG_SCRIPT_PATH):
        sys.exit(
            f"FATAL: cannot find {DIAG_SCRIPT_PATH} to import "
            f"build_switching_and_pool_tpm from. Not proceeding with a "
            f"re-implemented copy of the Day 57 merge path."
        )
    spec = importlib.util.spec_from_file_location("diag", DIAG_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stratify_zero_mass(tpm_series: pd.Series, threshold: float):
    """
    Splits a mean_tpm Series into (zero_stratum_values, nonzero_remainder_values).
    zero_stratum: tpm <= threshold. nonzero_remainder: tpm > threshold.
    """
    zero_stratum = tpm_series[tpm_series <= threshold]
    nonzero_remainder = tpm_series[tpm_series > threshold]
    return zero_stratum, nonzero_remainder


def compute_stratified_decile_counts(matcher, switching_tpm: pd.DataFrame,
                                      nonswitching_tpm: pd.DataFrame,
                                      threshold: float, cell_line: str):
    """
    For one cell line: strata switching and non-switching mean_tpm
    separately, pools the nonzero remainders for edge-fitting (mirrors
    compute_diagnostics()'s pooled-edge approach in compare_decile_structure_v1.py),
    then bins each nonzero remainder using matcher.assign_deciles().

    Returns a DataFrame with rows for each decile (0..N_DECILES-1) plus
    a final "ZERO_STRATUM" row, columns [decile, switch_n, pool_n].
    """
    sw_tpm = switching_tpm["mean_tpm"]
    ns_tpm = nonswitching_tpm["mean_tpm"]

    sw_zero, sw_nonzero = stratify_zero_mass(sw_tpm, threshold)
    ns_zero, ns_nonzero = stratify_zero_mass(ns_tpm, threshold)

    print(f"  [{cell_line}] switching: {len(sw_zero)} zero-stratum / {len(sw_nonzero)} nonzero-remainder "
          f"(of {len(sw_tpm)} total)")
    print(f"  [{cell_line}] pool:      {len(ns_zero)} zero-stratum / {len(ns_nonzero)} nonzero-remainder "
          f"(of {len(ns_tpm)} total)")

    pooled_nonzero = pd.concat([sw_nonzero, ns_nonzero], ignore_index=True)
    _, edges = matcher.assign_deciles(pooled_nonzero)

    sw_deciles, _ = matcher.assign_deciles(sw_nonzero, edges=edges)
    ns_deciles, _ = matcher.assign_deciles(ns_nonzero, edges=edges)

    sw_counts = pd.Series(sw_deciles).value_counts().reindex(range(N_DECILES), fill_value=0).sort_index()
    ns_counts = pd.Series(ns_deciles).value_counts().reindex(range(N_DECILES), fill_value=0).sort_index()

    rows = pd.DataFrame({
        "decile": list(range(N_DECILES)) + ["ZERO_STRATUM"],
        "switch_n": list(sw_counts.to_numpy()) + [len(sw_zero)],
        "pool_n": list(ns_counts.to_numpy()) + [len(ns_zero)],
    })
    return rows, edges


def synthetic_self_test():
    """
    Hand-derived toy case, built directly on stratify_zero_mass() and
    compute_stratified_decile_counts() rather than the full real pipeline
    (which requires real gate/expression files not available at self-test
    time).

    10 synthetic "switching" genes, 10 synthetic "pool" genes:
      switching: 2 at TPM=0.0, 8 spread 1.0-50.0
      pool:      5 at TPM=0.0 (deliberately zero-heavy, mimicking the
                 K562 hypothesis), 5 spread 1.0-50.0

    With ZERO_THRESHOLD=0.1:
      expected switching zero-stratum = 2, nonzero-remainder = 8
      expected pool zero-stratum = 5, nonzero-remainder = 5

    This is checked directly against stratify_zero_mass() before touching
    assign_deciles() at all -- isolates the stratification logic from
    the imported binning logic, so a self-test failure here can never be
    confused with a bug in matched_nonswitching_v1.py.
    """
    switching = pd.Series([0.0, 0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0])
    pool = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 10.0, 25.0, 50.0])

    sw_zero, sw_nonzero = stratify_zero_mass(switching, ZERO_THRESHOLD)
    ns_zero, ns_nonzero = stratify_zero_mass(pool, ZERO_THRESHOLD)

    if len(sw_zero) != 2 or len(sw_nonzero) != 8:
        print(f"SELF-TEST FAILED: switching stratification wrong. "
              f"zero={len(sw_zero)} (expected 2), nonzero={len(sw_nonzero)} (expected 8)")
        sys.exit(1)

    if len(ns_zero) != 5 or len(ns_nonzero) != 5:
        print(f"SELF-TEST FAILED: pool stratification wrong. "
              f"zero={len(ns_zero)} (expected 5), nonzero={len(ns_nonzero)} (expected 5)")
        sys.exit(1)

    if sw_nonzero.min() <= ZERO_THRESHOLD or ns_nonzero.min() <= ZERO_THRESHOLD:
        print("SELF-TEST FAILED: a nonzero-remainder value is <= threshold.")
        sys.exit(1)

    print("SELF-TEST PASSED: zero-stratification isolates the correct genes on hand-derived fixture.")


def run_real_data():
    matcher = _load_matcher_module()
    diag = _load_diag_module()

    print("=== K562 ===")
    k5_switch_tpm, k5_pool_tpm = diag.build_switching_and_pool_tpm(
        matcher, "k562", K562_GATE_PATH, K562_EXPR_PATH
    )
    k5_rows, k5_edges = compute_stratified_decile_counts(
        matcher, k5_switch_tpm, k5_pool_tpm, ZERO_THRESHOLD, "K562"
    )
    print("\nK562 zero-stratified decile table:")
    print(k5_rows.to_string(index=False))

    decile_1_row = k5_rows[k5_rows["decile"] == 0].iloc[0]
    if decile_1_row["switch_n"] == 0 and decile_1_row["pool_n"] == 0:
        print(
            "\nWARNING: K562 decile 1 (of the nonzero remainder) is STILL empty "
            "after zero-stratification. Threshold may be set too low, or the "
            "collapse has a different cause than zero-TPM mass -- revisit before "
            "trusting this rebinning."
        )
    else:
        print(
            f"\nK562 decile 1 (nonzero remainder) is now populated: "
            f"switch_n={decile_1_row['switch_n']}, pool_n={decile_1_row['pool_n']}. "
            f"Zero-mass hypothesis supported."
        )

    print("\n=== GM12878 (comparison only) ===")
    gm_switch_tpm, gm_pool_tpm = diag.build_switching_and_pool_tpm(
        matcher, "gm12878", GM12878_GATE_PATH, GM12878_EXPR_PATH
    )
    gm_rows, gm_edges = compute_stratified_decile_counts(
        matcher, gm_switch_tpm, gm_pool_tpm, ZERO_THRESHOLD, "GM12878"
    )
    print("\nGM12878 zero-stratified decile table (comparison only):")
    print(gm_rows.to_string(index=False))

    k5_zero_row = k5_rows[k5_rows["decile"] == "ZERO_STRATUM"].iloc[0]
    gm_zero_row = gm_rows[gm_rows["decile"] == "ZERO_STRATUM"].iloc[0]
    k5_zero_total = k5_zero_row["switch_n"] + k5_zero_row["pool_n"]
    gm_zero_total = gm_zero_row["switch_n"] + gm_zero_row["pool_n"]
    k5_all_total = len(k5_switch_tpm) + len(k5_pool_tpm)
    gm_all_total = len(gm_switch_tpm) + len(gm_pool_tpm)
    print(
        f"\n[ASYMMETRY CHECK] zero-stratum share of total genes: "
        f"K562 = {k5_zero_total}/{k5_all_total} ({k5_zero_total/k5_all_total:.1%}), "
        f"GM12878 = {gm_zero_total}/{gm_all_total} ({gm_zero_total/gm_all_total:.1%})"
    )

    k5_rows.to_csv(OUT_PATH, sep="\t", index=False)
    print(f"\nWrote {OUT_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true",
                         help="Run only the synthetic self-test and exit.")
    args = parser.parse_args()

    print("Running synthetic_self_test() ...")
    synthetic_self_test()
    print("Self-test passed.\n")

    if args.self_test:
        return

    print("Proceeding to real-data run.")
    run_real_data()


if __name__ == "__main__":
    main()