"""
extract_k562_gene_decile_table_v1.py

Day 63 | Logic Circuits | Block 1

Purpose: characterize_decile_shortfall_v1.py (Day 62) computed a full
per-gene, all-decile table internally (`gene_table` inside run_real_data())
but never wrote it to disk -- only the deciles-0-3-restricted aggregate
counts (k562_decile03_shortfall_characterization_v1.tsv) were saved.

This means the reported 57.6% (switching) / 27.7% (non-switching) figure --
share of each group's nonzero remainder sitting in deciles 0-3 -- cannot be
reproduced from the saved Day 62 output alone, since that file has no
denominator (total nonzero count across ALL deciles, 0-9).

This script does NOT re-derive zero-stratification, decile-binning, or the
switching/pool TPM assembly logic. It imports, via
importlib.util.spec_from_file_location, the same real pipeline as Day 62's
script:
  - build_k562_stratified_rebinning_v1.py's stratify_zero_mass(),
    ZERO_THRESHOLD, _load_matcher_module(), _load_diag_module()
  - matcher.assign_deciles()             (loaded transitively)
  - diag.build_switching_and_pool_tpm()  (loaded transitively)

Two things this script produces that Day 62's did not:
  1. k562_full_gene_decile_table_v1.tsv -- full per-gene rows
     (gene_id, mean_tpm, decile [0-9], is_switching), all deciles, both
     groups. This is the artifact Block 2's isolation script needs.
  2. A proper reproduction check of the 57.6% / 27.7% figure, computed
     with the correct denominator (each group's total nonzero-remainder
     count, all deciles), not just the deciles-0-3 subset.

Run from this script's own folder, alongside build_k562_stratified_rebinning_v1.py.

Usage:
    python extract_k562_gene_decile_table_v1.py --self-test
    python extract_k562_gene_decile_table_v1.py
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ---- all numeric / data imports go below this line, never above ----
import sys
import argparse
import importlib.util
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

REBINNING_MODULE_PATH = SCRIPT_DIR / "build_k562_stratified_rebinning_v1.py"
DAY62_OUTPUT_PATH = SCRIPT_DIR / "k562_decile03_shortfall_characterization_v1.tsv"

K562_GATE_PATH = "data/k562_gate_assignments_named.tsv"
K562_EXPR_PATH = "data/encode_rnaseq_k562_replicates_v1.csv"

TARGET_DECILES = [0, 1, 2, 3]
OUT_PATH = SCRIPT_DIR / "k562_full_gene_decile_table_v1.tsv"

EXPECTED_TPM_COLUMNS = {"gene_id", "mean_tpm"}
EXPECTED_DAY62_COLUMNS = {"switching", "nonswitching", "total"}


def _load_rebinning_module():
    if not REBINNING_MODULE_PATH.exists():
        sys.exit(
            f"FATAL: cannot find {REBINNING_MODULE_PATH}. This script depends on "
            f"its stratify_zero_mass(), ZERO_THRESHOLD, _load_matcher_module(), and "
            f"_load_diag_module() -- not proceeding with a reimplemented copy."
        )
    spec = importlib.util.spec_from_file_location("rebin_mod", REBINNING_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_tpm_schema(df: pd.DataFrame, label: str) -> None:
    actual = set(df.columns)
    if not EXPECTED_TPM_COLUMNS.issubset(actual):
        sys.exit(
            f"SCHEMA MISMATCH in {label}\n"
            f"  expected (subset of): {sorted(EXPECTED_TPM_COLUMNS)}\n"
            f"  actual columns:       {sorted(actual)}"
        )


def _check_stale_day62_output():
    """
    Windows stale-file risk check: confirm Day 62's saved output actually
    exists and report its structure and mtime, so we know we are building
    on top of the real Day 62 run and not assuming its shape blind.
    Does not exit fatally if missing -- this script does not depend on
    that file's contents, only flags the audit finding.
    """
    if not DAY62_OUTPUT_PATH.exists():
        print(f"NOTE: {DAY62_OUTPUT_PATH.name} not found in this folder. "
              f"Cannot cross-check against Day 62's saved aggregate output.")
        return None

    mtime = datetime.fromtimestamp(DAY62_OUTPUT_PATH.stat().st_mtime)
    print(f"Found Day 62 output: {DAY62_OUTPUT_PATH.name}, last modified {mtime}")

    day62_df = pd.read_csv(DAY62_OUTPUT_PATH, sep="\t", index_col=0)
    actual_cols = set(day62_df.columns)
    print(f"  columns: {sorted(actual_cols)}  |  rows (deciles present): {list(day62_df.index)}")

    if not EXPECTED_DAY62_COLUMNS.issubset(actual_cols):
        print(f"  WARNING: expected columns {sorted(EXPECTED_DAY62_COLUMNS)} not fully "
              f"present -- schema may have drifted since Day 62.")
    if set(day62_df.index) != set(TARGET_DECILES):
        print(f"  CONFIRMS: file is restricted to deciles {sorted(day62_df.index)} only -- "
              f"no all-decile denominator available from this file alone, as expected.")

    return day62_df


def compute_group_proportions(full_gene_table: pd.DataFrame) -> dict:
    """
    full_gene_table: DataFrame with columns [gene_id, mean_tpm, decile, is_switching],
    spanning ALL deciles (0-9), both groups, nonzero remainder only.

    Returns, per group, the count in TARGET_DECILES, the total nonzero count
    (all deciles), and the proportion -- the correctly-denominated version
    of the 57.6% / 27.7% figure.
    """
    out = {}
    for label, flag in [("switching", True), ("nonswitching", False)]:
        grp = full_gene_table[full_gene_table["is_switching"] == flag]
        total_n = len(grp)
        in_target_n = int(grp["decile"].isin(TARGET_DECILES).sum())
        prop = in_target_n / total_n if total_n > 0 else float("nan")
        out[label] = {
            "total_nonzero_n": total_n,
            "n_in_deciles_0_3": in_target_n,
            "proportion_in_0_3": prop,
        }
    return out


def synthetic_self_test():
    """
    Hand-derivable fixture spanning deciles 0-9, both groups, so the
    proportion calculation's denominator logic (total across ALL deciles,
    not just 0-3) is exercised and checked by hand.

    Switching group: 10 genes total, 6 in deciles 0-3 -> expect 0.6.
    Nonswitching group: 10 genes total, 3 in deciles 0-3 -> expect 0.3.
    """
    deciles_switch = [0, 0, 1, 2, 3, 3, 4, 5, 7, 9]      # 6 of 10 in {0,1,2,3}
    deciles_nonswitch = [0, 1, 3, 4, 4, 5, 6, 7, 8, 9]   # 3 of 10 in {0,1,2,3}

    fixture = pd.DataFrame({
        "gene_id": [f"SW{i}" for i in range(10)] + [f"NS{i}" for i in range(10)],
        "mean_tpm": np.concatenate([
            np.linspace(1, 10, 10),
            np.linspace(1, 10, 10),
        ]),
        "decile": deciles_switch + deciles_nonswitch,
        "is_switching": [True] * 10 + [False] * 10,
    })

    result = compute_group_proportions(fixture)

    expected = {
        "switching": {"total_nonzero_n": 10, "n_in_deciles_0_3": 6, "proportion_in_0_3": 0.6},
        "nonswitching": {"total_nonzero_n": 10, "n_in_deciles_0_3": 3, "proportion_in_0_3": 0.3},
    }

    for label in ["switching", "nonswitching"]:
        for key in ["total_nonzero_n", "n_in_deciles_0_3"]:
            got = result[label][key]
            exp = expected[label][key]
            if got != exp:
                print(f"SELF-TEST FAILED: {label}.{key} mismatch. got={got} expected={exp}")
                sys.exit(1)
        got_p = result[label]["proportion_in_0_3"]
        exp_p = expected[label]["proportion_in_0_3"]
        if abs(got_p - exp_p) > 1e-9:
            print(f"SELF-TEST FAILED: {label}.proportion_in_0_3 mismatch. "
                  f"got={got_p} expected={exp_p}")
            sys.exit(1)

    print("SELF-TEST PASSED: proportion-in-deciles-0-3 correctly denominated "
          "against total nonzero count across all deciles, both groups.")


def run_real_data():
    rebin_mod = _load_rebinning_module()
    matcher = rebin_mod._load_matcher_module()
    diag = rebin_mod._load_diag_module()

    print("=== K562: building switching/pool TPM tables (real pipeline) ===")
    sw_df, ns_df = diag.build_switching_and_pool_tpm(
        matcher, "k562", K562_GATE_PATH, K562_EXPR_PATH
    )

    _check_tpm_schema(sw_df, "K562 switch_tpm")
    _check_tpm_schema(ns_df, "K562 pool_tpm")

    sw_tpm = sw_df.set_index("gene_id")["mean_tpm"]
    ns_tpm = ns_df.set_index("gene_id")["mean_tpm"]

    sw_zero, sw_nonzero = rebin_mod.stratify_zero_mass(sw_tpm, rebin_mod.ZERO_THRESHOLD)
    ns_zero, ns_nonzero = rebin_mod.stratify_zero_mass(ns_tpm, rebin_mod.ZERO_THRESHOLD)

    print(f"  switching: {len(sw_zero)} zero-stratum / {len(sw_nonzero)} nonzero-remainder")
    print(f"  pool:      {len(ns_zero)} zero-stratum / {len(ns_nonzero)} nonzero-remainder")

    # Same pooled-edges decile assignment as Day 62 -- no re-derivation.
    pooled_nonzero = pd.concat([sw_nonzero, ns_nonzero])
    _, edges = matcher.assign_deciles(pooled_nonzero)

    sw_deciles, _ = matcher.assign_deciles(sw_nonzero, edges=edges)
    ns_deciles, _ = matcher.assign_deciles(ns_nonzero, edges=edges)

    sw_table = pd.DataFrame({
        "gene_id": sw_nonzero.index,
        "mean_tpm": sw_nonzero.to_numpy(),
        "decile": np.asarray(sw_deciles),
        "is_switching": True,
    })
    ns_table = pd.DataFrame({
        "gene_id": ns_nonzero.index,
        "mean_tpm": ns_nonzero.to_numpy(),
        "decile": np.asarray(ns_deciles),
        "is_switching": False,
    })
    full_gene_table = pd.concat([sw_table, ns_table], ignore_index=True)

    full_gene_table.to_csv(OUT_PATH, sep="\t", index=False)
    print(f"\nWrote full per-gene, all-decile table: {OUT_PATH}")
    print(f"  rows: {len(full_gene_table)}  "
          f"(switching: {(full_gene_table['is_switching']==True).sum()}, "
          f"nonswitching: {(full_gene_table['is_switching']==False).sum()})")

    print("\n=== Reproduction check: proportion in deciles 0-3 (correct denominator) ===")
    proportions = compute_group_proportions(full_gene_table)
    for label, stats in proportions.items():
        print(f"  {label}: {stats['n_in_deciles_0_3']}/{stats['total_nonzero_n']} "
              f"= {stats['proportion_in_0_3']*100:.1f}% in deciles 0-3")

    sw_pct = proportions["switching"]["proportion_in_0_3"] * 100
    ns_pct = proportions["nonswitching"]["proportion_in_0_3"] * 100
    print(f"\n  Day 62 reported figures: switching 57.6%, nonswitching 27.7%")
    print(f"  Recomputed from full table: switching {sw_pct:.1f}%, nonswitching {ns_pct:.1f}%")
    if abs(sw_pct - 57.6) > 0.5 or abs(ns_pct - 27.7) > 0.5:
        print("  WARNING: recomputed figures deviate from Day 62's reported values by "
              ">0.5pp. Do not proceed to Block 2 until this discrepancy is explained -- "
              "check whether Day 62's percentages were computed against a different "
              "denominator (e.g. total genes rather than nonzero remainder) before "
              "treating either number as authoritative.")
    else:
        print("  CONFIRMED: recomputed figures match Day 62's reported values within "
              "tolerance. Safe to proceed to Block 2.")

    return full_gene_table


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

    print("Checking Day 62 saved output for staleness / schema audit ...")
    _check_stale_day62_output()

    print("\nProceeding to real-data run.")
    run_real_data()


if __name__ == "__main__":
    main()