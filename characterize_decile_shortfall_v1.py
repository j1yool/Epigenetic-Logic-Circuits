"""
characterize_decile_shortfall_v1.py

Day 62 | Logic Circuits | Block 4

Purpose: Independent of the scale-mismatch verdict (CONFIRMED NO MISMATCH,
Day 62 Block 2 -- verified directly against compare_decile_structure_v1.py:154,
which shows mean_tpm as the untransformed linear-scale mean(rep1, rep2), matching
switching_gene_expression_variability_v1.py:217's mean_expr computation),
characterize whether K562's switching-gene set is disproportionately drawn
from low-nonzero-TPM genes concentrated in deciles 0-3, where Day 61 found
the 827-gene shortfall.

Question: within deciles 0-3 of the K562 nonzero remainder, does mean_tpm
differ significantly between switching and non-switching genes (Mann-Whitney U)?

This script does NOT re-derive zero-stratification, decile-binning, or the
switching/pool TPM assembly logic. It imports, via
importlib.util.spec_from_file_location:
  - build_k562_stratified_rebinning_v1.py's stratify_zero_mass(), ZERO_THRESHOLD,
    _load_matcher_module(), _load_diag_module()
  - matcher.assign_deciles()             (loaded transitively)
  - diag.build_switching_and_pool_tpm()  (loaded transitively)

CONFIRMED SCHEMA (verified directly against compare_decile_structure_v1.py,
Day 62): build_switching_and_pool_tpm() returns
(switching_tpm_df, nonswitching_tpm_df), each with columns [gene_id, mean_tpm].
gene_id is a plain column in both, not an index (compare_decile_structure_v1.py:
137-138, 157-158).

Run from this script's own folder, alongside build_k562_stratified_rebinning_v1.py.

Usage:
    python characterize_decile_shortfall_v1.py --self-test
    python characterize_decile_shortfall_v1.py
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

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

SCRIPT_DIR = Path(__file__).resolve().parent

REBINNING_MODULE_PATH = SCRIPT_DIR / "build_k562_stratified_rebinning_v1.py"

K562_GATE_PATH = "data/k562_gate_assignments_named.tsv"
K562_EXPR_PATH = "data/encode_rnaseq_k562_replicates_v1.csv"

TARGET_DECILES = [0, 1, 2, 3]
OUT_PATH = SCRIPT_DIR / "k562_decile03_shortfall_characterization_v1.tsv"

EXPECTED_TPM_COLUMNS = {"gene_id", "mean_tpm"}


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
    """
    Confirmed schema per compare_decile_structure_v1.py:137-138 -- gene_id
    and mean_tpm are both plain columns. sys.exit() with the real columns
    on any mismatch, never silent coercion.
    """
    actual = set(df.columns)
    if not EXPECTED_TPM_COLUMNS.issubset(actual):
        sys.exit(
            f"SCHEMA MISMATCH in {label}\n"
            f"  expected (subset of): {sorted(EXPECTED_TPM_COLUMNS)}\n"
            f"  actual columns:       {sorted(actual)}"
        )


def compute_decile03_mannwhitney(gene_table: pd.DataFrame):
    """
    gene_table: DataFrame with columns [gene_id, mean_tpm, decile, is_switching],
    already restricted to the nonzero remainder.

    Returns dict with U statistic, p-value, group sizes, and per-decile counts,
    computed only over TARGET_DECILES.
    """
    sub = gene_table[gene_table["decile"].isin(TARGET_DECILES)].copy()

    switching_tpm = sub.loc[sub["is_switching"] == True, "mean_tpm"].to_numpy()
    nonswitching_tpm = sub.loc[sub["is_switching"] == False, "mean_tpm"].to_numpy()

    if len(switching_tpm) == 0 or len(nonswitching_tpm) == 0:
        sys.exit(
            f"FATAL: one group is empty in deciles {TARGET_DECILES}: "
            f"switching n={len(switching_tpm)}, nonswitching n={len(nonswitching_tpm)}. "
            f"Check that both switch_tpm and pool_tpm nonzero remainders actually "
            f"produced genes in this decile range before treating this as a result."
        )

    u_stat, p_val = mannwhitneyu(switching_tpm, nonswitching_tpm, alternative="two-sided")

    per_decile_counts = (
        sub.groupby(["decile", "is_switching"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[True, False], fill_value=0)
        .rename(columns={True: "switching", False: "nonswitching"})
    )
    per_decile_counts["total"] = per_decile_counts.sum(axis=1)

    return {
        "u_statistic": u_stat,
        "p_value": p_val,
        "n_switching": len(switching_tpm),
        "n_nonswitching": len(nonswitching_tpm),
        "per_decile_counts": per_decile_counts,
    }


def synthetic_self_test():
    """
    Hand-derivable fixture, built directly against compute_decile03_mannwhitney()
    -- isolated from matcher/diag/rebin_mod entirely, so a self-test failure here
    can never be confused with a bug in the imported real pipeline functions.

    12 synthetic genes across deciles 0-3. Switching-group TPMs are
    deliberately shifted higher than nonswitching within these deciles,
    so the expected result is a significant Mann-Whitney separation
    (p < 0.05) with a known direction.
    """
    fixture = pd.DataFrame({
        "gene_id": [f"SYN{i:02d}" for i in range(1, 13)],
        "decile":       [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        "is_switching": [True, False, False, True, False, False,
                          True, False, True, False, True, False],
        "mean_tpm": [
            5.0, 1.0, 1.2,   # decile 0
            6.0, 1.5, 1.1,   # decile 1
            7.0, 1.3, 6.5,   # decile 2
            1.4, 7.5, 1.6,   # decile 3
        ],
    })

    expected_counts = (
        fixture.groupby(["decile", "is_switching"]).size()
        .unstack(fill_value=0).reindex(columns=[True, False], fill_value=0)
    )
    expected_switching_n = int((fixture["is_switching"] == True).sum())
    expected_nonswitching_n = int((fixture["is_switching"] == False).sum())

    result = compute_decile03_mannwhitney(fixture)

    if result["n_switching"] != expected_switching_n:
        print(f"SELF-TEST FAILED: n_switching mismatch. got={result['n_switching']} "
              f"expected={expected_switching_n}")
        sys.exit(1)

    if result["n_nonswitching"] != expected_nonswitching_n:
        print(f"SELF-TEST FAILED: n_nonswitching mismatch. got={result['n_nonswitching']} "
              f"expected={expected_nonswitching_n}")
        sys.exit(1)

    if result["p_value"] >= 0.05:
        print(f"SELF-TEST FAILED: expected significant separation (p<0.05), "
              f"got p={result['p_value']}")
        sys.exit(1)

    for decile in TARGET_DECILES:
        got_row = result["per_decile_counts"].loc[decile]
        exp_row = expected_counts.loc[decile]
        for col_bool, col_name in [(True, "switching"), (False, "nonswitching")]:
            got_val = got_row[col_name]
            exp_val = exp_row[col_bool]
            if got_val != exp_val:
                print(f"SELF-TEST FAILED: decile {decile} {col_name} count mismatch. "
                      f"got={got_val} expected={exp_val}")
                sys.exit(1)

    print("SELF-TEST PASSED: Mann-Whitney direction and per-decile counts confirmed "
          "on hand-derived fixture.")


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
    gene_table = pd.concat([sw_table, ns_table], ignore_index=True)

    result = compute_decile03_mannwhitney(gene_table)

    print(f"\n=== K562 Decile {TARGET_DECILES} Switching vs Non-Switching mean_tpm ===")
    print(f"Mann-Whitney U statistic: {result['u_statistic']}")
    print(f"p-value: {result['p_value']}")
    print(f"n switching: {result['n_switching']}")
    print(f"n nonswitching: {result['n_nonswitching']}")
    print("\nPer-decile counts:")
    print(result["per_decile_counts"])

    result["per_decile_counts"].to_csv(OUT_PATH, sep="\t")
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