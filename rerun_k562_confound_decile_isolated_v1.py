"""
rerun_k562_confound_decile_isolated_v1.py

Day 63 | Logic Circuits | Block 2

Purpose
-------
Day 62 Block 4 (characterize_decile_shortfall_v1.py) found K562 switching
genes disproportionately concentrated in deciles 0-3 of the nonzero
remainder (57.6% vs 27.7% non-switching, Mann-Whitney p=0.0006746) --
exactly the range where Day 61 found the 827-gene non-switching pool
shortfall. That is a correlation with the confound's location, not yet a
test of whether it DRIVES the confound (KS p=2.623e-11, Day 61's
zero-stratified match, currently the worst result in the lineage).

This script tests pool-concentration as the confound driver directly, via
two isolation conditions run against the SAME zero-stratified match Day 61
produced:

  Condition A (exclusion): rerun the Hansen KS check with deciles 0-3
    excluded entirely from BOTH switching and matched non-switching sets
    (ZERO stratum and deciles 4-9 retained).

  Condition B (down-weighting): rerun with deciles 0-3 retained, but the
    switching set's representation in deciles 0-3 down-sampled to match
    the non-switching pool's own proportion there (27.7%), rather than
    excluding the range outright. This tests whether equalizing the
    CONCENTRATION (not removing the range) is sufficient to collapse the
    KS statistic -- a stricter, more surgical version of Condition A.

If either condition's KS p-value collapses toward GM12878's clean baseline
(p=0.3917), pool-concentration is confirmed as the confound driver. If
neither moves the statistic meaningfully, pool-concentration stands as a
real, named, but non-driving feature -- parallel to zero-mass (Day 61) and
scale mismatch (Day 62) -- and the investigation pivots toward a
structural expression-distribution-shape difference between the two cell
lines, per the Day 63 forward note.

Code reuse (no logic re-implemented; all imported via
importlib.util.spec_from_file_location per project convention):
  - sgev.load_and_validate, sgev.REQUIRED_GATE_COLS, sgev.build_switching_sets,
    sgev.compute_noise_residual         <- code/switching_gene_expression_variability_v1.py
  - matcher.assign_deciles, matcher.match_by_decile
                                          <- code/matched_nonswitching_v1.py
  - rebinner.stratify_zero_mass         <- build_k562_stratified_rebinning_v1.py
  - confound_v2.match_k562_zero_stratified, confound_v2.retain_switching_per_stratum
                                          <- rerun_k562_confound_check_v2.py

This is the SAME match Day 61/62 produced -- reusing match_k562_zero_stratified()
and retain_switching_per_stratum() directly (rather than reconstructing the
zero-stratified match here) means Condition A/B are true isolation tests on
top of the real matched population, not a new match with its own drift.

NEW logic introduced here (covered by this script's own self-test, since it
does not exist in any imported module):
  - build_condition_a_exclude(): drops rows whose stratum is in a given
    exclusion set from both matched and retained tables.
  - build_condition_b_downweight(): resamples the switching (retained) set
    so its proportion within a given target-stratum set matches the
    non-switching (matched) set's proportion there, leaving all other
    strata untouched.

CONFIRMED SCHEMA (via rerun_k562_confound_check_v2.py, read directly before
writing this script):
  match_k562_zero_stratified() returns (matched_df, edges) where matched_df
  has columns [gene_id, mean_tpm, stratum]; stratum is the string "ZERO" or
  an int decile label 0-9.
  retain_switching_per_stratum() returns retained_df with the same three
  columns, same stratum semantics, one row per retained switching gene.
  sgev.compute_noise_residual(expr, "gene_id", rep_cols) returns a frame
  with columns including gene_id, mean_expr, noise_residual (confirmed via
  rerun_k562_confound_check_v2.py:317-334, where mean_expr is the field the
  real Hansen KS check is run on -- NOT noise_residual, which feeds the
  separate Mann-Whitney check).

Outputs
-------
  data/k562_decile_isolation_results_v1.csv
      One row per condition: baseline (Day 61 zero-stratified),
      Condition A (excluded), Condition B (down-weighted), plus the Day 56
      flat-decile and GM12878 clean-baseline numbers for full-lineage
      reference. Columns: condition, ks_stat, ks_p, n_switching,
      n_nonswitching, notes.

Usage
-----
    python rerun_k562_confound_decile_isolated_v1.py --self-test
    python rerun_k562_confound_decile_isolated_v1.py
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
from scipy.stats import ks_2samp

GATE_SCRIPT_PATH = "code/switching_gene_expression_variability_v1.py"
MATCHER_SCRIPT_PATH = "code/matched_nonswitching_v1.py"
REBINNER_SCRIPT_PATH = "build_k562_stratified_rebinning_v1.py"
CONFOUND_V2_SCRIPT_PATH = "rerun_k562_confound_check_v2.py"

K562_GATE_PATH = "data/k562_gate_assignments_named.tsv"
GM12878_GATE_PATH = "data/gate_assignments_named.tsv"
K562_EXPR_PATH = "data/encode_rnaseq_k562_replicates_v1.csv"

ZERO_THRESHOLD = 0.1  # locked Day 61, unchanged here -- see rerun_k562_confound_check_v2.py
RNG_SEED = 63  # new seed for this new procedure, following project convention
               # (Day 56 used 56, Day 61 used 61) -- not reused from either.

TARGET_STRATA = [0, 1, 2, 3]  # the deciles under test today

OUT_PATH = "data/k562_decile_isolation_results_v1.csv"

# Full-lineage reference numbers, hardcoded for direct table inclusion --
# same convention rerun_k562_confound_check_v2.py used for its Day 56
# comparison. If these ever change upstream, update here explicitly rather
# than re-reading from disk, so this script's self-test never depends on
# prior days' output files existing.
DAY56_FLAT_DECILE_KS_P = 0.0004043
DAY61_ZERO_STRATIFIED_KS_P_REFERENCE = 2.623e-11  # what we expect to reproduce as baseline
GM12878_CLEAN_BASELINE_KS_P = 0.3917


def _load_module(path: str, name: str):
    if not os.path.exists(path):
        sys.exit(
            f"FATAL: cannot find {path} to import {name} from. Not proceeding "
            f"with a re-implemented copy of that logic -- import the real one."
        )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_columns(df: pd.DataFrame, expected: set, label: str) -> None:
    actual = set(df.columns)
    if not expected.issubset(actual):
        sys.exit(
            f"SCHEMA MISMATCH in {label}\n"
            f"  expected (subset of): {sorted(expected)}\n"
            f"  actual columns:       {sorted(actual)}"
        )


def build_condition_a_exclude(matched: pd.DataFrame, retained: pd.DataFrame,
                               exclude_strata: list):
    """
    Drops rows whose stratum is in exclude_strata from BOTH tables. ZERO
    stratum and any decile outside exclude_strata are untouched.

    Returns (matched_filtered, retained_filtered).
    """
    matched_f = matched[~matched["stratum"].isin(exclude_strata)].copy()
    retained_f = retained[~retained["stratum"].isin(exclude_strata)].copy()
    return matched_f, retained_f


def build_condition_b_downweight(matched: pd.DataFrame, retained: pd.DataFrame,
                                  target_strata: list, rng: np.random.Generator):
    """
    Resamples the switching (retained) set so its proportion within
    target_strata matches the non-switching (matched) set's proportion
    there. All rows outside target_strata are kept untouched. The
    non-switching (matched) set is never modified -- only switching's
    over-concentration is corrected, since that is the asymmetry Day 62
    characterized (57.6% vs 27.7%).

    Returns (matched_unchanged, retained_downweighted, changed: bool).
    changed is False if the target proportion already matches or the
    switching set is already at/below the reference proportion (nothing
    to down-weight -- resampling can only remove rows, never add them).
    """
    is_target_sw = retained["stratum"].isin(target_strata)
    is_target_ns = matched["stratum"].isin(target_strata)

    n_total_sw = len(retained)
    n_target_sw = int(is_target_sw.sum())
    ns_target_prop = is_target_ns.mean() if len(matched) > 0 else 0.0

    target_n_sw_in_target = int(round(ns_target_prop * n_total_sw))

    if target_n_sw_in_target >= n_target_sw:
        return matched.copy(), retained.copy(), False

    target_pool = retained[is_target_sw]
    keep_idx = rng.choice(
        target_pool.index.to_numpy(), size=target_n_sw_in_target, replace=False
    )
    kept_target = retained.loc[keep_idx]
    kept_rest = retained[~is_target_sw]
    downweighted = pd.concat([kept_rest, kept_target], ignore_index=True)

    return matched.copy(), downweighted, True


def run_ks_check(matched: pd.DataFrame, retained: pd.DataFrame, noise_df: pd.DataFrame):
    """
    Given a (possibly filtered/resampled) matched/retained pair and the
    full noise_df (from sgev.compute_noise_residual, unfiltered), looks up
    mean_expr for each group's gene_id set and runs the Hansen KS check --
    identical structure to rerun_k562_confound_check_v2.py:330-334.

    Returns dict with ks_stat, ks_p, n_switching, n_nonswitching.
    """
    sw_ids = set(retained["gene_id"])
    ns_ids = set(matched["gene_id"])

    sw_mean_expr = noise_df[noise_df["gene_id"].isin(sw_ids)]["mean_expr"].dropna()
    ns_mean_expr = noise_df[noise_df["gene_id"].isin(ns_ids)]["mean_expr"].dropna()

    if len(sw_mean_expr) < 5 or len(ns_mean_expr) < 5:
        sys.exit(
            f"FATAL: too few genes with valid mean_expr after filtering "
            f"(switching n={len(sw_mean_expr)}, non-switching n={len(ns_mean_expr)}). "
            f"Not proceeding with an under-powered comparison."
        )

    ks_stat, ks_p = ks_2samp(sw_mean_expr, ns_mean_expr)
    return {
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
        "n_switching": len(sw_mean_expr),
        "n_nonswitching": len(ns_mean_expr),
    }


def synthetic_self_test():
    """
    Hand-derivable fixture for the two NEW functions only
    (build_condition_a_exclude, build_condition_b_downweight) --
    match_k562_zero_stratified(), retain_switching_per_stratum(), and
    ks_2samp() are already covered by their own scripts' self-tests / are
    scipy-tested, so this checks only the filtering/resampling logic added
    today.

    Fixture: matched (non-switching) has 10 rows -- 2 in ZERO, 6 spread
    across deciles {0,1,2,3} (60%), 2 in deciles {4,5}.
    retained (switching) has 10 rows -- 1 in ZERO, 8 spread across
    deciles {0,1,2,3} (80%), 1 in decile {5}.

    Condition A (exclude {0,1,2,3}): expect matched_f to have 2+2=4 rows,
    retained_f to have 1+1=2 rows.

    Condition B (down-weight to match non-switching's 60% in {0,1,2,3}):
    switching's target proportion should drop from 8/10=80% to
    round(0.6*10)=6 rows in {0,1,2,3}, total retained count 6+2=8, and
    matched must be UNCHANGED (still 10 rows).
    """
    matched = pd.DataFrame({
        "gene_id": [f"NS{i}" for i in range(10)],
        "mean_tpm": np.linspace(0.5, 40, 10),
        "stratum": ["ZERO", "ZERO", 0, 1, 2, 3, 0, 1, 4, 5],
    })
    retained = pd.DataFrame({
        "gene_id": [f"SW{i}" for i in range(10)],
        "mean_tpm": np.linspace(0.5, 40, 10),
        "stratum": ["ZERO", 0, 0, 1, 1, 2, 2, 3, 3, 5],
    })

    # --- Condition A ---
    matched_a, retained_a = build_condition_a_exclude(matched, retained, TARGET_STRATA)
    if len(matched_a) != 4:
        print(f"SELF-TEST FAILED: Condition A matched_f expected 4 rows, got {len(matched_a)}")
        sys.exit(1)
    if len(retained_a) != 2:
        print(f"SELF-TEST FAILED: Condition A retained_f expected 2 rows, got {len(retained_a)}")
        sys.exit(1)
    if set(matched_a["stratum"]) - {"ZERO", 4, 5} != set():
        print(f"SELF-TEST FAILED: Condition A matched_f contains unexpected strata: "
              f"{set(matched_a['stratum'])}")
        sys.exit(1)

    # --- Condition B ---
    rng = np.random.default_rng(RNG_SEED)
    matched_b, retained_b, changed = build_condition_b_downweight(
        matched, retained, TARGET_STRATA, rng
    )
    if not changed:
        print("SELF-TEST FAILED: Condition B expected changed=True on this fixture "
              "(switching starts more concentrated than non-switching in target strata).")
        sys.exit(1)
    if len(matched_b) != len(matched):
        print(f"SELF-TEST FAILED: Condition B must leave matched (non-switching) "
              f"unchanged. expected {len(matched)} rows, got {len(matched_b)}.")
        sys.exit(1)
    n_target_b = retained_b["stratum"].isin(TARGET_STRATA).sum()
    if n_target_b != 6:
        print(f"SELF-TEST FAILED: Condition B expected 6 switching rows in target "
              f"strata after down-weighting to non-switching's 60% proportion, got {n_target_b}.")
        sys.exit(1)
    if len(retained_b) != 8:
        print(f"SELF-TEST FAILED: Condition B expected total retained count 8 "
              f"(6 target + 2 untouched), got {len(retained_b)}.")
        sys.exit(1)

    # --- Condition B no-op case: target already below reference proportion ---
    retained_already_low = pd.DataFrame({
        "gene_id": [f"SW{i}" for i in range(10)],
        "mean_tpm": np.linspace(0.5, 40, 10),
        "stratum": ["ZERO", 0, 4, 4, 5, 5, 6, 6, 7, 7],  # only 1/10 = 10% in {0,1,2,3}
    })
    _, retained_noop, changed_noop = build_condition_b_downweight(
        matched, retained_already_low, TARGET_STRATA, rng
    )
    if changed_noop:
        print("SELF-TEST FAILED: Condition B expected changed=False when switching's "
              "proportion in target strata is already at/below non-switching's -- "
              "resampling should never ADD rows.")
        sys.exit(1)
    if len(retained_noop) != len(retained_already_low):
        print("SELF-TEST FAILED: Condition B no-op case must return the set unchanged.")
        sys.exit(1)

    print("SELF-TEST PASSED: Condition A exclusion and Condition B down-weighting "
          "(including the no-op guard) behave as hand-derived on the toy fixture.")


def run_real_data():
    sgev = _load_module(GATE_SCRIPT_PATH, "sgev")
    matcher = _load_module(MATCHER_SCRIPT_PATH, "matcher")
    rebinner = _load_module(REBINNER_SCRIPT_PATH, "rebinner")
    confound_v2 = _load_module(CONFOUND_V2_SCRIPT_PATH, "confound_v2")

    gm_gate_df = sgev.load_and_validate(GM12878_GATE_PATH, sgev.REQUIRED_GATE_COLS, sep="\t")
    k562_gate_df = sgev.load_and_validate(K562_GATE_PATH, sgev.REQUIRED_GATE_COLS, sep="\t")
    switching, stable, _ = sgev.build_switching_sets(gm_gate_df, k562_gate_df)

    expr = pd.read_csv(K562_EXPR_PATH)
    expr["mean_tpm"] = expr[["rep1", "rep2"]].mean(axis=1)

    switching_tpm = expr[["gene_id", "mean_tpm"]].merge(
        switching[["gene_id"]], on="gene_id", how="inner"
    )
    nonswitching_tpm = expr[["gene_id", "mean_tpm"]].merge(
        stable[["gene_id"]], on="gene_id", how="inner"
    )

    print(f"K562 switching genes with expression data: {len(switching_tpm)} (of {len(switching)})")
    print(f"K562 non-switching pool with expression data: {len(nonswitching_tpm)} "
          f"(of {len(stable)})")

    # --- Reproduce the exact Day 61 zero-stratified match, unmodified ---
    rng = np.random.default_rng(RNG_SEED)
    matched, edges = confound_v2.match_k562_zero_stratified(
        matcher, rebinner, switching_tpm, nonswitching_tpm, ZERO_THRESHOLD, rng
    )
    retained = confound_v2.retain_switching_per_stratum(
        switching_tpm, matched, ZERO_THRESHOLD, edges, rng
    )

    _check_columns(matched, {"gene_id", "mean_tpm", "stratum"}, "matched (Day 61 reproduction)")
    _check_columns(retained, {"gene_id", "mean_tpm", "stratum"}, "retained (Day 61 reproduction)")

    rep_cols = ["rep1", "rep2"]
    noise_df = sgev.compute_noise_residual(expr, "gene_id", rep_cols)
    _check_columns(noise_df, {"gene_id", "mean_expr"}, "noise_df (compute_noise_residual output)")

    results_rows = []

    # --- Baseline: reproduce Day 61's zero-stratified result exactly ---
    baseline = run_ks_check(matched, retained, noise_df)
    print(f"\n=== BASELINE (Day 61 zero-stratified, reproduced) ===")
    print(f"  KS={baseline['ks_stat']:.4f}, p={baseline['ks_p']:.4g}, "
          f"n_switching={baseline['n_switching']}, n_nonswitching={baseline['n_nonswitching']}")
    if abs(baseline["ks_p"] - DAY61_ZERO_STRATIFIED_KS_P_REFERENCE) / DAY61_ZERO_STRATIFIED_KS_P_REFERENCE > 0.05:
        print(
            f"  WARNING: reproduced baseline p={baseline['ks_p']:.4g} deviates from "
            f"Day 61's reported {DAY61_ZERO_STRATIFIED_KS_P_REFERENCE:.4g} by >5%. "
            f"Check RNG seed / stale-file risk before trusting Conditions A/B below -- "
            f"they are only meaningful relative to a correctly-reproduced baseline."
        )
    results_rows.append({
        "condition": "baseline_day61_zero_stratified",
        "ks_stat": baseline["ks_stat"], "ks_p": baseline["ks_p"],
        "n_switching": baseline["n_switching"], "n_nonswitching": baseline["n_nonswitching"],
        "notes": "reproduction of Day 61's zero-stratified match, unmodified",
    })

    # --- Condition A: exclude deciles 0-3 entirely ---
    matched_a, retained_a = build_condition_a_exclude(matched, retained, TARGET_STRATA)
    cond_a = run_ks_check(matched_a, retained_a, noise_df)
    print(f"\n=== CONDITION A: deciles {TARGET_STRATA} excluded ===")
    print(f"  KS={cond_a['ks_stat']:.4f}, p={cond_a['ks_p']:.4g}, "
          f"n_switching={cond_a['n_switching']}, n_nonswitching={cond_a['n_nonswitching']}")
    results_rows.append({
        "condition": "condition_a_exclude_deciles_0_3",
        "ks_stat": cond_a["ks_stat"], "ks_p": cond_a["ks_p"],
        "n_switching": cond_a["n_switching"], "n_nonswitching": cond_a["n_nonswitching"],
        "notes": f"deciles {TARGET_STRATA} removed from both groups; ZERO + deciles 4-9 retained",
    })

    # --- Condition B: down-weight switching's concentration in deciles 0-3 ---
    rng_b = np.random.default_rng(RNG_SEED)
    matched_b, retained_b, changed_b = build_condition_b_downweight(
        matched, retained, TARGET_STRATA, rng_b
    )
    cond_b = run_ks_check(matched_b, retained_b, noise_df)
    print(f"\n=== CONDITION B: switching down-weighted to non-switching's decile "
          f"{TARGET_STRATA} proportion ===")
    print(f"  changed={changed_b}")
    print(f"  KS={cond_b['ks_stat']:.4f}, p={cond_b['ks_p']:.4g}, "
          f"n_switching={cond_b['n_switching']}, n_nonswitching={cond_b['n_nonswitching']}")
    if not changed_b:
        print(
            "  NOTE: down-weighting was a no-op on real data -- switching's proportion "
            "in deciles 0-3 was already at or below non-switching's. This itself is a "
            "finding worth logging: it would mean the 57.6%/27.7% gap characterized on "
            "Day 62 does not survive into the zero-stratified RETAINED set the same way "
            "it appeared in the raw nonzero remainder."
        )
    results_rows.append({
        "condition": "condition_b_downweight_deciles_0_3",
        "ks_stat": cond_b["ks_stat"], "ks_p": cond_b["ks_p"],
        "n_switching": cond_b["n_switching"], "n_nonswitching": cond_b["n_nonswitching"],
        "notes": f"switching resampled to match non-switching's proportion in "
                 f"deciles {TARGET_STRATA}; changed={changed_b}",
    })

    # --- Full-lineage reference rows ---
    results_rows.append({
        "condition": "day56_flat_decile_reference",
        "ks_stat": np.nan, "ks_p": DAY56_FLAT_DECILE_KS_P,
        "n_switching": np.nan, "n_nonswitching": np.nan,
        "notes": "reference only, not rerun today",
    })
    results_rows.append({
        "condition": "gm12878_clean_baseline_reference",
        "ks_stat": np.nan, "ks_p": GM12878_CLEAN_BASELINE_KS_P,
        "n_switching": np.nan, "n_nonswitching": np.nan,
        "notes": "reference only, not rerun today -- target for collapse comparison",
    })

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(OUT_PATH, index=False)
    print(f"\n=== FULL COMPARISON TABLE ===")
    print(results_df.to_string(index=False))
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