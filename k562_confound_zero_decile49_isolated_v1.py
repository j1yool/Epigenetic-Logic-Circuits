"""
k562_confound_zero_decile49_isolated_v1.py

Day 64 | Logic Circuits | Block 2

Purpose
-------
Day 63's Condition A (excluding deciles 0-3, retaining ZERO+4-9) made the
KS p-value FIVE ORDERS OF MAGNITUDE WORSE (per Day 63 forward note), ruling
out low-TPM pool-concentration in deciles 0-3 as the confound driver and
pointing at ZERO and/or deciles 4-9 as the real location of the mismatch.
Day 63's Condition A tested them as an undifferentiated complement (ZERO
and 4-9 lumped together as "everything retained"). Today isolates them
from EACH OTHER.

Four new conditions plus one direct cross-check, all built on the SAME
zero-stratified match, reproduced fresh with RNG_SEED=61 (correcting Day
63's own seed-63 approximation, per the Day 64 forward note -- Day 63 used
RNG_SEED=63 for a "new procedure" naming convention, which means its own
baseline was only an approximate reproduction of Day 61's exact 2.623e-11
figure, not an exact one):

  BASELINE              -- full retained population, unmodified.
  CONDITION_B            -- exclude ZERO stratum only (retain deciles 0-9).
  CONDITION_C            -- exclude deciles 4-9 only (retain ZERO + 0-3).
  CONDITION_D            -- exclude ZERO AND deciles 4-9 (retain ONLY 0-3).
  CONDITION_A_RESEEDED   -- exclude deciles 0-3 only (retain ZERO + 4-9).
                             This is Day 63's Condition A definition
                             exactly, rerun on the seed-61 baseline for a
                             true apples-to-apples comparison against Day
                             63's reported result (which used seed 63).
                             NOTE: this is NOT the same set as CONDITION_D
                             -- excluding {0,1,2,3} and retaining ONLY
                             {0,1,2,3} are complements of each other, not
                             equivalent tests. An earlier draft of this
                             script incorrectly claimed CONDITION_D would
                             "match Day 63's Condition A" -- it does not,
                             and that claim has been removed.
                             CONDITION_A_RESEEDED is the correct cross-check.

Interpretation logic (applied in the verdict doc, not in this script):
  - If CONDITION_B moves sharply toward GM12878's clean regime (ks_p
    approaching 0.3917) while CONDITION_C stays near-baseline -> mismatch
    concentrates in the ZERO stratum.
  - If the reverse -> mismatch concentrates in deciles 4-9.
  - If both individually improve and only CONDITION_D (excluding both)
    restores clean behavior -> mismatch is distributed across both.
  - If none of B/C/D move the needle -> ZERO/deciles-4-9 isolation is
    ruled out too, and this becomes a fourth named negative result.

Code reuse (no logic re-implemented; all imported via
importlib.util.spec_from_file_location per project convention):
  - day63.build_condition_a_exclude   <- rerun_k562_confound_decile_isolated_v1.py
                                          (Day 63). Already generic over an
                                          arbitrary exclude_strata list --
                                          NOT hardcoded to deciles 0-3 --
                                          so it is reused directly for all
                                          five conditions below rather than
                                          reimplemented.
  - day63.run_ks_check                <- same file. Already validated
                                          against ks_2samp on Day 63's own
                                          self-test; reused as-is.
  - day63._check_columns              <- same file. Trivial schema-guard
                                          helper, reused rather than
                                          duplicated.
  - sgev.load_and_validate, sgev.REQUIRED_GATE_COLS, sgev.build_switching_sets,
    sgev.compute_noise_residual        <- code/switching_gene_expression_variability_v1.py
  - matcher (module, passed through)   <- code/matched_nonswitching_v1.py
  - rebinner (module, passed through)  <- build_k562_stratified_rebinning_v1.py
  - confound_v2.match_k562_zero_stratified, confound_v2.retain_switching_per_stratum
                                        <- rerun_k562_confound_check_v2.py
    CONFIRMED SIGNATURES (read directly from Day 63's script, not assumed):
      match_k562_zero_stratified(matcher, rebinner, switching_tpm,
                                  nonswitching_tpm, ZERO_THRESHOLD, rng)
          -> (matched_df, edges); matched_df columns [gene_id, mean_tpm, stratum]
      retain_switching_per_stratum(switching_tpm, matched_df, ZERO_THRESHOLD,
                                    edges, rng)
          -> retained_df, same three columns, same stratum semantics.

NEW logic introduced here (covered by this script's own self-test):
  - the five exclude_strata lists themselves (ZERO-only, HIGH-only,
    ZERO+HIGH, LOW-only exclusion for the reseeded Condition-A replicate)
  - sigfig_tolerance_check(): pure comparison helper.
  - reproduction gate in run_real_data(): hard-stops the run if the
    freshly reproduced baseline's ks_p does not match Day 61's exact
    2.623e-11 to 4 significant figures, per the Day 64 forward note's
    requirement that isolation deltas be measured against the exact
    figure, not an approximation.

CONFIRMED SCHEMA note carried forward from Day 63: mean_expr (from
sgev.compute_noise_residual) is the field the Hansen KS check runs on --
NOT noise_residual, which feeds a separate Mann-Whitney check elsewhere.

Outputs
-------
  data/k562_zero_decile49_isolation_results_v1.csv
      One row per condition: baseline, condition_b (exclude ZERO),
      condition_c (exclude deciles 4-9), condition_d (exclude both,
      retain LOW only), condition_a_reseeded (exclude LOW, retain
      ZERO+HIGH -- Day 63 Condition A rerun on seed 61).
      Columns: condition, ks_stat, ks_p, n_switching, n_nonswitching, notes.

Usage
-----
    python rerun_k562_confound_zero_decile49_isolated_v1.py --self-test
    python rerun_k562_confound_zero_decile49_isolated_v1.py
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

DAY63_SCRIPT_PATH = "rerun_k562_confound_decile_isolated_v1.py"

RNG_SEED_BASELINE = 61  # exact Day 61 seed -- correcting Day 63's seed-63
                         # approximation, per the Day 64 forward note.

DAY61_ZERO_STRATIFIED_KS_P_REFERENCE = 2.623e-11  # this is a P-VALUE, not
                                                    # a KS statistic -- Day 63's
                                                    # own script labels it
                                                    # KS_P_REFERENCE.
BASELINE_TOLERANCE_SIGFIGS = 4

LOW_DECILES = [0, 1, 2, 3]
HIGH_DECILES = [4, 5, 6, 7, 8, 9]
ZERO_LABEL = ["ZERO"]

OUT_PATH = "data/k562_zero_decile49_isolation_results_v1.csv"


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


def sigfig_tolerance_check(observed: float, expected: float, sigfigs: int) -> tuple:
    """
    Returns (passes: bool, relative_diff: float). Pure function, independent
    of any real data, so it can be exercised directly in the self-test.
    """
    if expected == 0:
        sys.exit("FATAL: expected reference value is zero -- tolerance check undefined.")
    rel_diff = abs(observed - expected) / abs(expected)
    tolerance = 10 ** (-sigfigs)
    return rel_diff <= tolerance, rel_diff


def synthetic_self_test(day63):
    """
    Hand-derivable fixture covering the NEW logic added today:
      1. Each of the four exclude_strata lists, applied via Day 63's own
         (already self-tested) build_condition_a_exclude(), partitions a
         known toy population to hand-computed counts.
      2. CONDITION_D (retain LOW only) and CONDITION_A_RESEEDED (retain
         ZERO+HIGH) are genuine complements of each other on the fixture --
         confirming they are NOT the same test, which is the exact mistake
         this script corrects from an earlier draft.
      3. sigfig_tolerance_check() correctly passes a near-match and fails
         a far-match.
      4. day63.run_ks_check(), reused unmodified, reproduces an
         independently-computed scipy ks_2samp() result on the fixture.

    Fixture: matched (non-switching) has 12 rows -- 3 ZERO, 4 in LOW
    deciles {0,1,2,3}, 5 in HIGH deciles {4..9}.
    retained (switching) has 10 rows -- 2 ZERO, 3 in LOW, 5 in HIGH.
    """
    matched = pd.DataFrame({
        "gene_id": [f"NS{i}" for i in range(12)],
        "mean_tpm": np.linspace(0.5, 48, 12),
        "stratum": ["ZERO", "ZERO", "ZERO", 0, 1, 2, 3, 4, 5, 6, 7, 8],
    })
    retained = pd.DataFrame({
        "gene_id": [f"SW{i}" for i in range(10)],
        "mean_tpm": np.linspace(1.0, 40, 10),
        "stratum": ["ZERO", "ZERO", 0, 1, 2, 4, 5, 6, 7, 8],
    })
    noise_df = pd.DataFrame({
        "gene_id": [f"NS{i}" for i in range(12)] + [f"SW{i}" for i in range(10)],
        "mean_expr": np.concatenate([
            np.linspace(0.2, 5.0, 12),   # non-switching mean_expr
            np.linspace(3.0, 9.0, 10),   # switching mean_expr -- deliberately
                                          # shifted so KS is nonzero and unambiguous
        ]),
    })

    # --- Test 1: partition counts for each of the four conditions ---
    expected_counts = {
        "CONDITION_B (exclude ZERO)": {"matched": 9, "retained": 8},
        "CONDITION_C (exclude HIGH)": {"matched": 7, "retained": 5},
        "CONDITION_D (exclude ZERO+HIGH, retain LOW only)": {"matched": 4, "retained": 3},
        "CONDITION_A_RESEEDED (exclude LOW, retain ZERO+HIGH)": {"matched": 8, "retained": 7},
    }
    condition_excludes = {
        "CONDITION_B (exclude ZERO)": ZERO_LABEL,
        "CONDITION_C (exclude HIGH)": HIGH_DECILES,
        "CONDITION_D (exclude ZERO+HIGH, retain LOW only)": ZERO_LABEL + HIGH_DECILES,
        "CONDITION_A_RESEEDED (exclude LOW, retain ZERO+HIGH)": LOW_DECILES,
    }
    for cond_name, exclude_strata in condition_excludes.items():
        m_f, r_f = day63.build_condition_a_exclude(matched, retained, exclude_strata)
        exp = expected_counts[cond_name]
        if len(m_f) != exp["matched"] or len(r_f) != exp["retained"]:
            sys.exit(
                f"SELF-TEST FAILURE: {cond_name} expected "
                f"(matched={exp['matched']}, retained={exp['retained']}), got "
                f"(matched={len(m_f)}, retained={len(r_f)})."
            )

    # --- Test 2: CONDITION_D and CONDITION_A_RESEEDED must be genuine
    # complements, NOT the same set -- this is the exact error corrected
    # from an earlier draft, so it is asserted explicitly here. ---
    m_d, r_d = day63.build_condition_a_exclude(matched, retained, ZERO_LABEL + HIGH_DECILES)
    m_ar, r_ar = day63.build_condition_a_exclude(matched, retained, LOW_DECILES)
    if set(m_d["gene_id"]) & set(m_ar["gene_id"]):
        sys.exit(
            "SELF-TEST FAILURE: CONDITION_D and CONDITION_A_RESEEDED matched "
            "sets overlap -- they must be disjoint complements (LOW-only vs. "
            "ZERO+HIGH-only). If they overlap, the exclude_strata lists are wrong."
        )
    if len(m_d) + len(m_ar) != len(matched):
        sys.exit(
            "SELF-TEST FAILURE: CONDITION_D and CONDITION_A_RESEEDED matched "
            "sets do not partition the full matched population -- expected "
            f"{len(matched)} total, got {len(m_d) + len(m_ar)}."
        )

    # --- Test 3: sigfig_tolerance_check() ---
    passes_near, _ = sigfig_tolerance_check(2.6231e-11, 2.623e-11, 4)
    if not passes_near:
        sys.exit("SELF-TEST FAILURE: sigfig_tolerance_check() rejected a value "
                  "that should pass at 4 sig figs.")
    passes_far, _ = sigfig_tolerance_check(3.352e-11, 2.623e-11, 4)
    if passes_far:
        sys.exit("SELF-TEST FAILURE: sigfig_tolerance_check() accepted a value "
                  "that should fail at 4 sig figs (this is Day 63's actual "
                  "seed-63 approximation vs. Day 61's exact figure -- they "
                  "genuinely differ at this precision, which is the whole "
                  "reason today's script reseeds to 61).")

    # --- Test 4: run_ks_check() reused unmodified, checked against
    # independently-computed scipy result on the same fixture ---
    result = day63.run_ks_check(matched, retained, noise_df)
    sw_ids = set(retained["gene_id"])
    ns_ids = set(matched["gene_id"])
    known_stat, known_p = ks_2samp(
        noise_df[noise_df["gene_id"].isin(sw_ids)]["mean_expr"],
        noise_df[noise_df["gene_id"].isin(ns_ids)]["mean_expr"],
    )
    if not np.isclose(result["ks_stat"], known_stat, rtol=1e-10):
        sys.exit(
            f"SELF-TEST FAILURE: run_ks_check() ks_stat ({result['ks_stat']}) "
            f"does not match independently-computed scipy value ({known_stat})."
        )

    print("synthetic_self_test() PASSED: exclusion partitioning, D/A_RESEEDED "
          "complement relationship, tolerance check, and reused run_ks_check() "
          "all verified against hand-derivable fixtures.")


def run_condition(name, exclude_strata, matched, retained, noise_df, day63, notes=""):
    m_f, r_f = day63.build_condition_a_exclude(matched, retained, exclude_strata)
    result = day63.run_ks_check(m_f, r_f, noise_df)
    print(f"\n=== {name} (excluding strata: {exclude_strata}) ===")
    print(f"  KS={result['ks_stat']:.4f}, p={result['ks_p']:.4g}, "
          f"n_switching={result['n_switching']}, n_nonswitching={result['n_nonswitching']}")
    return {
        "condition": name,
        "ks_stat": result["ks_stat"],
        "ks_p": result["ks_p"],
        "n_switching": result["n_switching"],
        "n_nonswitching": result["n_nonswitching"],
        "notes": notes,
    }


def run_real_data(day63):
    sgev = _load_module(day63.GATE_SCRIPT_PATH, "sgev")
    matcher = _load_module(day63.MATCHER_SCRIPT_PATH, "matcher")
    rebinner = _load_module(day63.REBINNER_SCRIPT_PATH, "rebinner")
    confound_v2 = _load_module(day63.CONFOUND_V2_SCRIPT_PATH, "confound_v2")

    gm_gate_df = sgev.load_and_validate(day63.GM12878_GATE_PATH, sgev.REQUIRED_GATE_COLS, sep="\t")
    k562_gate_df = sgev.load_and_validate(day63.K562_GATE_PATH, sgev.REQUIRED_GATE_COLS, sep="\t")
    switching, stable, _ = sgev.build_switching_sets(gm_gate_df, k562_gate_df)

    expr = pd.read_csv(day63.K562_EXPR_PATH)
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

    # --- Reproduce Day 61's exact zero-stratified match: RNG_SEED=61 ---
    rng = np.random.default_rng(RNG_SEED_BASELINE)
    matched, edges = confound_v2.match_k562_zero_stratified(
        matcher, rebinner, switching_tpm, nonswitching_tpm, day63.ZERO_THRESHOLD, rng
    )
    retained = confound_v2.retain_switching_per_stratum(
        switching_tpm, matched, day63.ZERO_THRESHOLD, edges, rng
    )

    day63._check_columns(matched, {"gene_id", "mean_tpm", "stratum"}, "matched (seed-61 reproduction)")
    day63._check_columns(retained, {"gene_id", "mean_tpm", "stratum"}, "retained (seed-61 reproduction)")

    rep_cols = ["rep1", "rep2"]
    noise_df = sgev.compute_noise_residual(expr, "gene_id", rep_cols)
    day63._check_columns(noise_df, {"gene_id", "mean_expr"}, "noise_df (compute_noise_residual output)")

    results_rows = []

    # --- BASELINE: reproduce Day 61's exact figure. HARD HALT if it doesn't
    # match to 4 sig figs -- isolation deltas below are meaningless against
    # an unreproduced baseline. ---
    baseline = day63.run_ks_check(matched, retained, noise_df)
    print(f"\n=== BASELINE (seed=61, exact Day 61 reproduction target) ===")
    print(f"  KS={baseline['ks_stat']:.4f}, p={baseline['ks_p']:.4g}, "
          f"n_switching={baseline['n_switching']}, n_nonswitching={baseline['n_nonswitching']}")

    passes, rel_diff = sigfig_tolerance_check(
        baseline["ks_p"], DAY61_ZERO_STRATIFIED_KS_P_REFERENCE, BASELINE_TOLERANCE_SIGFIGS
    )
    baseline_row = {
        "condition": "baseline_seed61",
        "ks_stat": baseline["ks_stat"], "ks_p": baseline["ks_p"],
        "n_switching": baseline["n_switching"], "n_nonswitching": baseline["n_nonswitching"],
        "notes": "",
    }
    if not passes:
        baseline_row["notes"] = (
            f"DID NOT REPRODUCE Day 61 exactly: expected ks_p={DAY61_ZERO_STRATIFIED_KS_P_REFERENCE:.6e}, "
            f"got {baseline['ks_p']:.6e} (relative diff {rel_diff:.2%})."
        )
        results_rows.append(baseline_row)
        pd.DataFrame(results_rows).to_csv(OUT_PATH, index=False)
        sys.exit(
            f"HALTED: baseline ks_p ({baseline['ks_p']:.6e}) does not reproduce "
            f"Day 61's exact {DAY61_ZERO_STRATIFIED_KS_P_REFERENCE:.6e} to "
            f"{BASELINE_TOLERANCE_SIGFIGS} sig figs (relative diff {rel_diff:.2%}). "
            f"Partial results written to {OUT_PATH}. Diagnose this discrepancy "
            f"before trusting any condition below -- they are only meaningful "
            f"relative to an exactly-reproduced baseline, per the Day 64 "
            f"forward note."
        )
    baseline_row["notes"] = f"Reproduces Day 61 exactly within {BASELINE_TOLERANCE_SIGFIGS} sig figs."
    results_rows.append(baseline_row)
    print(f"  Reproduction confirmed (relative diff {rel_diff:.2e}).")

    # --- CONDITION_B: exclude ZERO only ---
    results_rows.append(run_condition(
        "condition_b_exclude_zero", ZERO_LABEL, matched, retained, noise_df, day63,
        notes="ZERO stratum excluded from both groups; all nonzero deciles 0-9 retained",
    ))

    # --- CONDITION_C: exclude deciles 4-9 only ---
    results_rows.append(run_condition(
        "condition_c_exclude_high", HIGH_DECILES, matched, retained, noise_df, day63,
        notes="deciles 4-9 excluded from both groups; ZERO + deciles 0-3 retained",
    ))

    # --- CONDITION_D: exclude ZERO and deciles 4-9 (retain LOW only) ---
    results_rows.append(run_condition(
        "condition_d_exclude_zero_and_high", ZERO_LABEL + HIGH_DECILES, matched, retained, noise_df, day63,
        notes="ZERO and deciles 4-9 both excluded; ONLY deciles 0-3 retained",
    ))

    # --- CONDITION_A_RESEEDED: exclude deciles 0-3 only (Day 63's Condition
    # A definition exactly, rerun on the seed-61 baseline for a true
    # cross-check against Day 63's reported result) ---
    results_rows.append(run_condition(
        "condition_a_reseeded_exclude_low", LOW_DECILES, matched, retained, noise_df, day63,
        notes="Day 63 Condition A definition (exclude deciles 0-3, retain "
              "ZERO+4-9), rerun with RNG_SEED=61 instead of Day 63's seed=63 "
              "for direct cross-check comparability against Day 63's reported "
              "five-orders-of-magnitude-worse result. NOT equivalent to "
              "condition_d -- these are complements, not duplicates.",
    ))

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

    day63 = _load_module(DAY63_SCRIPT_PATH, "day63")

    print("Running synthetic_self_test() ...")
    synthetic_self_test(day63)
    print("Self-test passed.\n")

    if args.self_test:
        return

    print("Proceeding to real-data run.")
    run_real_data(day63)


if __name__ == "__main__":
    main()