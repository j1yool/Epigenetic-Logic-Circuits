"""
rerun_k562_confound_check_v2.py

Day 61 Block 4 | Epigenetic Logic Circuits

SUPERSEDES: the K562 confound verdict in rerun_matched_comparison_v1.py
(Day 56, KS=confound_ks_stat/confound_ks_p in data/matched_comparison_results_v1.json,
K562 p=0.0004043 -- STILL CONFOUNDED). That flat-decile match is not being
edited; this is a new file per the supersede-don't-append rule. GM12878's
Day 56 result (p=0.3917, resolved) is NOT rerun here -- the forward note
identifies K562 alone as unresolved, and GM12878 showed no decile-1
collapse in Block 2's zero-stratification (25.2% zero-mass share vs
K562's 35.4%), so there is no motivating problem to rerun it against.

Purpose
-------
rerun_matched_comparison_v1.py matched switching genes to non-switching
genes via FLAT deciles (matched_nonswitching_v1.py), then ran a Hansen KS
test on mean_expr (via sgev.compute_noise_residual) to confirm the match
actually removed the expression confound. For K562 it did not (p=0.0004).
Block 2 of Day 61 showed this is very likely driven by K562 having a much
larger zero/near-zero-TPM gene mass (35.4% vs GM12878's 25.2%) than flat
decile-edge fitting can handle -- it collapses decile 1 to a zero-width
bin.

This script re-does the K562 match with the zero mass carved out into
its own explicit matching stratum BEFORE decile-matching the nonzero
remainder, then reruns the identical KS confound check structure so the
new p-value is directly comparable to the old one.

Code reuse (no logic re-implemented; all imported via
importlib.util.spec_from_file_location per project convention):
  - sgev.load_and_validate, sgev.REQUIRED_GATE_COLS, sgev.build_switching_sets,
    sgev.compute_noise_residual        <- code/switching_gene_expression_variability_v1.py
  - matcher.assign_deciles, matcher.match_by_decile
                                        <- code/matched_nonswitching_v1.py
  - rebinner.stratify_zero_mass        <- build_k562_stratified_rebinning_v1.py

NEW logic introduced here (covered by this script's own self-test, since
it does not exist in any imported module):
  - match_k562_zero_stratified(): combines a directly-matched zero
    stratum (non-switching zero-mass genes sampled 1:1 against switching
    zero-mass genes) with matcher.match_by_decile()'s existing proportional
    matching applied ONLY to the nonzero remainder.
  - retain_switching_per_stratum(): the per-stratum retention cap from
    rerun_matched_comparison_v1.py, extended to treat "ZERO" as an
    additional stratum bucket alongside deciles 0-9, so a shortfall in
    the zero stratum's matched pool is capped exactly like a shortfall
    in any decile (never silently kept at full switching-set size).

CONFIRMED SCHEMA (unchanged from Day 56/57/61 prior scripts):
  data/gate_assignments_named.tsv, data/k562_gate_assignments_named.tsv
  data/encode_rnaseq_gm12878_replicates_v1.csv, data/encode_rnaseq_k562_replicates_v1.csv
      (gene_id, rep1, rep2 -- mean_tpm = mean(rep1, rep2))

Outputs
-------
  data/matched_nonswitching_k562_zero_stratified_v1.tsv
      (gene_id, mean_tpm, stratum) -- stratum is "ZERO" or decile int 0-9
  data/k562_confound_check_v2_results.json
      same fields as rerun_matched_comparison_v1.py's per-cell-line
      result dict, K562 only, plus the Day 56 comparison numbers inline
      for direct before/after reading.

Usage
-----
    python rerun_k562_confound_check_v2.py --self-test
    python rerun_k562_confound_check_v2.py
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ---- all numeric / data imports go below this line, never above ----
import sys
import json
import argparse
import importlib.util
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ks_2samp

GATE_SCRIPT_PATH = "code/switching_gene_expression_variability_v1.py"
MATCHER_SCRIPT_PATH = "code/matched_nonswitching_v1.py"
REBINNER_SCRIPT_PATH = "build_k562_stratified_rebinning_v1.py"

K562_GATE_PATH = "data/k562_gate_assignments_named.tsv"
GM12878_GATE_PATH = "data/gate_assignments_named.tsv"
K562_EXPR_PATH = "data/encode_rnaseq_k562_replicates_v1.csv"

ZERO_THRESHOLD = 0.1  # locked in Block 2 (Day 61) -- do not redefine here

# New seed for this new matching procedure. Day 56's RNG_SEED=56 governed
# a materially different (flat-decile) matching process; reusing it here
# would not make results comparable, it would just be a leftover constant.
RNG_SEED = 61

OUT_MATCHED_PATH = "data/matched_nonswitching_k562_zero_stratified_v1.tsv"
OUT_RESULTS_PATH = "data/k562_confound_check_v2_results.json"

# Day 56 result being superseded, hardcoded from data/matched_comparison_results_v1.json
# for direct before/after printing. If that file's K562 numbers ever change,
# update this constant -- it is intentionally not re-read from disk so this
# script's self-test does not depend on Day 56's output existing.
DAY56_K562_CONFOUND_KS_STAT = None  # filled from JSON at runtime in run_real_data()
DAY56_K562_CONFOUND_KS_P = 0.0004043


def _load_gate_module():
    if not os.path.exists(GATE_SCRIPT_PATH):
        sys.exit(f"FATAL: cannot find {GATE_SCRIPT_PATH}.")
    spec = importlib.util.spec_from_file_location("sgev", GATE_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_matcher_module():
    if not os.path.exists(MATCHER_SCRIPT_PATH):
        sys.exit(f"FATAL: cannot find {MATCHER_SCRIPT_PATH}.")
    spec = importlib.util.spec_from_file_location("matcher", MATCHER_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_rebinner_module():
    if not os.path.exists(REBINNER_SCRIPT_PATH):
        sys.exit(f"FATAL: cannot find {REBINNER_SCRIPT_PATH}.")
    spec = importlib.util.spec_from_file_location("rebinner", REBINNER_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def match_k562_zero_stratified(matcher, rebinner, switching_tpm: pd.DataFrame,
                                nonswitching_tpm: pd.DataFrame,
                                threshold: float, rng: np.random.Generator):
    """
    Splits switching and non-switching mean_tpm into zero-stratum and
    nonzero-remainder (via rebinner.stratify_zero_mass, reused not
    reimplemented), matches the zero stratum 1:1-target directly, and
    matches the nonzero remainder via matcher.match_by_decile() exactly
    as Day 56 did -- just scoped to the nonzero remainder only.

    Returns (matched_df, edges) where matched_df has columns
    [gene_id, mean_tpm, stratum] and stratum is "ZERO" or an int decile
    label (0-9).
    """
    sw_zero_vals, sw_nonzero_vals = rebinner.stratify_zero_mass(
        switching_tpm["mean_tpm"], threshold
    )
    ns_zero_vals, ns_nonzero_vals = rebinner.stratify_zero_mass(
        nonswitching_tpm["mean_tpm"], threshold
    )

    sw_zero_df = switching_tpm.loc[sw_zero_vals.index]
    ns_zero_df = nonswitching_tpm.loc[ns_zero_vals.index]
    sw_nonzero_df = switching_tpm.loc[sw_nonzero_vals.index]
    ns_nonzero_df = nonswitching_tpm.loc[ns_nonzero_vals.index]

    # --- Zero stratum: direct 1:1-target sample, same shortfall discipline
    #     as matched_nonswitching_v1.py's decile shortfall warning ---
    n_zero_target = len(sw_zero_df)
    if len(ns_zero_df) < n_zero_target:
        print(
            f"  [WARN] zero stratum: requested {n_zero_target}, only "
            f"{len(ns_zero_df)} available in non-switching pool. Taking all "
            f"available -- matched zero stratum will fall short by "
            f"{n_zero_target - len(ns_zero_df)}."
        )
        zero_matched = ns_zero_df.copy()
    else:
        idx = rng.choice(ns_zero_df.index.to_numpy(), size=n_zero_target, replace=False)
        zero_matched = ns_zero_df.loc[idx].copy()
    zero_matched["stratum"] = "ZERO"

    # --- Nonzero remainder: reuse matcher.match_by_decile() unmodified ---
    nonzero_matched, edges = matcher.match_by_decile(sw_nonzero_df, ns_nonzero_df, rng)
    nonzero_deciles, _ = matcher.assign_deciles(nonzero_matched["mean_tpm"], edges=edges)
    nonzero_matched = nonzero_matched.copy()
    nonzero_matched["stratum"] = nonzero_deciles

    combined = pd.concat([zero_matched, nonzero_matched], ignore_index=True)
    return combined, edges


def retain_switching_per_stratum(switching_tpm: pd.DataFrame, matched: pd.DataFrame,
                                  threshold: float, edges: np.ndarray,
                                  rng: np.random.Generator):
    """
    Assigns each switching gene a stratum label ("ZERO" or decile int,
    using the SAME edges the matcher used on the nonzero remainder), then
    caps the retained switching genes per stratum at whatever the matched
    non-switching set actually achieved in that stratum -- identical
    discipline to rerun_matched_comparison_v1.py's per-decile pairing cap,
    extended to include "ZERO" as a stratum.
    """
    switching_tpm = switching_tpm.copy()
    is_zero = switching_tpm["mean_tpm"] <= threshold
    log_tpm = np.log1p(switching_tpm.loc[~is_zero, "mean_tpm"].to_numpy())
    nonzero_deciles = np.digitize(log_tpm, edges[1:-1], right=False)

    strata = pd.Series(index=switching_tpm.index, dtype=object)
    strata.loc[is_zero] = "ZERO"
    strata.loc[~is_zero] = nonzero_deciles
    switching_tpm["stratum"] = strata

    achieved_counts = matched["stratum"].value_counts().to_dict()
    retained_frames = []
    for stratum, group in switching_tpm.groupby("stratum"):
        cap = achieved_counts.get(stratum, 0)
        if cap == 0:
            continue
        if len(group) <= cap:
            retained_frames.append(group)
        else:
            idx = rng.choice(group.index.to_numpy(), size=cap, replace=False)
            retained_frames.append(group.loc[idx])

    retained = pd.concat(retained_frames, ignore_index=True) if retained_frames \
        else switching_tpm.iloc[0:0].copy()
    return retained


def synthetic_self_test():
    """
    Hand-constructed fixture validating the NEW logic only (zero-stratum
    matching + combined per-stratum retention cap). match_by_decile() and
    stratify_zero_mass() are exercised here but their correctness is
    already covered by their own scripts' self-tests -- this test checks
    that COMBINING them behaves as hand-derivable.

    Fixture: 6 switching genes (2 zero, 4 nonzero spread wide), 20
    non-switching genes (3 zero, 17 nonzero spread wide -- deliberately
    NOT starved, so no shortfall path is exercised here).

    Expected: zero stratum matched count = min(2, 3) = 2 (no shortfall).
    Expected: retained switching count == len(matched) exactly, since
    no stratum is starved on either side in this fixture.
    """
    rng = np.random.default_rng(RNG_SEED)
    matcher = _load_matcher_module()
    rebinner = _load_rebinner_module()

    switching_tpm = pd.DataFrame({
        "gene_id": [f"SYN_SW_{i}" for i in range(6)],
        "mean_tpm": [0.0, 0.0, 1.0, 5.0, 20.0, 45.0],
    })
    nonswitching_tpm = pd.DataFrame({
        "gene_id": [f"SYN_NS_{i}" for i in range(20)],
        "mean_tpm": [0.0, 0.0, 0.0] + list(np.linspace(0.5, 50.0, 17)),
    })

    matched, edges = match_k562_zero_stratified(
        matcher, rebinner, switching_tpm, nonswitching_tpm, ZERO_THRESHOLD, rng
    )

    zero_matched_n = (matched["stratum"] == "ZERO").sum()
    if zero_matched_n != 2:
        print(f"SELF-TEST FAILED: expected 2 zero-stratum matches, got {zero_matched_n}")
        sys.exit(1)

    retained = retain_switching_per_stratum(switching_tpm, matched, ZERO_THRESHOLD, edges, rng)

    if len(retained) != len(matched):
        print(
            f"SELF-TEST FAILED: expected retained switching count ({len(retained)}) "
            f"to equal matched non-switching count ({len(matched)}) in an "
            f"unstarved fixture, but they differ."
        )
        sys.exit(1)

    retained_zero_n = (retained["stratum"] == "ZERO").sum()
    if retained_zero_n != 2:
        print(f"SELF-TEST FAILED: expected 2 retained switching genes in ZERO stratum, got {retained_zero_n}")
        sys.exit(1)

    print("SELF-TEST PASSED: zero-stratum matching and combined per-stratum retention cap "
          "behave as hand-derived on the toy fixture.")


def run_real_data():
    sgev = _load_gate_module()
    matcher = _load_matcher_module()
    rebinner = _load_rebinner_module()

    gm_gate_df = sgev.load_and_validate(GM12878_GATE_PATH, sgev.REQUIRED_GATE_COLS, sep="\t")
    k562_gate_df = sgev.load_and_validate(K562_GATE_PATH, sgev.REQUIRED_GATE_COLS, sep="\t")
    switching, stable, _ = sgev.build_switching_sets(gm_gate_df, k562_gate_df)

    expr = pd.read_csv(K562_EXPR_PATH)
    expr["mean_tpm"] = expr[["rep1", "rep2"]].mean(axis=1)

    switching_tpm = expr[["gene_id", "mean_tpm"]].merge(switching[["gene_id"]], on="gene_id", how="inner")
    nonswitching_tpm = expr[["gene_id", "mean_tpm"]].merge(stable[["gene_id"]], on="gene_id", how="inner")

    print(f"K562 switching genes with expression data: {len(switching_tpm)} (of {len(switching)})")
    print(f"K562 non-switching pool with expression data: {len(nonswitching_tpm)} (of {len(stable)})")

    rng = np.random.default_rng(RNG_SEED)
    matched, edges = match_k562_zero_stratified(
        matcher, rebinner, switching_tpm, nonswitching_tpm, ZERO_THRESHOLD, rng
    )
    matched.to_csv(OUT_MATCHED_PATH, sep="\t", index=False)
    print(f"Wrote {OUT_MATCHED_PATH} ({len(matched)} matched non-switching genes)")

    retained = retain_switching_per_stratum(switching_tpm, matched, ZERO_THRESHOLD, edges, rng)
    dropped_n = len(switching_tpm) - len(retained)
    print(f"Switching genes: {len(switching_tpm)} total, {len(retained)} retained after "
          f"per-stratum pairing cap ({dropped_n} dropped)")

    rep_cols = ["rep1", "rep2"]
    noise_df = sgev.compute_noise_residual(expr, "gene_id", rep_cols)

    sw_ids = set(retained["gene_id"])
    ns_ids = set(matched["gene_id"])
    sw_scores = noise_df[noise_df["gene_id"].isin(sw_ids)]["noise_residual"].dropna()
    ns_scores = noise_df[noise_df["gene_id"].isin(ns_ids)]["noise_residual"].dropna()

    if len(sw_scores) < 5 or len(ns_scores) < 5:
        sys.exit(
            f"FATAL: too few genes with valid noise_residual (switching n={len(sw_scores)}, "
            f"non-switching n={len(ns_scores)}). Not proceeding with an under-powered comparison."
        )

    u_stat, p_val = mannwhitneyu(sw_scores, ns_scores, alternative="two-sided")

    sw_mean_expr = noise_df[noise_df["gene_id"].isin(sw_ids)]["mean_expr"].dropna()
    ns_mean_expr = noise_df[noise_df["gene_id"].isin(ns_ids)]["mean_expr"].dropna()
    ks_stat, ks_p = ks_2samp(sw_mean_expr, ns_mean_expr)

    print(f"\nMann-Whitney: switching mean_residual={sw_scores.mean():.4f}, "
          f"non-switching mean_residual={ns_scores.mean():.4f}, U={u_stat:.1f}, p={p_val:.4g}")
    print(f"Hansen confound check (zero-stratified matched groups): KS={ks_stat:.4f}, p={ks_p:.4g} "
          f"({'STILL CONFOUNDED' if ks_p < 0.05 else 'confound resolved'})")
    print(f"\nBEFORE/AFTER (Day 56 flat-decile match vs Day 61 zero-stratified match), K562:")
    print(f"  Day 56 (flat decile):        KS p = {DAY56_K562_CONFOUND_KS_P:.4g}  (STILL CONFOUNDED)")
    print(f"  Day 61 (zero-stratified):    KS p = {ks_p:.4g}  "
          f"({'STILL CONFOUNDED' if ks_p < 0.05 else 'confound resolved'})")

    results = {
        "k562_zero_stratified": {
            "n_switching_total": int(len(switching_tpm)),
            "n_switching_retained": int(len(retained)),
            "n_switching_dropped": int(dropped_n),
            "n_matched_nonswitching": int(len(matched)),
            "sw_n_valid_residual": int(len(sw_scores)),
            "ns_n_valid_residual": int(len(ns_scores)),
            "sw_mean_residual": float(sw_scores.mean()),
            "ns_mean_residual": float(ns_scores.mean()),
            "mw_u": float(u_stat),
            "mw_p": float(p_val),
            "confound_ks_stat": float(ks_stat),
            "confound_ks_p": float(ks_p),
        },
        "day56_flat_decile_comparison": {
            "confound_ks_p": DAY56_K562_CONFOUND_KS_P,
            "verdict": "STILL CONFOUNDED",
        },
    }

    with open(OUT_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_RESULTS_PATH}")


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