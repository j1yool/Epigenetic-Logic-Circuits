"""
rerun_matched_comparison_v1.py

Day 56 Block 2. Consumes matched_nonswitching_v1.py's output. Per-cell-line
switching set is DROPPED to only the genes that actually landed a matched
partner (deciles 3-5 were starved in the non-switching pool; see
matched_nonswitching_gm12878.tsv / matched_nonswitching_k562.tsv run log).
This keeps the comparison a true matched-pairs design at the cost of
switching n (7,474 -> 6,162 GM12878, 7,474 -> 6,493 K562). Oversampling
the shorted deciles with replacement was rejected: it would fabricate
non-switching data points and reintroduce the exact confound this
exercise exists to remove.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import json
import importlib.util
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ks_2samp

GATE_SCRIPT_PATH = "code/switching_gene_expression_variability_v1.py"
RNG_SEED = 56
MATCHED_PATHS = {
    "gm12878": "data/matched_nonswitching_gm12878.tsv",
    "k562": "data/matched_nonswitching_k562.tsv",
}
EXPR_PATHS = {
    "gm12878": "data/encode_rnaseq_gm12878_replicates_v1.csv",
    "k562": "data/encode_rnaseq_k562_replicates_v1.csv",
}
GM12878_GATE_PATH = "data/gate_assignments_named.tsv"
K562_GATE_PATH = "data/k562_gate_assignments_named.tsv"
OUT_PATH = "data/matched_comparison_results_v1.json"


def _load_gate_module():
    spec = importlib.util.spec_from_file_location("sgev", GATE_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    sgev = _load_gate_module()
    gm_gate_df = sgev.load_and_validate(GM12878_GATE_PATH, sgev.REQUIRED_GATE_COLS, sep="\t")
    k562_gate_df = sgev.load_and_validate(K562_GATE_PATH, sgev.REQUIRED_GATE_COLS, sep="\t")
    switching, stable, _ = sgev.build_switching_sets(gm_gate_df, k562_gate_df)

    results = {}
    for cell_line in ("gm12878", "k562"):
        print(f"\n=== {cell_line.upper()} ===")

        matched = pd.read_csv(MATCHED_PATHS[cell_line], sep="\t")
        expr = pd.read_csv(EXPR_PATHS[cell_line])
        expr["mean_tpm"] = expr[["rep1", "rep2"]].mean(axis=1)

        switching_tpm = expr[["gene_id", "mean_tpm"]].merge(
            switching[["gene_id"]], on="gene_id", how="inner"
        )
        sw_deciles, edges = _assign_deciles_against_pool(switching_tpm, matched, stable, expr)
        switching_tpm = switching_tpm.copy()
        switching_tpm["decile"] = sw_deciles

        # True per-decile pairing: cap switching genes retained in each
        # decile at the number of non-switching genes ACTUALLY matched in
        # that decile (not just "decile present in matched at all"). A
        # decile with a shortfall (e.g. matched=733 vs switching=1006 in
        # GM12878 decile 3) means only 733 switching genes from that
        # decile can be kept as genuinely paired -- the rest get dropped,
        # subsampled deterministically (seed=56) so the retained switching
        # subset's decile PROPORTIONS actually match the matched
        # non-switching set's proportions. Keeping all 7,474 switching
        # genes while the matched pool fell short in 3 deciles (the first
        # version of this script's bug) silently re-skews the comparison
        # and is exactly why the confound was still firing below.
        achieved_counts = matched["decile"].value_counts().to_dict()
        rng = np.random.default_rng(RNG_SEED)
        retained_frames = []
        for decile, group in switching_tpm.groupby("decile"):
            cap = achieved_counts.get(decile, 0)
            if cap == 0:
                continue
            if len(group) <= cap:
                retained_frames.append(group)
            else:
                idx = rng.choice(group.index.to_numpy(), size=cap, replace=False)
                retained_frames.append(group.loc[idx])
        switching_matched = pd.concat(retained_frames, ignore_index=True) if retained_frames \
            else switching_tpm.iloc[0:0].copy()

        dropped_n = len(switching_tpm) - len(switching_matched)
        print(f"  switching genes: {len(switching_tpm)} total, {len(switching_matched)} "
              f"retained after per-decile pairing cap ({dropped_n} dropped)")
        print(f"  matched non-switching genes: {len(matched)}")

        # Noise residual via the Day 55 script's own function, applied to
        # the full expression table (fit needs the full background, not
        # just the matched subset) then subset to our two groups.
        rep_cols = ["rep1", "rep2"]
        noise_df = sgev.compute_noise_residual(expr, "gene_id", rep_cols)

        sw_ids = set(switching_matched["gene_id"])
        ns_ids = set(matched["gene_id"])
        sw_scores = noise_df[noise_df["gene_id"].isin(sw_ids)]["noise_residual"].dropna()
        ns_scores = noise_df[noise_df["gene_id"].isin(ns_ids)]["noise_residual"].dropna()

        if len(sw_scores) < 5 or len(ns_scores) < 5:
            sys.exit(
                f"FATAL ({cell_line}): too few genes with valid noise_residual "
                f"(switching n={len(sw_scores)}, non-switching n={len(ns_scores)}). "
                f"Not proceeding with an under-powered comparison."
            )

        u_stat, p_val = mannwhitneyu(sw_scores, ns_scores, alternative="two-sided")

        sw_mean_expr = noise_df[noise_df["gene_id"].isin(sw_ids)]["mean_expr"].dropna()
        ns_mean_expr = noise_df[noise_df["gene_id"].isin(ns_ids)]["mean_expr"].dropna()
        ks_stat, ks_p = ks_2samp(sw_mean_expr, ns_mean_expr)

        results[cell_line] = {
            "n_switching_total": int(len(switching_tpm)),
            "n_switching_retained": int(len(switching_matched)),
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
        }

        print(f"  Mann-Whitney: switching mean_residual={sw_scores.mean():.4f}, "
              f"non-switching mean_residual={ns_scores.mean():.4f}, "
              f"U={u_stat:.1f}, p={p_val:.4g}")
        print(f"  Hansen confound check (matched groups): KS={ks_stat:.4f}, p={ks_p:.4g} "
              f"({'STILL CONFOUNDED' if ks_p < 0.05 else 'confound resolved'})")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


def _assign_deciles_against_pool(switching_tpm, matched, stable, expr):
    """
    Recompute the same pooled-edge decile assignment matched_nonswitching_v1.py
    used, so switching genes here are binned identically to how the matcher
    binned them. Pool = full switching set (pre-drop) + full non-switching
    (stable) set with expression data, matching matched_nonswitching_v1.py's
    pooling exactly.
    """
    nonswitching_tpm = expr[["gene_id", "mean_tpm"]].merge(
        stable[["gene_id"]], on="gene_id", how="inner"
    )
    pooled = pd.concat(
        [switching_tpm["mean_tpm"], nonswitching_tpm["mean_tpm"]], ignore_index=True
    )
    log_pooled = np.log1p(pooled.to_numpy())
    edges = np.percentile(log_pooled, np.linspace(0, 100, 11))
    edges[0] = -np.inf
    edges[-1] = np.inf
    log_sw = np.log1p(switching_tpm["mean_tpm"].to_numpy())
    deciles = np.digitize(log_sw, edges[1:-1], right=False)
    return deciles, edges


if __name__ == "__main__":
    main()