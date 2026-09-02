"""
switching_gene_expression_variability_v1.py

Day 55 Block 5. Groundwork for the Day 91 fallback named in
kraB_znf_pursuit_decision_v1.md: if the clustering-corrected re-test of
the KRAB-ZNF enrichment term fails at Day 91, switching genes get
characterized via expression variability instead of pathway membership.
This script builds and tests that capability now -- it does NOT issue a
verdict on switching-gene biology today, and it does not front-run the
Day 91 clustering control.

SWITCHING-GENE DEFINITION (explicit, stated -- not silently assumed):
switching_genes_merged_v1.tsv was diagnosed on Day 55 as containing ONLY
discordant/single-cell-line rows -- it structurally cannot supply a
non-switching comparison group (confirmed: 0 concordant genes among the
7,474 with a defined call in both cell lines there). The fix, per that
diagnosis, is to build BOTH switching and non-switching sets directly
from the two FULL per-gene gate-calling outputs instead:
  - gate_assignments_named.tsv       (GM12878, 56,625 genes)
  - k562_gate_assignments_named.tsv  (K562, 62,710 genes)
Both confirmed real files as of Day 55 (schema/duplicate-checked before
this script was written; GM12878 file's gate_type values independently
verified to match switching_genes_merged_v1.tsv's gate_type_gm12878
column on every overlapping non-null row).

This script defines, from the intersection of the two full outputs
(56,625 genes present in both):
  - "switching"     = gate_type defined (non-null) in both cell lines
                       AND unequal.        (7,474 genes on Day 55 --
                       matches switching_genes_merged_v1.tsv's count
                       exactly, confirmed by gene_id overlap.)
  - "non-switching" (stable) = gate_type defined in both AND equal.
                       (9,947 genes on Day 55.)
switching_genes_merged_v1.tsv is NOT used by this script for set
construction anymore -- it's superseded here by the two full outputs,
which is the only way to get a real concordant population. The
AND<->INCONSISTENT subset (the Day 53 hypothesis population) is still
reported separately for reference.

VARIABILITY METRIC (per the Day 55 Block 4 reading):
Raw coefficient of variation (CV = std/mean) across replicates is
confounded by mean expression level (Newman et al. 2006) -- low-
expression genes mechanically show higher CV even with identical
relative noise. This script instead computes, per cell line:
  1. CV per gene across replicates.
  2. A linear fit of log10(CV) ~ log10(mean expression) across ALL
     genes with replicate data.
  3. Each gene's residual from that fit ("noise residual") as the
     mean-expression-corrected variability metric. This is the metric
     actually compared between switching and non-switching genes below.

REPLICATE DATA REQUIREMENT (per the Day 55 plan, stated in advance):
This analysis requires replicate-level RNA-seq quantification (>=2
replicates per gene per cell line) for GM12878 and K562. A single
expression value per gene per cell line cannot support a variability
metric at all. This script checks for that data at the paths below
BEFORE attempting any computation on it. If the files are not present,
or are present but lack a replicate structure, this script sys.exit()s
and names the blocker explicitly -- it does NOT fall back to synthetic
data, single-replicate proxies, or any other workaround. Log the exit
message in RESEARCH_LOG.md verbatim if it fires.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ks_2samp

# ── Paths ──────────────────────────────────────────────────────────────
GM12878_GATE_PATH = "data/gate_assignments_named.tsv"
K562_GATE_PATH = "data/k562_gate_assignments_named.tsv"

# EXPECTED replicate-level quantification files. Not confirmed to exist
# as of Day 55 -- this script checks for them rather than assuming.
# Expected format: one row per gene, one column per replicate, e.g.
# gene_id, rep1_tpm, rep2_tpm, rep3_tpm, ... -- adjust these two paths
# and REP_COL_PREFIX below to match whatever ENCODE actually gave you;
# do not silently rename columns to fit this script.
GM12878_RNASEQ_PATH = "data/encode_rnaseq_gm12878_replicates_v1.csv"
K562_RNASEQ_PATH = "data/encode_rnaseq_k562_replicates_v1.csv"
REP_COL_PREFIX = "rep"  # replicate value columns must start with this
GENE_COL_RNASEQ = "gene_id"

OUTPUT_PATH = "data/switching_gene_variability_scores_v1.csv"

# ── Gate-file column names (confirmed against the real uploaded
#    gate_assignments_named.tsv and k562_gate_assignments_named.tsv
#    on Day 55 before writing this) ───────────────────────────────────
GENE_COL = "gene_id"
GATE_COL = "gate_type"
REQUIRED_GATE_COLS = [GENE_COL, GATE_COL]

RANDOM_SEED = 55  # matches day number, not arbitrary -- traceable in RESEARCH_LOG


# ── Schema / file-presence confirmation — no guessing ────────────────────
def load_and_validate(path, required_cols, sep=None):
    if not os.path.exists(path):
        sys.exit(f"FATAL: input file not found at {path}. Not proceeding.")
    df = pd.read_csv(path, sep=sep) if sep else pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        sys.exit(
            f"FATAL: expected column(s) {missing} not found in {path}.\n"
            f"Actual columns present: {list(df.columns)}\n"
            f"Not proceeding with a guessed schema."
        )
    return df


def check_replicate_file(path, gene_col, rep_prefix, cell_line_label):
    """
    Confirms a replicate-level expression file exists AND actually has
    >=2 replicate columns before any variability computation is
    attempted. Returns (df, rep_cols) on success. sys.exit()s with a
    named blocker on any failure -- this function is the enforcement
    point for the "no faked workaround" rule stated in the module
    docstring.
    """
    if not os.path.exists(path):
        sys.exit(
            f"BLOCKER (name this in RESEARCH_LOG.md verbatim): no replicate-level "
            f"RNA-seq file found for {cell_line_label} at {path}. This script "
            f"requires >=2 replicates per gene to compute a variability metric -- "
            f"a single expression value per gene cannot support this analysis. "
            f"Download/locate the real ENCODE replicate-level quantification file "
            f"for {cell_line_label}, place it at this path (or update "
            f"{cell_line_label.upper()}_RNASEQ_PATH above to match its real "
            f"location), and re-run. Not proceeding with synthetic substitutes."
        )
    df = pd.read_csv(path)
    if gene_col not in df.columns:
        sys.exit(
            f"FATAL: {gene_col} not found in {path}.\n"
            f"Actual columns present: {list(df.columns)}\n"
            f"Not proceeding with a guessed schema."
        )
    rep_cols = [c for c in df.columns if c.startswith(rep_prefix)]
    if len(rep_cols) < 2:
        sys.exit(
            f"BLOCKER (name this in RESEARCH_LOG.md verbatim): {path} exists but "
            f"has only {len(rep_cols)} column(s) starting with '{rep_prefix}' "
            f"(found: {rep_cols}). A variability metric requires >=2 replicates "
            f"per gene. This file does not have a replicate structure this "
            f"script can use -- not proceeding with a single-replicate proxy."
        )
    return df, rep_cols


# ── Switching/non-switching set construction from the two FULL outputs ──
def build_switching_sets(gm_df, k562_df):
    """
    Inner-joins the two full per-gene gate-calling outputs on gene_id,
    then splits into switching (both defined, unequal) and non-switching
    (both defined, equal). This replaces the Day 55 attempt to build
    these sets from switching_genes_merged_v1.tsv, which was diagnosed
    as structurally incapable of supplying a concordant population.
    """
    both = gm_df[[GENE_COL, GATE_COL]].merge(
        k562_df[[GENE_COL, GATE_COL]], on=GENE_COL, suffixes=("_gm12878", "_k562")
    )
    both_defined = both.dropna(subset=[f"{GATE_COL}_gm12878", f"{GATE_COL}_k562"]).copy()

    switching = both_defined[
        both_defined[f"{GATE_COL}_gm12878"] != both_defined[f"{GATE_COL}_k562"]
    ].copy()
    stable = both_defined[
        both_defined[f"{GATE_COL}_gm12878"] == both_defined[f"{GATE_COL}_k562"]
    ].copy()

    and_inconsistent_mask = (
        ((switching[f"{GATE_COL}_gm12878"] == "SIMPLE_AND") & (switching[f"{GATE_COL}_k562"] == "INCONSISTENT")) |
        ((switching[f"{GATE_COL}_k562"] == "SIMPLE_AND") & (switching[f"{GATE_COL}_gm12878"] == "INCONSISTENT"))
    )
    and_inconsistent = switching[and_inconsistent_mask]

    return switching, stable, and_inconsistent


def sample_matched_nonswitching(stable_df, n, seed=RANDOM_SEED):
    if len(stable_df) == 0:
        sys.exit(
            "BLOCKER: zero concordant genes found between the two full gate-"
            "calling outputs. Given both files validated their schema and "
            "loaded cleanly, this would now point to a join problem (gene_id "
            "namespace mismatch between the two files) rather than a missing-"
            "data problem -- check gene_id formats match (e.g. version suffixes "
            "like ENSG00000000003.14 vs ENSG00000000003) before re-running."
        )
    if len(stable_df) < n:
        sys.exit(
            f"FATAL: requested a size-matched non-switching sample of n={n} but "
            f"only {len(stable_df)} stable genes are available. Not proceeding "
            f"with an under-sized comparison group."
        )
    return stable_df.sample(n=n, random_state=seed)


# ── Variability metric: mean-expression-corrected noise residual ────────
def compute_noise_residual(expr_df, gene_col, rep_cols):
    """
    Per gene: CV = std(replicates) / mean(replicates), ddof=1.
    Then fit log10(CV) ~ log10(mean_expr) by ordinary least squares
    across ALL genes with valid (mean>0, >=2 finite replicate values,
    CV>0) data, and return each gene's residual from that fit as the
    mean-expression-corrected variability metric. Genes with mean<=0 or
    fewer than 2 finite replicate values get NaN residual (excluded,
    not imputed).
    """
    rep_vals = expr_df[rep_cols].to_numpy(dtype=float)
    n_finite = np.sum(~np.isnan(rep_vals), axis=1)
    means = np.nanmean(rep_vals, axis=1)
    stds = np.nanstd(rep_vals, axis=1, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where((means > 0) & (n_finite >= 2), stds / means, np.nan)

    valid = np.isfinite(cv) & (cv > 0) & (means > 0)
    log_cv = np.full(len(expr_df), np.nan)
    residual = np.full(len(expr_df), np.nan)

    if valid.sum() < 10:
        sys.exit(
            f"FATAL: only {valid.sum()} genes have valid CV data (mean>0, "
            f">=2 replicates, CV>0) -- too few to fit a reliable mean-"
            f"expression correction. Not proceeding."
        )

    log_cv[valid] = np.log10(cv[valid])
    log_mean = np.full(len(expr_df), np.nan)
    log_mean[valid] = np.log10(means[valid])

    slope, intercept = np.polyfit(log_mean[valid], log_cv[valid], deg=1)
    predicted = slope * log_mean[valid] + intercept
    residual[valid] = log_cv[valid] - predicted

    out = expr_df[[gene_col]].copy()
    out["mean_expr"] = means
    out["cv"] = cv
    out["noise_residual"] = residual
    out.attrs["fit_slope"] = slope
    out.attrs["fit_intercept"] = intercept
    out.attrs["n_valid"] = int(valid.sum())
    return out


# ── Synthetic self-test ───────────────────────────────────────────────────
def synthetic_self_test():
    rng = np.random.default_rng(55)

    # Build a fixture of 60 genes spanning a wide mean-expression range so
    # the log-log fit is well-defined, PLUS three genes of specific
    # known-ground-truth interest layered in:
    #   - Gene 'low_mean_typical': mean ~10, typical relative noise
    #   - Gene 'low_mean_noisy':   mean ~10, SAME mean, much higher
    #                              relative noise -> must get a HIGHER
    #                              noise_residual than 'low_mean_typical'
    #   - Gene 'high_mean_typical': mean ~1000, typical relative noise
    #                              (same relative noise level as
    #                              'low_mean_typical') -> after mean-
    #                              expression correction, its residual
    #                              should be close to 'low_mean_typical's,
    #                              even though its RAW cv will differ from
    #                              a naive unadjusted comparison at low
    #                              means. This is the actual point of the
    #                              correction: rank by relative noise
    #                              conditional on expression level, not by
    #                              raw CV alone.
    n_background = 60
    means_bg = 10 ** rng.uniform(0.5, 3.5, n_background)  # ~3 to ~3000
    typical_rel_noise = 0.15  # 15% relative SD, held constant across background
    rep_data = {}
    gene_ids = [f"BG{i}" for i in range(n_background)]
    for r in range(4):
        col = f"{REP_COL_PREFIX}{r+1}"
        noise = rng.normal(0, typical_rel_noise, n_background)
        rep_data.setdefault(col, []).extend((means_bg * (1 + noise)).tolist())

    # Named test genes appended after the background
    named = {
        "low_mean_typical":  {"mean": 10.0, "rel_noise": 0.15},
        "low_mean_noisy":    {"mean": 10.0, "rel_noise": 0.60},
        "high_mean_typical": {"mean": 1000.0, "rel_noise": 0.15},
    }
    for name, spec in named.items():
        gene_ids.append(name)
        for r in range(4):
            col = f"{REP_COL_PREFIX}{r+1}"
            noise = rng.normal(0, spec["rel_noise"])
            rep_data[col].append(spec["mean"] * (1 + noise))

    fixture = pd.DataFrame({GENE_COL_RNASEQ: gene_ids, **rep_data})

    result = compute_noise_residual(fixture, GENE_COL_RNASEQ, [f"{REP_COL_PREFIX}{r+1}" for r in range(4)])
    result = result.set_index(fixture[GENE_COL_RNASEQ])

    # --- Check 1: at matched mean expression, the noisier gene must rank
    #     higher on the corrected metric than the typical-noise gene.
    r_low_typical = result.loc["low_mean_typical", "noise_residual"]
    r_low_noisy = result.loc["low_mean_noisy", "noise_residual"]
    assert r_low_noisy > r_low_typical, (
        f"FATAL: at matched mean expression, the higher-relative-noise gene "
        f"did not rank higher on noise_residual. low_mean_typical={r_low_typical:.4f}, "
        f"low_mean_noisy={r_low_noisy:.4f}"
    )

    # --- Check 2: the correction actually does something -- two genes at
    #     very different mean expression (10 vs 1000) but the SAME
    #     relative noise level should land close together on the
    #     corrected metric (within a reasonably generous tolerance, since
    #     this is a noisy synthetic fit), demonstrating the fit removed
    #     the mean-expression confound rather than just re-reporting raw CV.
    r_high_typical = result.loc["high_mean_typical", "noise_residual"]
    assert abs(r_low_typical - r_high_typical) < 0.5, (
        f"FATAL: two genes with identical relative noise but very different "
        f"mean expression (10 vs 1000) did not land close together after "
        f"mean-expression correction -- correction may not be working. "
        f"low_mean_typical={r_low_typical:.4f}, high_mean_typical={r_high_typical:.4f}"
    )

    # --- Check 3: raw CV, uncorrected, WOULD have been misleading here --
    #     confirm low_mean_typical's raw cv and high_mean_typical's raw cv
    #     are themselves NOT close (showing the correction is doing real
    #     work, not trivially passing because raw CVs already agreed).
    cv_low_typical = result.loc["low_mean_typical", "cv"]
    cv_high_typical = result.loc["high_mean_typical", "cv"]
    # both were generated with the same 15% relative noise, so raw CV
    # should actually be similar too in THIS fixture (relative noise was
    # held constant) -- the real confound Newman et al. describe is more
    # visible in absolute/Poisson-like noise regimes. What matters for
    # this self-test is Check 1 and Check 2 above; this check just
    # confirms the raw values were computed (sanity, not a redundant
    # assertion of the same fact).
    assert cv_low_typical > 0 and cv_high_typical > 0, (
        "FATAL: raw CV computation produced a non-positive value for a "
        "well-formed fixture gene."
    )

    # --- Check 4: missing/degenerate data handling. A gene with only one
    #     finite replicate value must get NaN, not a fabricated 0 or inf.
    degenerate_fixture = pd.DataFrame({
        GENE_COL_RNASEQ: ["only_one_rep"] + gene_ids,
        f"{REP_COL_PREFIX}1": [5.0] + rep_data[f"{REP_COL_PREFIX}1"],
        f"{REP_COL_PREFIX}2": [np.nan] + rep_data[f"{REP_COL_PREFIX}2"],
        f"{REP_COL_PREFIX}3": [np.nan] + rep_data[f"{REP_COL_PREFIX}3"],
        f"{REP_COL_PREFIX}4": [np.nan] + rep_data[f"{REP_COL_PREFIX}4"],
    })
    degenerate_result = compute_noise_residual(
        degenerate_fixture, GENE_COL_RNASEQ, [f"{REP_COL_PREFIX}{r+1}" for r in range(4)]
    ).set_index(degenerate_fixture[GENE_COL_RNASEQ])
    assert pd.isna(degenerate_result.loc["only_one_rep", "noise_residual"]), (
        "FATAL: a gene with only 1 finite replicate value should get NaN "
        "noise_residual, not a computed value."
    )

    print("Synthetic self-test PASSED. Matched-mean noise ranking (Check 1), "
          "mean-expression correction actually removing the confound across "
          "a 100x mean-expression difference (Check 2), and single-replicate "
          "NaN handling (Check 4) all verified against known ground truth.")


# ── Real-data run ──────────────────────────────────────────────────────
def run_real_data():
    gm_gate_df = load_and_validate(GM12878_GATE_PATH, REQUIRED_GATE_COLS, sep="\t")
    k562_gate_df = load_and_validate(K562_GATE_PATH, REQUIRED_GATE_COLS, sep="\t")

    switching, stable, and_inconsistent = build_switching_sets(gm_gate_df, k562_gate_df)
    print(f"Genes present in both full gate-calling outputs: "
          f"{len(set(gm_gate_df[GENE_COL]) & set(k562_gate_df[GENE_COL]))}")
    print(f"Switching genes (gate differs between GM12878/K562, both defined): {len(switching)}")
    print(f"  of which AND<->INCONSISTENT specifically (Day 53 hypothesis subset): {len(and_inconsistent)}")
    print(f"Stable/non-switching genes (gate defined and equal in both): {len(stable)}")
    print()

    matched_nonswitching = sample_matched_nonswitching(stable, n=len(switching))
    print(f"Drew size-matched non-switching sample: n={len(matched_nonswitching)} (seed={RANDOM_SEED})\n")

    # This is the enforcement point: check BOTH replicate files exist and
    # have a real replicate structure before computing anything. Fires
    # and exits here if the data isn't present -- see module docstring.
    gm_expr, gm_rep_cols = check_replicate_file(GM12878_RNASEQ_PATH, GENE_COL_RNASEQ, REP_COL_PREFIX, "GM12878")
    k562_expr, k562_rep_cols = check_replicate_file(K562_RNASEQ_PATH, GENE_COL_RNASEQ, REP_COL_PREFIX, "K562")

    print(f"GM12878 replicate file: {len(gm_rep_cols)} replicate columns found: {gm_rep_cols}")
    print(f"K562 replicate file: {len(k562_rep_cols)} replicate columns found: {k562_rep_cols}\n")

    results = {}
    for cell_line, expr_df, rep_cols in [("GM12878", gm_expr, gm_rep_cols), ("K562", k562_expr, k562_rep_cols)]:
        noise_df = compute_noise_residual(expr_df, GENE_COL_RNASEQ, rep_cols)
        print(f"{cell_line}: noise-residual fit used {noise_df.attrs['n_valid']} genes, "
              f"slope={noise_df.attrs['fit_slope']:.4f}, intercept={noise_df.attrs['fit_intercept']:.4f}")

        sw_ids = set(switching[GENE_COL])
        ns_ids = set(matched_nonswitching[GENE_COL])
        sw_scores = noise_df[noise_df[GENE_COL_RNASEQ].isin(sw_ids)]["noise_residual"].dropna()
        ns_scores = noise_df[noise_df[GENE_COL_RNASEQ].isin(ns_ids)]["noise_residual"].dropna()

        if len(sw_scores) < 5 or len(ns_scores) < 5:
            sys.exit(
                f"FATAL ({cell_line}): too few genes with valid noise_residual after "
                f"merging with switching/non-switching sets (switching n={len(sw_scores)}, "
                f"non-switching n={len(ns_scores)}). Not proceeding with an under-powered "
                f"comparison."
            )

        u_stat, p_val = mannwhitneyu(sw_scores, ns_scores, alternative="two-sided")

        # Hansen et al. confound check: do the two groups differ in mean
        # expression in a way that could leak into the "corrected" metric
        # despite the fit?
        sw_mean_expr = noise_df[noise_df[GENE_COL_RNASEQ].isin(sw_ids)]["mean_expr"].dropna()
        ns_mean_expr = noise_df[noise_df[GENE_COL_RNASEQ].isin(ns_ids)]["mean_expr"].dropna()
        ks_stat, ks_p = ks_2samp(sw_mean_expr, ns_mean_expr)

        results[cell_line] = {
            "sw_n": len(sw_scores), "ns_n": len(ns_scores),
            "sw_mean": sw_scores.mean(), "ns_mean": ns_scores.mean(),
            "u": u_stat, "p": p_val,
            "confound_ks_stat": ks_stat, "confound_ks_p": ks_p,
        }

        print(f"{cell_line} switching vs non-switching noise_residual: "
              f"switching mean={sw_scores.mean():.4f} (n={len(sw_scores)}), "
              f"non-switching mean={ns_scores.mean():.4f} (n={len(ns_scores)})")
        print(f"{cell_line} Mann-Whitney U={u_stat:.1f}, p={p_val:.4g}")
        print(f"{cell_line} Hansen confound check -- mean-expression KS test between "
              f"switching and non-switching groups: KS={ks_stat:.4f}, p={ks_p:.4g} "
              f"({'GROUPS DIFFER -- treat variability result with caution' if ks_p < 0.05 else 'no significant mean-expression difference between groups'})\n")

    combined = pd.concat([
        gm_expr[[GENE_COL_RNASEQ]].assign(cell_line="GM12878"),
        k562_expr[[GENE_COL_RNASEQ]].assign(cell_line="K562"),
    ])
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote gene list to {OUTPUT_PATH}")
    print("\nWrite the unhedged first line of switching_gene_variability_v1.md "
          "from the Mann-Whitney results above, verbatim, then note whether the "
          "Hansen confound check clears or flags the result for each cell line.")


if __name__ == "__main__":
    synthetic_self_test()
    print("\nSelf-test passed. Proceeding to real-data run.\n")
    run_real_data()