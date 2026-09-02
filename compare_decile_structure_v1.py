"""
compare_decile_structure_v1.py

Day 57 | Epigenetic Logic Circuits

Purpose
-------
Day 56's matched_nonswitching_v1.py cleared the Hansen confound for
GM12878 (KS p=0.3917) but not K562 (KS p=0.0004043). This script runs
the ONE named diagnostic from the Day 56->57 forward note: compare
GM12878's and K562's pooled decile edges and within-decile TPM pool
sizes directly, to establish whether K562's non-switching pool is
thin exactly where K562 switching genes concentrate -- a matching-power
artifact -- versus something structurally different between the two
cell lines' expression distributions.

This is NOT a re-implementation of matched_nonswitching_v1.py's binning
logic. It imports assign_deciles(), match_by_decile(), and
_load_gate_module() directly from that script via importlib, per
project convention (cross-script imports never via sys.path
manipulation). Recomputing the edges independently here would risk a
silent mismatch with the edges actually used in the Day 56 matched run
-- the whole point of this diagnostic is to explain THAT run, not a
new one.

CONFIRMED SCHEMA (checked against real files before writing this
script, Day 57):
  data/gate_assignments_named.tsv
      columns: gene_id, gate_type, complexity_score, gene_name
      56,626 rows (GM12878)
  data/k562_gate_assignments_named.tsv
      columns: gene_id, gate_type, complexity_score
      62,711 rows (K562)
  Neither file contains mean_tpm or any expression column. TPM is only
  available after merging gate calls against the expression replicate
  tables (encode_rnaseq_{cell_line}_replicates_v1.csv), exactly as
  matched_nonswitching_v1.py's run_real_data() does it. This script
  reuses that same merge path rather than assuming TPM lives in the
  gate files.

Inputs (real)
-------------
  code/matched_nonswitching_v1.py                (imported, not copied)
  code/switching_gene_expression_variability_v1.py  (imported transitively)
  data/gate_assignments_named.tsv
  data/k562_gate_assignments_named.tsv
  data/encode_rnaseq_gm12878_replicates_v1.csv
  data/encode_rnaseq_k562_replicates_v1.csv

Outputs
-------
  data/decile_edge_comparison_v1.tsv
      one row per decile boundary (0-10): GM12878 log-TPM edge, K562
      log-TPM edge, difference. Each cell line's edges are computed on
      ITS OWN pooled (switching + non-switching) distribution, matching
      how match_by_decile() computed them in the real Day 56 run --
      the two cell lines were never binned on a shared edge set, so
      this output makes that fact explicit rather than implying a
      shared scale.
  data/switch_decile_concentration_v1.tsv
      one row per decile (0-9): switching-gene count, non-switching
      pool count, and pool ratio (switching / pool), for each cell
      line side by side.

Usage
-----
    python compare_decile_structure_v1.py --self-test   # synthetic fixture only
    python compare_decile_structure_v1.py                 # self-test, then real-data run
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

GM12878_GATE_PATH = "data/gate_assignments_named.tsv"
K562_GATE_PATH = "data/k562_gate_assignments_named.tsv"
GM12878_EXPR_PATH = "data/encode_rnaseq_gm12878_replicates_v1.csv"
K562_EXPR_PATH = "data/encode_rnaseq_k562_replicates_v1.csv"

EXPECTED_EXPR_COLUMNS = {"gene_id", "rep1", "rep2"}
EXPECTED_GATE_COLUMNS_MIN = {"gene_id", "gate_type", "complexity_score"}

N_DECILES = 10

OUT_EDGES_PATH = "data/decile_edge_comparison_v1.tsv"
OUT_CONCENTRATION_PATH = "data/switch_decile_concentration_v1.tsv"


def _load_matcher_module():
    """
    Cross-script import via importlib.util.spec_from_file_location, per
    project convention. Loading this module does NOT trigger its
    __main__ block (guarded by if __name__ == "__main__"). This gives
    us assign_deciles(), match_by_decile(), and _load_gate_module()
    without duplicating any binning logic.
    """
    if not os.path.exists(MATCHER_SCRIPT_PATH):
        sys.exit(
            f"FATAL: cannot find {MATCHER_SCRIPT_PATH} to import "
            f"assign_deciles / match_by_decile / _load_gate_module from. "
            f"Not proceeding with a re-implemented copy of that logic."
        )
    spec = importlib.util.spec_from_file_location("matcher", MATCHER_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_schema(df: pd.DataFrame, expected: set, source_label: str) -> None:
    actual = set(df.columns)
    if not expected.issubset(actual):
        sys.exit(
            f"SCHEMA MISMATCH in {source_label}\n"
            f"  expected (subset of): {sorted(expected)}\n"
            f"  actual columns:       {sorted(actual)}"
        )


def build_switching_and_pool_tpm(matcher, cell_line: str, gate_path: str, expr_path: str):
    """
    Reproduces the exact merge path from matcher.run_real_data() for a
    single cell line: load gate calls, build switching/stable sets via
    the real sgev.build_switching_sets(), load expression, merge to
    get mean_tpm for both switching and non-switching (pool) genes.

    Returns (switching_tpm_df, nonswitching_tpm_df), both with columns
    [gene_id, mean_tpm].
    """
    sgev = matcher._load_gate_module()

    gm_gate_df = sgev.load_and_validate(GM12878_GATE_PATH, sgev.REQUIRED_GATE_COLS, sep="\t")
    k562_gate_df = sgev.load_and_validate(K562_GATE_PATH, sgev.REQUIRED_GATE_COLS, sep="\t")
    switching, stable, _and_inconsistent = sgev.build_switching_sets(gm_gate_df, k562_gate_df)

    if not os.path.exists(expr_path):
        sys.exit(
            f"FATAL: {expr_path} not found. Run reshape_encode_rnaseq_v1.py "
            f"first to produce the cleaned wide replicate table."
        )
    expr = pd.read_csv(expr_path)
    _check_schema(expr, EXPECTED_EXPR_COLUMNS, f"{cell_line} expression file")
    expr = expr.copy()
    expr["mean_tpm"] = expr[["rep1", "rep2"]].mean(axis=1)
    expr_slim = expr[["gene_id", "mean_tpm"]]

    switching_tpm = expr_slim.merge(switching[["gene_id"]], on="gene_id", how="inner")
    nonswitching_tpm = expr_slim.merge(stable[["gene_id"]], on="gene_id", how="inner")

    print(f"  [{cell_line}] switching genes with expression data: {len(switching_tpm)} (of {len(switching)})")
    print(f"  [{cell_line}] non-switching pool with expression data: {len(nonswitching_tpm)} (of {len(stable)})")

    return switching_tpm, nonswitching_tpm


def compute_diagnostics(matcher, switching_tpm: pd.DataFrame, nonswitching_tpm: pd.DataFrame):
    """
    Given a cell line's switching and non-switching TPM tables, compute:
      - pooled decile edges (via matcher.assign_deciles on the pooled
        switching+non-switching distribution -- same edge-fitting call
        match_by_decile() uses internally)
      - per-decile switching count, non-switching pool count, and the
        ratio switching_count / pool_count

    Returns (edges, concentration_df) where concentration_df has columns
    [decile, switch_n, pool_n, switch_pool_ratio].
    """
    pooled = pd.concat(
        [switching_tpm["mean_tpm"], nonswitching_tpm["mean_tpm"]], ignore_index=True
    )
    _, edges = matcher.assign_deciles(pooled)

    sw_deciles, _ = matcher.assign_deciles(switching_tpm["mean_tpm"], edges=edges)
    ns_deciles, _ = matcher.assign_deciles(nonswitching_tpm["mean_tpm"], edges=edges)

    sw_counts = pd.Series(sw_deciles).value_counts().reindex(range(N_DECILES), fill_value=0).sort_index()
    ns_counts = pd.Series(ns_deciles).value_counts().reindex(range(N_DECILES), fill_value=0).sort_index()

    ratio = sw_counts / ns_counts.replace(0, np.nan)

    concentration_df = pd.DataFrame({
        "decile": range(N_DECILES),
        "switch_n": sw_counts.to_numpy(),
        "pool_n": ns_counts.to_numpy(),
        "switch_pool_ratio": ratio.to_numpy(),
    })
    return edges, concentration_df


def synthetic_self_test() -> None:
    """
    Two synthetic cell lines, both with the SAME switching set (n=1,000,
    80% concentrated in deciles 8-9 by construction) but DIFFERENT
    non-switching pool shapes:

      - "EVEN": non-switching pool is wide and dense everywhere
        (n=40,000, sigma=1.8 lognormal) -- expected: switch_pool_ratio
        roughly flat/low across all deciles, no spike.

      - "THIN": non-switching pool is deliberately starved in the
        high-TPM region (deciles 8-9 pulled down to n~150 total via a
        truncated draw) while dense elsewhere -- expected:
        switch_pool_ratio spikes sharply in deciles 8-9 ONLY, since the
        same switching concentration divided by a much smaller pool
        produces a much larger ratio there.

    This validates that compute_diagnostics() actually recovers a known
    thin-pool signature rather than just producing noise, before it's
    trusted on real K562/GM12878 data.
    """
    rng = np.random.default_rng(57)
    matcher = _load_matcher_module()

    n_switch = 1000
    high_tpm = rng.lognormal(mean=4.2, sigma=0.25, size=int(n_switch * 0.8))
    low_tpm = rng.lognormal(mean=1.0, sigma=0.8, size=int(n_switch * 0.2))
    switching_df = pd.DataFrame({
        "gene_id": [f"SYN_SW_{i}" for i in range(len(high_tpm) + len(low_tpm))],
        "mean_tpm": np.concatenate([high_tpm, low_tpm]),
    })

    # EVEN pool: dense everywhere
    n_even = 40000
    even_pool = pd.DataFrame({
        "gene_id": [f"SYN_EVEN_{i}" for i in range(n_even)],
        "mean_tpm": rng.lognormal(mean=2.5, sigma=1.8, size=n_even),
    })

    # THIN pool: same generative shape, but genes with log1p(tpm) landing
    # in the top ~20% of the pooled distribution are subsampled down to
    # a near-starvation level, mimicking a pool that's thin exactly
    # where the switching set concentrates.
    n_thin_base = 40000
    thin_pool_raw = pd.DataFrame({
        "gene_id": [f"SYN_THIN_{i}" for i in range(n_thin_base)],
        "mean_tpm": rng.lognormal(mean=2.5, sigma=1.8, size=n_thin_base),
    })
    pooled_for_cutoff = pd.concat([switching_df["mean_tpm"], thin_pool_raw["mean_tpm"]], ignore_index=True)
    high_cutoff = np.percentile(np.log1p(pooled_for_cutoff), 80)
    is_high = np.log1p(thin_pool_raw["mean_tpm"]) >= high_cutoff
    high_rows = thin_pool_raw[is_high]
    low_rows = thin_pool_raw[~is_high]
    starved_high = high_rows.sample(n=min(150, len(high_rows)), random_state=57)
    thin_pool = pd.concat([low_rows, starved_high], ignore_index=True)

    edges_even, conc_even = compute_diagnostics(matcher, switching_df, even_pool)
    edges_thin, conc_thin = compute_diagnostics(matcher, switching_df, thin_pool)

    print("[SELF-TEST] EVEN pool switch_pool_ratio by decile:")
    print(conc_even[["decile", "switch_n", "pool_n", "switch_pool_ratio"]].to_string(index=False))
    print("[SELF-TEST] THIN pool switch_pool_ratio by decile:")
    print(conc_thin[["decile", "switch_n", "pool_n", "switch_pool_ratio"]].to_string(index=False))

    # Sanity check: total switch_n across all deciles must equal the
    # fixture's switching set size, for both pools -- confirms
    # compute_diagnostics() isn't silently dropping genes during binning.
    if conc_even["switch_n"].sum() != len(switching_df):
        print(f"[SELF-TEST FAIL] EVEN: switch_n sums to {conc_even['switch_n'].sum()}, expected {len(switching_df)}.")
        sys.exit(1)
    if conc_thin["switch_n"].sum() != len(switching_df):
        print(f"[SELF-TEST FAIL] THIN: switch_n sums to {conc_thin['switch_n'].sum()}, expected {len(switching_df)}.")
        sys.exit(1)

    # Primary pass/fail check: this deliberately does NOT compare decile
    # bins directly between EVEN and THIN. assign_deciles() fits edges on
    # each fixture's own pooled (switching+non-switching) distribution
    # (matching how the real per-cell-line matching works), so starving
    # part of the THIN pool shifts ITS edges relative to EVEN's -- the
    # same boundary-jitter effect the real GM12878-vs-K562 comparison
    # has to be robust to. So instead the check uses a threshold that is
    # EXTERNAL to either fixture's own quantile computation: the
    # switching set's own median TPM (identical between both fixtures,
    # since switching_df is shared). Counting non-switching pool genes
    # at or above that fixed external threshold isolates the "thin at
    # the high end" signal without being confounded by where the
    # decile boundaries happened to land in each fixture.
    high_tpm_threshold = switching_df["mean_tpm"].median()
    even_high_pool_n = (even_pool["mean_tpm"] >= high_tpm_threshold).sum()
    thin_high_pool_n = (thin_pool["mean_tpm"] >= high_tpm_threshold).sum()

    print(
        f"[SELF-TEST] pool genes with mean_tpm >= switching median "
        f"({high_tpm_threshold:.2f}): EVEN={even_high_pool_n}, THIN={thin_high_pool_n}"
    )

    if even_high_pool_n == 0:
        print("[SELF-TEST FAIL] EVEN pool has zero genes above the switching median -- fixture is degenerate.")
        sys.exit(1)

    starvation_ratio = thin_high_pool_n / even_high_pool_n
    if starvation_ratio > 0.15:
        print(
            f"[SELF-TEST FAIL] expected THIN pool's high-TPM gene count to be "
            f"<=15% of EVEN's (deliberately starved fixture), got "
            f"{starvation_ratio:.2%} ({thin_high_pool_n} vs {even_high_pool_n}). "
            f"Fixture construction did not actually starve the high-TPM region."
        )
        sys.exit(1)

    # Confirm the starvation actually shows up as an ELEVATED switch_pool_ratio
    # somewhere in the THIN run's decile table (using THIN's own edges,
    # which is the only self-consistent way to read that table) --
    # not just that the raw pool counts differ, but that
    # compute_diagnostics() actually surfaces it as a ratio spike.
    max_ratio_thin = conc_thin["switch_pool_ratio"].max()
    max_ratio_even = conc_even["switch_pool_ratio"].max()
    if max_ratio_thin <= max_ratio_even:
        print(
            f"[SELF-TEST FAIL] expected THIN run's peak switch_pool_ratio "
            f"({max_ratio_thin:.4f}) to exceed EVEN run's peak "
            f"({max_ratio_even:.4f}) -- the starved fixture should produce a "
            f"more extreme ratio spike somewhere in its decile table."
        )
        sys.exit(1)

    print(
        f"[SELF-TEST PASS] THIN pool has {starvation_ratio:.1%} of EVEN's "
        f"high-TPM gene count (starvation confirmed via external threshold), "
        f"and produces a higher peak switch_pool_ratio ({max_ratio_thin:.4f} "
        f"vs {max_ratio_even:.4f})."
    )


def run_real_data() -> None:
    matcher = _load_matcher_module()

    print("=== GM12878 ===")
    gm_switch_tpm, gm_pool_tpm = build_switching_and_pool_tpm(
        matcher, "gm12878", GM12878_GATE_PATH, GM12878_EXPR_PATH
    )
    gm_edges, gm_conc = compute_diagnostics(matcher, gm_switch_tpm, gm_pool_tpm)

    print("\n=== K562 ===")
    k5_switch_tpm, k5_pool_tpm = build_switching_and_pool_tpm(
        matcher, "k562", K562_GATE_PATH, K562_EXPR_PATH
    )
    k5_edges, k5_conc = compute_diagnostics(matcher, k5_switch_tpm, k5_pool_tpm)

    # --- Output 1: edge comparison ---
    n_edges = len(gm_edges)
    if len(k5_edges) != n_edges:
        sys.exit(
            f"FATAL: GM12878 produced {n_edges} decile edges, K562 produced "
            f"{len(k5_edges)}. assign_deciles() should always return "
            f"N_DECILES+1 edges regardless of input -- this indicates a "
            f"deeper problem in the imported matcher module, not something "
            f"to paper over here."
        )
    edge_comparison = pd.DataFrame({
        "decile_boundary": range(n_edges),
        "gm12878_log_tpm_edge": gm_edges,
        "k562_log_tpm_edge": k5_edges,
    })
    # first/last edges are +-inf by construction (see assign_deciles); the
    # diff column is only meaningful for the interior boundaries.
    edge_comparison["edge_diff_k562_minus_gm12878"] = np.where(
        np.isfinite(edge_comparison["gm12878_log_tpm_edge"]) & np.isfinite(edge_comparison["k562_log_tpm_edge"]),
        edge_comparison["k562_log_tpm_edge"] - edge_comparison["gm12878_log_tpm_edge"],
        np.nan,
    )
    edge_comparison.to_csv(OUT_EDGES_PATH, sep="\t", index=False)

    # --- Output 2: switch/pool concentration side by side ---
    concentration = gm_conc.merge(k5_conc, on="decile", suffixes=("_gm12878", "_k562"))
    concentration.to_csv(OUT_CONCENTRATION_PATH, sep="\t", index=False)

    print("\n=== DECILE EDGE COMPARISON ===")
    print(edge_comparison.to_string(index=False))
    print("\n=== SWITCH/POOL CONCENTRATION BY DECILE ===")
    print(concentration.to_string(index=False))
    print(f"\nWrote {OUT_EDGES_PATH}")
    print(f"Wrote {OUT_CONCENTRATION_PATH}")

    # --- Flag, do not interpret: point at the deciles worth writing about ---
    gm_ratio = concentration["switch_pool_ratio_gm12878"]
    k5_ratio = concentration["switch_pool_ratio_k562"]
    gap = (k5_ratio - gm_ratio).abs()
    flagged = concentration.loc[gap.sort_values(ascending=False).index[:3], "decile"].tolist()
    print(
        f"\n[FLAG] largest GM12878-vs-K562 switch_pool_ratio gaps at deciles: "
        f"{flagged}. Inspect these against the K562 KS failure before writing "
        f"the scoping decision -- this script reports the pattern, it does "
        f"not decide what it means."
    )


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