"""
matched_nonswitching_v1.py  (v2 — corrected against real Day 55 pipeline)

Day 56 | Epigenetic Logic Circuits

CORRECTION FROM FIRST DRAFT: the original version of this script assumed
switching_genes_v1.tsv / nonswitching_genes_v1.tsv existed as standalone
files. They do not. switching_gene_expression_variability_v1.py (Day 55)
computes these sets INLINE via build_switching_sets(gm_gate_df,
k562_gate_df) on every run -- there is no persisted intermediate file.
This version imports that function live via importlib rather than
guessing a file that doesn't exist. Confirmed against real data:
switching=7,474, stable/non-switching=9,947 (matches the Day 55 forward
note exactly).

Purpose
-------
Day 55's switching (n=7,474) vs non-switching (n=9,947) noise-residual
comparison failed the Hansen confound check (KS p ~ 2e-215 GM12878,
~1e-188 K562): raw mean expression differs badly between the two
groups. sample_matched_nonswitching() in the Day 55 script drew a FLAT
random non-switching sample (stable_df.sample(n=n)) -- no expression
matching at all, which is exactly why the confound fired.

This script replaces that flat draw with a decile-stratified draw,
done SEPARATELY PER CELL LINE (GM12878 and K562 have different mean-TPM
distributions for the same genes, so a single pooled match would still
leave one cell line's comparison confounded).

Inputs (real, confirmed)
-------------------------
  data/gate_assignments_named.tsv        (GM12878 gate calls)
  data/k562_gate_assignments_named.tsv   (K562 gate calls)
  data/encode_rnaseq_gm12878_replicates_v1.csv  (from reshape_encode_rnaseq_v1.py)
  data/encode_rnaseq_k562_replicates_v1.csv     (from reshape_encode_rnaseq_v1.py)

Outputs
-------
  data/matched_nonswitching_gm12878.tsv  (gene_id, mean_tpm, decile)
  data/matched_nonswitching_k562.tsv     (gene_id, mean_tpm, decile)
  Row count per file == 7,474 (the switching-set size), unless a decile
  bin is starved in the non-switching pool, in which case the script
  prints a named [WARN] with the exact shortfall rather than silently
  padding or proceeding as if the match were exact.

Usage
-----
    python matched_nonswitching_v1.py --self-test   # synthetic fixture only
    python matched_nonswitching_v1.py                 # self-test, then real-data run
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
from scipy.stats import chisquare

RNG_SEED = 56
N_DECILES = 10

GATE_SCRIPT_PATH = "code/switching_gene_expression_variability_v1.py"
GM12878_GATE_PATH = "data/gate_assignments_named.tsv"
K562_GATE_PATH = "data/k562_gate_assignments_named.tsv"
GM12878_EXPR_PATH = "data/encode_rnaseq_gm12878_replicates_v1.csv"
K562_EXPR_PATH = "data/encode_rnaseq_k562_replicates_v1.csv"

OUT_PATHS = {
    "gm12878": "data/matched_nonswitching_gm12878.tsv",
    "k562": "data/matched_nonswitching_k562.tsv",
}

EXPECTED_EXPR_COLUMNS = {"gene_id", "rep1", "rep2"}


def _load_gate_module():
    """
    Cross-script import via importlib.util.spec_from_file_location, per
    project convention (never sys.path manipulation). Loading this module
    does NOT trigger its __main__ block (guarded by if __name__ ==
    "__main__"), confirmed before this script was written.
    """
    if not os.path.exists(GATE_SCRIPT_PATH):
        sys.exit(
            f"FATAL: cannot find {GATE_SCRIPT_PATH} to import "
            f"build_switching_sets / load_and_validate from. Not proceeding "
            f"with a re-implemented copy of that logic -- import the real one."
        )
    spec = importlib.util.spec_from_file_location("sgev", GATE_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_schema(df: pd.DataFrame, expected: set, source_label: str) -> None:
    actual = set(df.columns)
    if not expected.issubset(actual):
        print(f"[SCHEMA MISMATCH] {source_label}")
        print(f"  expected (subset of): {sorted(expected)}")
        print(f"  actual columns:       {sorted(actual)}")
        sys.exit(1)


def assign_deciles(mean_tpm: pd.Series, edges: np.ndarray = None):
    """
    Assign decile bins (0-9) based on provided bin edges (fit on log1p
    TPM to keep bin edges well-behaved across the long-tailed TPM
    distribution), or compute edges from this series if none given.
    Returns (decile_labels, edges).
    """
    log_tpm = np.log1p(mean_tpm.to_numpy())
    if edges is None:
        quantiles = np.linspace(0, 100, N_DECILES + 1)
        edges = np.percentile(log_tpm, quantiles)
        edges[0] = -np.inf
        edges[-1] = np.inf
    deciles = np.digitize(log_tpm, edges[1:-1], right=False)
    return deciles, edges


def match_by_decile(switching_tpm: pd.DataFrame, nonswitching_tpm: pd.DataFrame,
                     rng: np.random.Generator):
    """
    Bin edges computed on the POOLED (switching + non-switching)
    distribution so both share identical boundaries. Sample non-switching
    genes proportionally to the switching set's decile distribution.

    Returns (matched_df, edges). Any caller verifying the match must bin
    the switching set against these SAME edges -- recomputing edges from
    the switching set alone trivially yields a uniform 10%-per-decile
    split regardless of its real shape (this bug was caught and fixed
    during Block 1 self-test debugging; see synthetic_self_test()).
    """
    pooled = pd.concat(
        [switching_tpm["mean_tpm"], nonswitching_tpm["mean_tpm"]], ignore_index=True
    )
    _, edges = assign_deciles(pooled)

    sw_deciles, _ = assign_deciles(switching_tpm["mean_tpm"], edges=edges)
    ns_deciles, _ = assign_deciles(nonswitching_tpm["mean_tpm"], edges=edges)

    switching_tpm = switching_tpm.copy()
    nonswitching_tpm = nonswitching_tpm.copy()
    switching_tpm["decile"] = sw_deciles
    nonswitching_tpm["decile"] = ns_deciles

    n_target = len(switching_tpm)
    sw_counts = switching_tpm["decile"].value_counts().reindex(range(N_DECILES), fill_value=0)
    sw_fracs = sw_counts / sw_counts.sum()

    raw_targets = sw_fracs * n_target
    int_targets = np.floor(raw_targets).astype(int)
    remainder = n_target - int_targets.sum()

    frac_parts = (raw_targets - int_targets).sort_values(ascending=False)
    for decile in frac_parts.index:
        if remainder <= 0:
            break
        available = (nonswitching_tpm["decile"] == decile).sum()
        if available > int_targets[decile]:
            int_targets[decile] += 1
            remainder -= 1

    sampled_frames = []
    shortfalls = []
    for decile, target_n in int_targets.items():
        if target_n == 0:
            continue
        pool = nonswitching_tpm[nonswitching_tpm["decile"] == decile]
        if len(pool) < target_n:
            shortfalls.append((decile, target_n, len(pool)))
            take_n = len(pool)
        else:
            take_n = target_n
        if take_n > 0:
            idx = rng.choice(pool.index.to_numpy(), size=take_n, replace=False)
            sampled_frames.append(nonswitching_tpm.loc[idx])

    for decile, requested, available in shortfalls:
        print(
            f"  [WARN] decile {decile}: requested {requested}, only "
            f"{available} available in non-switching pool. Taking all "
            f"available -- matched total will fall short by "
            f"{requested - available}."
        )

    matched = pd.concat(sampled_frames, ignore_index=True) if sampled_frames else \
        nonswitching_tpm.iloc[0:0].copy()
    return matched, edges


def synthetic_self_test() -> None:
    """
    Non-degenerate fixture: switching population deliberately concentrated
    in high deciles (80% high / 20% low), non-switching pool wide and
    dense enough (n=60000, sigma=1.8) that no decile the switching set
    touches is starved -- this test validates the matching LOGIC, not the
    starvation-fallback path.
    """
    rng = np.random.default_rng(RNG_SEED)

    n_switch = 2000
    high_tpm = rng.lognormal(mean=4.0, sigma=0.3, size=int(n_switch * 0.8))
    low_tpm = rng.lognormal(mean=1.0, sigma=0.8, size=int(n_switch * 0.2))
    switching_df = pd.DataFrame({
        "gene_id": [f"SYN_SW_{i}" for i in range(len(high_tpm) + len(low_tpm))],
        "mean_tpm": np.concatenate([high_tpm, low_tpm]),
    })

    n_nonswitch = 60000
    nonswitching_df = pd.DataFrame({
        "gene_id": [f"SYN_NS_{i}" for i in range(n_nonswitch)],
        "mean_tpm": rng.lognormal(mean=2.5, sigma=1.8, size=n_nonswitch),
    })

    matched, edges = match_by_decile(switching_df, nonswitching_df, rng)

    if len(matched) != len(switching_df):
        print(f"[SELF-TEST FAIL] matched size {len(matched)} != switching size {len(switching_df)}")
        sys.exit(1)

    sw_deciles, _ = assign_deciles(switching_df["mean_tpm"], edges=edges)
    switching_df = switching_df.copy()
    switching_df["decile"] = sw_deciles
    matched_deciles, _ = assign_deciles(matched["mean_tpm"], edges=edges)
    matched = matched.copy()
    matched["decile"] = matched_deciles

    sw_counts = switching_df["decile"].value_counts().reindex(range(N_DECILES), fill_value=0).sort_index()
    matched_counts = matched["decile"].value_counts().reindex(range(N_DECILES), fill_value=0).sort_index()

    expected = sw_counts / sw_counts.sum() * matched_counts.sum()
    expected = expected.replace(0, 1e-6)
    stat, p_value = chisquare(f_obs=matched_counts.to_numpy(), f_exp=expected.to_numpy())

    print(f"[SELF-TEST] matched size = {len(matched)} (target {len(switching_df)})")
    print(f"[SELF-TEST] switching decile counts:\n{sw_counts.to_dict()}")
    print(f"[SELF-TEST] matched decile counts:\n{matched_counts.to_dict()}")
    print(f"[SELF-TEST] chi-square p-value = {p_value:.6f}")

    if p_value <= 0.5:
        print(
            f"[SELF-TEST FAIL] matched decile distribution does not track "
            f"switching decile distribution closely enough (p={p_value:.6f}, "
            f"require p > 0.5). Matching logic is likely falling back to "
            f"unstratified sampling."
        )
        sys.exit(1)

    print("[SELF-TEST PASS] matched sample tracks switching decile shape.")


def run_real_data() -> None:
    sgev = _load_gate_module()

    gm_gate_df = sgev.load_and_validate(GM12878_GATE_PATH, sgev.REQUIRED_GATE_COLS, sep="\t")
    k562_gate_df = sgev.load_and_validate(K562_GATE_PATH, sgev.REQUIRED_GATE_COLS, sep="\t")
    switching, stable, and_inconsistent = sgev.build_switching_sets(gm_gate_df, k562_gate_df)

    print(f"Switching genes: {len(switching)} | Non-switching (stable): {len(stable)} "
          f"| AND<->INCONSISTENT subset: {len(and_inconsistent)}")

    for cell_line, expr_path in [("gm12878", GM12878_EXPR_PATH), ("k562", K562_EXPR_PATH)]:
        print(f"\n=== {cell_line.upper()} ===")

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

        switching_tpm = expr_slim.merge(
            switching[["gene_id"]], on="gene_id", how="inner"
        )
        nonswitching_tpm = expr_slim.merge(
            stable[["gene_id"]], on="gene_id", how="inner"
        )

        print(f"  switching genes with expression data: {len(switching_tpm)} (of {len(switching)})")
        print(f"  non-switching genes with expression data: {len(nonswitching_tpm)} (of {len(stable)})")
        if len(switching_tpm) != len(switching):
            print(
                f"  [WARN] {len(switching) - len(switching_tpm)} switching genes had "
                f"no matching expression row -- check gene_id format consistency "
                f"before trusting downstream counts."
            )

        rng = np.random.default_rng(RNG_SEED)
        matched, _edges = match_by_decile(switching_tpm, nonswitching_tpm, rng)

        out_path = OUT_PATHS[cell_line]
        matched.to_csv(out_path, sep="\t", index=False)
        print(f"  matched non-switching set: {len(matched)} rows -> {out_path}")
        if len(matched) != len(switching_tpm):
            print(
                f"  [WARN] matched set size {len(matched)} != switching set size "
                f"{len(switching_tpm)}. Report this exact shortfall in the "
                f"verdict doc -- do not treat the match as exact."
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