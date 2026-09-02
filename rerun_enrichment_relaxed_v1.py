"""
rerun_enrichment_relaxed_v1.py

Purpose:
    Relaxed-threshold (nominal P-value < 0.05, NOT FDR-corrected) second pass
    over the switching-gene enrichment already computed on Day 47. This does
    NOT reimplement enrichment logic -- it imports and reuses three functions
    verbatim from run_pathway_enrichment_v1.py via importlib.util:
        - load_symbol_map()
        - load_background_and_origin_groups(symbol_map)
        - run_enrichment(gene_symbols, background_symbols, label)

    IMPORTANT (confirmed by reading the real Day 47 script, not guessed):
    switching_genes_gm12878_v1.txt and switching_genes_k562_v1.txt are
    IDENTICAL gene sets by construction -- Day 47 documented this explicitly
    and does NOT enrich on them. The actual per-cell-line comparison is the
    and_origin split (GM12878-origin vs K562-origin AND<->INC switchers)
    derived from switching_genes_merged_v1.tsv. This script reuses that same
    derivation via load_background_and_origin_groups() rather than reading
    the .txt files, or it would silently answer nothing (two identical
    relaxed-threshold runs on the same 3,690 genes).

    Output is explicitly labeled EXPLORATORY. The Day 48 FDR-corrected Case B
    null is NOT overturned by anything this script produces -- nominal p<0.05
    is a "is there any qualitative directional signal worth a Discussion
    sentence" check, not a statistical claim.

Hard rules enforced:
    - Threading env vars set before any numeric imports.
    - Synthetic self-test MUST pass before real data is touched. Uses the
      SAME real run_enrichment() call path (offline gp.enrich, local
      hypergeometric test) so the self-test actually exercises the code this
      script runs on real data -- not a separate mocked-up filter check.
    - FATAL exit (no silent coercion) on missing required Block 1 outputs,
      missing expected functions in the Day 47 module, or unexpected output
      columns.
    - Must be run from the same working directory as run_pathway_enrichment_v1.py
      (i.e. LOGIC CIRCUITS/code/), since load_background_and_origin_groups()
      reads GM_REGEN / K562_REGEN / SWITCHERS_MERGED as paths relative to cwd,
      exactly as Day 47 script does. This script does not change that.

*** ONE THING TO CONFIRM BEFORE RUNNING ***
    EXPECTED_ENRICHMENT_COLUMNS below assumes gseapy's standard gp.enrich()
    output includes a raw 'P-value' column alongside 'Adjusted P-value' (Day 47's
    own script only ever reads 'Adjusted P-value', so this hasn't been
    explicitly confirmed against your installed gseapy version's actual output).
    The script will FATAL exit with the real column list printed if 'P-value'
    isn't there -- if that happens, don't guess a substitute column, read the
    printed columns and tell me what's actually present.
"""

import os

# ---- Threading fix: MUST be set before any numeric imports ----
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import importlib.util
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
DAY47_SCRIPT_PATH = "run_pathway_enrichment_v1.py"  # relative to cwd, matching Day 47 convention
DAY47_MODULE_ALIAS = "day47_enrichment"

OUTPUT_GM12878 = "enrichment_relaxed_gm12878_origin_v1.csv"
OUTPUT_K562 = "enrichment_relaxed_k562_origin_v1.csv"

NOMINAL_P_THRESHOLD = 0.05

# gseapy's gp.enrich() standard local-hypergeometric-test output columns.
# 'P-value' is the raw/nominal value this script filters on; 'Adjusted P-value'
# is the FDR-corrected value Day 47 already used and is carried through here
# unfiltered for reference only, not re-tested.
EXPECTED_ENRICHMENT_COLUMNS = ["Term", "Overlap", "P-value", "Adjusted P-value", "Genes"]

# Reused directly from Day 47's own positive control -- known cell-cycle genes.
# Used here ONLY to validate the relaxed-threshold filtering logic runs
# correctly end-to-end through the real run_enrichment() call before real
# switching-gene data is touched. A separate, unrelated decoy set is added so
# the self-test has both a signal group and a noise group, mirroring what a
# real gene list looks like.
SELF_TEST_SIGNAL_GENES = [
    "CCNB1", "CDK1", "CCNE1", "MCM2", "PCNA",
    "CDC20", "AURKA", "PLK1", "BUB1", "TOP2A",
]
SELF_TEST_DECOY_GENES = [
    "ACTB", "GAPDH", "TUBB", "RPL13A", "HPRT1",
    "B2M", "YWHAZ", "SDHA", "PPIA", "TBP",
    "GUSB", "HMBS", "PGK1", "POLR2A", "RPLP0",
    "TFRC", "UBC", "YWHAE", "ALAS1", "IPO8",
]
SELF_TEST_KEYWORDS = [
    "cell cycle", "e2f targets", "g2-m checkpoint", "g2m checkpoint", "mitotic spindle",
]


def fatal(msg: str) -> None:
    print("\n" + "=" * 70)
    print("FATAL:", msg)
    print("=" * 70)
    sys.exit(1)


def load_day47_module(script_path: str, module_alias: str):
    """
    Load run_pathway_enrichment_v1.py via importlib.util, per the project's
    locked import convention. Importing does NOT run main() (the Day 47
    script guards it with if __name__ == "__main__"), so this is safe --
    it does not re-trigger Day 47's own self-test, file writes, or its
    online Enrichr positive-control call.
    """
    if not os.path.isfile(script_path):
        fatal(f"Day 47 script not found at '{script_path}' (resolved from cwd: {os.getcwd()}). "
              f"This script must be run from the same directory as run_pathway_enrichment_v1.py.")

    spec = importlib.util.spec_from_file_location(module_alias, script_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        fatal(f"Day 47 script at '{script_path}' called sys.exit() during import -- "
              f"most likely gseapy is not installed (see the error it printed above). "
              f"Fix that first (conda activate genomics; pip install gseapy), then rerun this script.")
    except Exception as e:
        fatal(f"Day 47 script at '{script_path}' failed to import cleanly.\n"
              f"Underlying error: {type(e).__name__}: {e}")

    required_names = ["load_symbol_map", "load_background_and_origin_groups", "run_enrichment"]
    missing = [n for n in required_names if not hasattr(module, n)]
    if missing:
        available = [n for n in dir(module) if not n.startswith("_") and callable(getattr(module, n, None))]
        fatal(f"Day 47 module is missing expected function(s): {missing}.\n"
              f"Callable names actually found: {available}\n"
              f"If run_pathway_enrichment_v1.py has been renamed/refactored since Day 47, "
              f"update the function names this script imports -- do not guess a substitute.")

    return module


def validate_enrichment_output_schema(df: pd.DataFrame, source_label: str):
    """FATAL exit if the dataframe is missing an expected column. No renaming, no guessing."""
    missing = [c for c in EXPECTED_ENRICHMENT_COLUMNS if c not in df.columns]
    if missing:
        fatal(f"Enrichment output for {source_label} is missing expected column(s): {missing}.\n"
              f"Actual columns found: {list(df.columns)}\n"
              f"Update EXPECTED_ENRICHMENT_COLUMNS in this script's CONFIG to match the real "
              f"columns, or investigate why gp.enrich()'s output shape changed, before proceeding.")


def synthetic_self_test(day47_module) -> None:
    """
    Real call through the actual reused run_enrichment() function -- offline
    gp.enrich() against real gene set libraries, real background -- using a
    known cell-cycle positive control (10 genes) plus 20 unrelated
    housekeeping decoy genes as background. Confirms the relaxed-threshold
    (nominal P-value < 0.05) filter recovers a cell-cycle-related term before
    any real switching-gene data is touched.

    This requires network access to fetch gene set libraries (gp.get_library),
    same as Day 47's own runs -- if this fails on connectivity grounds, that's
    an environment issue to fix, not a logic error in this script.
    """
    print("=" * 60)
    print("SYNTHETIC SELF-TEST -- must pass before real data is touched")
    print("=" * 60)

    # NOTE (fixed after real run): foreground must be a proper SUBSET of
    # background, not identical to it. gp.enrich()'s hypergeometric survival
    # function is P(X>=k | N=background size, K=term hits in background,
    # n=foreground size). If n==N (foreground==background), the draw is
    # deterministic and every term's p-value collapses to 1.0 regardless of
    # real signal -- confirmed analytically: with foreground=background=30,
    # p=1.0 for a term with perfect overlap; with foreground=10 (signal only)
    # and background=30 (signal+decoy), the same overlap gives p=3e-08.
    combined = SELF_TEST_SIGNAL_GENES + SELF_TEST_DECOY_GENES
    result = day47_module.run_enrichment(
        gene_symbols=SELF_TEST_SIGNAL_GENES,
        background_symbols=combined,
        label="SYNTHETIC_SELF_TEST",
    )

    if result is None or not isinstance(result, pd.DataFrame) or result.empty:
        fatal("Self-test: run_enrichment() returned no results (None, wrong type, or empty "
              "DataFrame) on the known positive-control gene list. Do not proceed to real "
              "data -- confirm gseapy/network connectivity first (Day 47's own self-test "
              "already validated this pipeline once, so a failure here likely means "
              "something changed in the environment since then).")

    validate_enrichment_output_schema(result, "SYNTHETIC_SELF_TEST")

    relaxed = result[result["P-value"] < NOMINAL_P_THRESHOLD]
    if relaxed.empty:
        fatal(f"Self-test: zero terms recovered at nominal P-value<{NOMINAL_P_THRESHOLD} "
              f"on the known cell-cycle positive control. The relaxed-threshold filtering "
              f"logic cannot be trusted on real data until this is understood -- do not proceed.")

    terms_lower = relaxed["Term"].str.lower()
    hit = terms_lower.apply(lambda t: any(k in t for k in SELF_TEST_KEYWORDS)).any()
    if not hit:
        fatal(f"Self-test: {len(relaxed)} term(s) passed nominal P-value<{NOMINAL_P_THRESHOLD}, "
              f"but none matched expected cell-cycle keywords {SELF_TEST_KEYWORDS}. "
              f"Got terms: {relaxed['Term'].tolist()}. Do not proceed -- something about the "
              f"positive control or filtering is not behaving as expected.")

    print(f"SELF-TEST PASSED: {len(relaxed)} term(s) at nominal P-value<{NOMINAL_P_THRESHOLD}, "
          f"including at least one expected cell-cycle-related term.")
    print("Proceeding to real switching-gene data.")
    print("=" * 60)


def run_relaxed_pass(day47_module, gene_symbols: list, background_symbols: list,
                      label: str, output_path: str) -> pd.DataFrame:
    if len(gene_symbols) < 5:
        print(f"{label}: only {len(gene_symbols)} gene symbol(s) available -- too few for a "
              f"meaningful enrichment call (same floor Day 47 used). Skipping, not fabricating output.")
        return pd.DataFrame()

    result = day47_module.run_enrichment(
        gene_symbols=gene_symbols,
        background_symbols=background_symbols,
        label=f"{label} (RELAXED PASS)",
    )

    if result is None or not isinstance(result, pd.DataFrame):
        fatal(f"{label}: run_enrichment() did not return a DataFrame. Got: {type(result)}")

    if result.empty:
        print(f"{label}: run_enrichment() returned an empty DataFrame (consistent with the "
              f"Day 48 Case B null). Writing empty relaxed-pass output -- this is a real "
              f"result, not a script failure.")
        result.to_csv(output_path, index=False)
        return result

    validate_enrichment_output_schema(result, label)

    relaxed = result[result["P-value"] < NOMINAL_P_THRESHOLD].copy()
    relaxed = relaxed.sort_values("P-value", ascending=True)
    relaxed["label"] = "EXPLORATORY -- NOT FDR-CORRECTED"
    relaxed["origin_group"] = label
    relaxed["nominal_p_threshold_used"] = NOMINAL_P_THRESHOLD

    relaxed.to_csv(output_path, index=False)
    print(f"{label}: {len(relaxed)} / {len(result)} tested terms at nominal "
          f"P-value<{NOMINAL_P_THRESHOLD} -> {output_path}")
    print(f"{label}: REMINDER -- these are uncorrected p-values. None carry the statistical "
          f"weight of the Day 48 FDR-corrected Case B result.")
    return relaxed


def main():
    day47 = load_day47_module(DAY47_SCRIPT_PATH, DAY47_MODULE_ALIAS)

    synthetic_self_test(day47)

    print("\nLoading real symbol map and origin groups (reused from Day 47, not recomputed)...")
    symbol_map = day47.load_symbol_map()
    background_symbols, gm_origin_symbols, k562_origin_symbols, n_gm_ids, n_k562_ids = \
        day47.load_background_and_origin_groups(symbol_map)

    print(f"Background (tested universe): {len(background_symbols)} symbols")
    print(f"GM12878-origin AND<->INC switchers: {n_gm_ids} gene_id -> {len(gm_origin_symbols)} symbols")
    print(f"K562-origin AND<->INC switchers: {n_k562_ids} gene_id -> {len(k562_origin_symbols)} symbols")

    print("\nRunning relaxed pass: GM12878-origin switchers...")
    gm_relaxed = run_relaxed_pass(
        day47, gm_origin_symbols, background_symbols, "GM12878-origin", OUTPUT_GM12878
    )

    print("\nRunning relaxed pass: K562-origin switchers...")
    k562_relaxed = run_relaxed_pass(
        day47, k562_origin_symbols, background_symbols, "K562-origin", OUTPUT_K562
    )

    print("\n" + "=" * 60)
    print("RELAXED-THRESHOLD PASS COMPLETE")
    print(f"GM12878-origin: {len(gm_relaxed)} terms at nominal P-value<{NOMINAL_P_THRESHOLD}")
    print(f"K562-origin:    {len(k562_relaxed)} terms at nominal P-value<{NOMINAL_P_THRESHOLD}")
    print("Next: compare against relaxed_pass_baseline_v1.md (Block 1's pre-registered top-5 "
          "table) and write the addendum in case_b_ratification_v1.md by hand -- do not "
          "auto-generate the interpretation.")
    print("=" * 60)


if __name__ == "__main__":
    main()