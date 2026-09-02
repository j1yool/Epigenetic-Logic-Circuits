"""
test_relaxed_switching_definition_v1.py

Day 51, Block 2 (Logic Circuits) — Epigenetic Logic Circuits repo.

Purpose
-------
Empirically test whether relaxing the strict AND<->origin switching-gene
definition grows the K562-origin foreground without collapsing the
AND<->INCONSISTENT distinction that motivates the origin split. This is
the pre-registered test named in gene_set_definition_decision_v1.md /
verdict_comparison_case_a_vs_case_b_v1.md.

Confirmed against real files (Day 51):
- run_pathway_enrichment_v1.py: load_background_and_origin_groups(symbol_map)
  takes symbol_map as an argument and returns a 5-tuple:
  (background_symbols, gm_origin_symbols, k562_origin_symbols, n_gm_ids, n_k562_ids).
  Both origin-symbol lists are ALREADY restricted to the strict AND<->INCONSISTENT
  subset -- this script reuses that call unmodified for the strict baseline.
- switching_genes_merged_v1.tsv real columns: gene_id, gate_type_gm12878,
  gate_type_k562, and_origin. Real gate-type values confirmed present:
  SIMPLE_AND, SIMPLE_OR, BIVALENT, REPRESSED, COMPLEX, INCONSISTENT, and the
  literal string "NULL" (no gate call in that cell line).
- extract_switching_genes_v2.py explicitly flags that pandas' default NA
  handling treats the literal string "NULL" as NaN on read -- this script
  uses keep_default_na=False / na_values=[""] to avoid that, matching the
  fix already applied in load_symbol_map() elsewhere in the codebase.
- and_origin in the raw file is NOT restricted to AND<->INCONSISTENT pairs --
  it labels whichever line holds SIMPLE_AND regardless of the other line's
  value (NEITHER if neither line is SIMPLE_AND). It is used here only as an
  informational cross-check column, not as the classification logic itself.

Convention compliance
----------------------
- Threading env vars set before any numeric import.
- synthetic_self_test() runs first; sys.exit(1) on any assertion failure.
- Schema validated against the real, confirmed column list; sys.exit(1)
  with the ACTUAL columns/values on any future drift.
- Cross-script import via importlib.util.spec_from_file_location.
- Reuses load_background_and_origin_groups() and map_ids_to_symbols() from
  the real pipeline module rather than reimplementing symbol mapping or
  background loading.
- One relaxed variant only (one-diagnostic-maximum). The relaxation is a
  single constant, RELAXED_DEST_STATES, below.

REQUIRED EDIT BEFORE RUNNING:
  CODE_DIR — path to the LOGIC CIRCUITS/code/ directory containing
  run_pathway_enrichment_v1.py, gate_assignments_regenerated_v1.tsv,
  k562_gate_assignments_regenerated_v1.tsv, and switching_genes_merged_v1.tsv
  (the pipeline module's own path constants are relative to this directory,
  per its module docstring -- this script chdir's into CODE_DIR so those
  relative paths resolve exactly as they do when the pipeline runs normally).

DEPENDENCY NOTE: run_pathway_enrichment_v1.py does `import gseapy as gp` at
module scope and sys.exit(1)s if it's missing. Importing that module here
(even though this script never calls gp.enrich()) will trigger that check.
Make sure `conda activate genomics` / gseapy is available before running.
"""

import os

# --- Threading env vars: MUST be set before any numeric import ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import importlib.util
from datetime import datetime

import pandas as pd  # noqa: E402  (numeric import, must come after env vars)


# ============================================================
# CONFIG — EDIT ME
# ============================================================

# EDIT ME: the LOGIC CIRCUITS/code/ directory (see docstring above for why)
CODE_DIR = r"C:\Users\jamoo\Downloads\LOGIC CIRCUITS\code"

PIPELINE_MODULE_PATH = os.path.join(CODE_DIR, "run_pathway_enrichment_v1.py")
OUTPUT_CSV = os.path.join(CODE_DIR, "relaxed_vs_strict_gene_set_comparison_v1.csv")

# Confirmed real schema (Day 51) — do not edit unless the file changes
GENE_ID_COL = "gene_id"
GM_COL = "gate_type_gm12878"
K562_COL = "gate_type_k562"
AND_ORIGIN_COL = "and_origin"
EXPECTED_COLUMNS = [GENE_ID_COL, GM_COL, K562_COL, AND_ORIGIN_COL]

# Confirmed real category values (Day 51), including the literal "NULL"
# no-call placeholder — NOT the same as NaN, see docstring.
VALID_GATE_TYPES = {
    "SIMPLE_AND",
    "SIMPLE_OR",
    "BIVALENT",
    "REPRESSED",
    "COMPLEX",
    "INCONSISTENT",
    "NULL",
}

ORIGIN_STATE = "SIMPLE_AND"

# Strict rule (matches load_background_and_origin_groups()'s own and_inc
# filter exactly): origin line = SIMPLE_AND, other line = exactly INCONSISTENT.
STRICT_DEST_STATES = {"INCONSISTENT"}

# EDIT ME after Block 1's Timmons et al. 2015 reading if a different single
# relaxation is warranted. Default tested here: also accept COMPLEX (not
# just INCONSISTENT) as the destination state.
RELAXED_DEST_STATES = {"INCONSISTENT", "COMPLEX"}

# Threshold for flagging asymmetric growth as worth manual scrutiny (see
# growth_asymmetry_ratio in main()). Not an auto-reject threshold -- a flag.
ASYMMETRY_FLAG_RATIO = 2.0


# ============================================================
# Core classification logic (the new code under test)
# ============================================================

def classify_origin_switchers(df: pd.DataFrame, dest_states: set) -> tuple[set, set]:
    """
    Return (gm12878_origin_gene_ids, k562_origin_gene_ids) under the given
    destination-state rule.

    gm12878_origin: GM12878 line = SIMPLE_AND, K562 line in dest_states
    k562_origin:    K562 line = SIMPLE_AND, GM12878 line in dest_states
    """
    gm_origin_mask = (df[GM_COL] == ORIGIN_STATE) & (df[K562_COL].isin(dest_states))
    k562_origin_mask = (df[K562_COL] == ORIGIN_STATE) & (df[GM_COL].isin(dest_states))

    gm_origin_ids = set(df.loc[gm_origin_mask, GENE_ID_COL])
    k562_origin_ids = set(df.loc[k562_origin_mask, GENE_ID_COL])

    return gm_origin_ids, k562_origin_ids


# ============================================================
# Synthetic self-test — MUST pass before any real-data run
# ============================================================

def synthetic_self_test() -> None:
    # NOTE on semantics (confirmed against real switching_genes_merged_v1.tsv,
    # row: SIMPLE_AND | INCONSISTENT | and_origin=GM12878): "GM12878-origin"
    # means GM12878 HOLDS the SIMPLE_AND call, regardless of what K562 shows.
    # "K562-origin" is the mirror: K562 holds SIMPLE_AND.
    synthetic_rows = [
        ("GENE_A", "SIMPLE_AND", "INCONSISTENT"),  # strict GM12878-origin: YES
        ("GENE_B", "INCONSISTENT", "SIMPLE_AND"),  # strict K562-origin: YES
        ("GENE_C", "SIMPLE_AND", "COMPLEX"),       # strict: NO, relaxed GM12878-origin: YES
        ("GENE_D", "COMPLEX", "SIMPLE_AND"),       # strict: NO, relaxed K562-origin: YES
        ("GENE_E", "SIMPLE_AND", "SIMPLE_AND"),    # not a switcher under either rule
        ("GENE_F", "SIMPLE_OR", "INCONSISTENT"),   # not AND-origin at all
        ("GENE_G", "SIMPLE_AND", "REPRESSED"),     # neither strict nor relaxed
        ("GENE_H", "SIMPLE_AND", "BIVALENT"),      # neither strict nor relaxed
        ("GENE_I", "NULL", "SIMPLE_AND"),          # literal "NULL" no-call: not origin, not dest
        ("GENE_J", "SIMPLE_AND", "NULL"),          # "NULL" is not a dest state under either rule
    ]
    df = pd.DataFrame(synthetic_rows, columns=[GENE_ID_COL, GM_COL, K562_COL])

    all_calls = set(df[GM_COL]) | set(df[K562_COL])
    assert all_calls.issubset(VALID_GATE_TYPES), (
        f"Synthetic test data uses invalid gate types: {all_calls - VALID_GATE_TYPES}"
    )

    strict_gm, strict_k562 = classify_origin_switchers(df, STRICT_DEST_STATES)
    relaxed_gm, relaxed_k562 = classify_origin_switchers(df, RELAXED_DEST_STATES)

    expected_strict_gm = {"GENE_A"}
    expected_strict_k562 = {"GENE_B"}
    expected_relaxed_gm = {"GENE_A", "GENE_C"}
    expected_relaxed_k562 = {"GENE_B", "GENE_D"}

    assert strict_gm == expected_strict_gm, (
        f"Strict GM12878-origin mismatch: got {strict_gm}, expected {expected_strict_gm}"
    )
    assert strict_k562 == expected_strict_k562, (
        f"Strict K562-origin mismatch: got {strict_k562}, expected {expected_strict_k562}"
    )
    assert relaxed_gm == expected_relaxed_gm, (
        f"Relaxed GM12878-origin mismatch: got {relaxed_gm}, expected {expected_relaxed_gm}"
    )
    assert relaxed_k562 == expected_relaxed_k562, (
        f"Relaxed K562-origin mismatch: got {relaxed_k562}, expected {expected_relaxed_k562}"
    )

    null_genes = {"GENE_I", "GENE_J"}
    all_classified = strict_gm | strict_k562 | relaxed_gm | relaxed_k562
    assert null_genes.isdisjoint(all_classified), (
        f"Literal 'NULL' rows were incorrectly classified: "
        f"{null_genes & all_classified}. Check keep_default_na handling."
    )

    assert strict_k562.issubset(relaxed_k562), (
        "Relaxed K562-origin set is not a superset of strict K562-origin set."
    )
    assert strict_gm.issubset(relaxed_gm), (
        "Relaxed GM12878-origin set is not a superset of strict GM12878-origin set."
    )

    # Structural property, confirmed by construction: gm_origin and k562_origin
    # are ALWAYS disjoint (a gene can't have GM==AND while also being in
    # dest_states, since AND is never itself a dest state). This is why a
    # "does relaxation make a gene double-count across origins" check would
    # be vacuous — it can never fire — and is not attempted here. See main()
    # for the actual (asymmetric-growth) diagnostic used instead.
    assert strict_gm.isdisjoint(strict_k562), "gm/k562 origin sets must be disjoint by construction"
    assert relaxed_gm.isdisjoint(relaxed_k562), "gm/k562 origin sets must be disjoint by construction"

    print("[synthetic_self_test] PASSED — classification logic, NULL handling, "
          "superset property, and origin-disjointness all confirmed on synthetic data.")


# ============================================================
# Real-data run
# ============================================================

def load_pipeline_module():
    if not os.path.exists(PIPELINE_MODULE_PATH):
        sys.exit(f"[FATAL] PIPELINE_MODULE_PATH does not exist: {PIPELINE_MODULE_PATH}")

    spec = importlib.util.spec_from_file_location("pipeline_module", PIPELINE_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for fn_name in ("load_symbol_map", "load_background_and_origin_groups", "map_ids_to_symbols"):
        if not hasattr(module, fn_name):
            sys.exit(
                f"[FATAL] {fn_name}() not found in {PIPELINE_MODULE_PATH}.\n"
                f"Functions actually available: "
                f"{[n for n in dir(module) if not n.startswith('_')]}\n"
                "Do not proceed by guessing a replacement — the pipeline script "
                "has changed since Day 51's confirmation; re-diagnose."
            )
    return module


def load_and_validate_gate_calls(switchers_path) -> pd.DataFrame:
    if not os.path.exists(switchers_path):
        sys.exit(f"[FATAL] switching_genes_merged_v1.tsv not found at: {switchers_path}")

    # keep_default_na=False / na_values=[""]: the literal string "NULL" in
    # this file is a real category (no gate call), not missing data.
    # Reading it with pandas defaults would silently convert "NULL" to NaN.
    df = pd.read_csv(switchers_path, sep="\t", keep_default_na=False, na_values=[""])

    actual_columns = list(df.columns)
    missing = [c for c in EXPECTED_COLUMNS if c not in actual_columns]
    if missing:
        sys.exit(
            f"[FATAL] Schema mismatch in {switchers_path}.\n"
            f"Expected columns: {EXPECTED_COLUMNS}\n"
            f"Missing: {missing}\n"
            f"ACTUAL columns in file: {actual_columns}\n"
            "This file's schema was confirmed on Day 51 — if it's changed, "
            "diagnose why before updating this script's constants."
        )

    observed_gate_types = set(df[GM_COL]) | set(df[K562_COL])
    unexpected = observed_gate_types - VALID_GATE_TYPES
    if unexpected:
        sys.exit(
            f"[FATAL] Unexpected gate-type values found: {unexpected}\n"
            f"Valid categories (confirmed Day 51): {VALID_GATE_TYPES}\n"
            "This indicates schema drift or a new gate-calling category this "
            "script doesn't know about. Diagnose before proceeding."
        )

    return df


def main():
    print("=" * 70)
    print(f"test_relaxed_switching_definition_v1.py — run at {datetime.now().isoformat()}")
    print("=" * 70)

    synthetic_self_test()

    if not os.path.isdir(CODE_DIR):
        sys.exit(f"[FATAL] CODE_DIR does not exist: {CODE_DIR}. Edit CODE_DIR in CONFIG.")

    original_cwd = os.getcwd()
    try:
        # The pipeline module's GM_REGEN / K562_REGEN / SWITCHERS_MERGED path
        # constants are relative to the code directory (per its own
        # docstring) — chdir here so calling its functions unmodified
        # resolves those paths exactly as they do in normal pipeline runs.
        os.chdir(CODE_DIR)

        pipeline_module = load_pipeline_module()

        try:
            symbol_map = pipeline_module.load_symbol_map()
        except SystemExit:
            raise
        except Exception as e:
            sys.exit(f"[FATAL] load_symbol_map() raised an exception: {e}")

        try:
            (background_symbols, gm_origin_symbols, k562_origin_symbols,
             n_gm_ids, n_k562_ids) = pipeline_module.load_background_and_origin_groups(symbol_map)
        except SystemExit:
            raise
        except Exception as e:
            sys.exit(
                f"[FATAL] load_background_and_origin_groups() raised an exception: {e}\n"
                "Fix the underlying function/inputs before re-running this comparison."
            )

        background_symbols_set = set(background_symbols)
        strict_gm_symbols = set(gm_origin_symbols)
        strict_k562_symbols = set(k562_origin_symbols)

        # Load raw table directly (same relative path the pipeline module
        # itself uses) for the strict-vs-relaxed ID-level comparison.
        switchers_path = str(pipeline_module.SWITCHERS_MERGED)
        gate_df = load_and_validate_gate_calls(switchers_path)

    finally:
        os.chdir(original_cwd)

    # Recompute strict IDs from raw table as a consistency check against
    # the pipeline's own n_gm_ids / n_k562_ids counts.
    strict_gm_ids_recomputed, strict_k562_ids_recomputed = classify_origin_switchers(
        gate_df, STRICT_DEST_STATES
    )
    if len(strict_k562_ids_recomputed) != n_k562_ids or len(strict_gm_ids_recomputed) != n_gm_ids:
        print(
            "[WARNING] Recomputed strict ID counts do NOT match "
            "load_background_and_origin_groups()'s reported counts.\n"
            f"  K562-origin: recomputed={len(strict_k562_ids_recomputed)} vs "
            f"pipeline n_k562_ids={n_k562_ids}\n"
            f"  GM12878-origin: recomputed={len(strict_gm_ids_recomputed)} vs "
            f"pipeline n_gm_ids={n_gm_ids}\n"
            "  This means this script's rule differs from the pipeline's actual "
            "rule, or the file has changed since the pipeline last ran. Resolve "
            "this discrepancy before trusting the relaxed comparison below."
        )

    # Relaxed IDs, then map to symbols using the pipeline's own mapper.
    relaxed_gm_ids, relaxed_k562_ids = classify_origin_switchers(gate_df, RELAXED_DEST_STATES)

    relaxed_gm_symbols = set(pipeline_module.map_ids_to_symbols(
        list(relaxed_gm_ids), symbol_map, "Relaxed GM12878-origin (Day 51 test)"
    ))
    relaxed_k562_symbols = set(pipeline_module.map_ids_to_symbols(
        list(relaxed_k562_ids), symbol_map, "Relaxed K562-origin (Day 51 test)"
    ))

    # Restrict to mapped background symbols (matches the "n=170 mapped" vs
    # "n=188 raw" distinction from the Day 49 doc).
    strict_k562_mapped = strict_k562_symbols & background_symbols_set
    relaxed_k562_mapped = relaxed_k562_symbols & background_symbols_set
    strict_gm_mapped = strict_gm_symbols & background_symbols_set
    relaxed_gm_mapped = relaxed_gm_symbols & background_symbols_set

    # Asymmetric-growth diagnostic (replaces an earlier "collapse" criterion
    # that was structurally vacuous -- gm_origin and k562_origin sets are
    # ALWAYS disjoint by construction, see synthetic_self_test, so a
    # cross-origin overlap check can never fire and can't distinguish
    # anything). The actual risk named in verdict_comparison_case_a_vs_case_b_v1.md
    # is that relaxing to include COMPLEX could disproportionately rescue
    # the underpowered K562-origin side specifically, rather than reflecting
    # a symmetric, biologically real broadening of "lost simple AND logic."
    # This is checkable: compare relaxed/strict growth ratios between the
    # two origins directly.
    k562_growth_ratio = (
        len(relaxed_k562_mapped) / len(strict_k562_mapped)
        if len(strict_k562_mapped) > 0 else float("nan")
    )
    gm_growth_ratio = (
        len(relaxed_gm_mapped) / len(strict_gm_mapped)
        if len(strict_gm_mapped) > 0 else float("nan")
    )
    growth_asymmetry_ratio = (
        k562_growth_ratio / gm_growth_ratio
        if (not pd.isna(gm_growth_ratio) and gm_growth_ratio > 0) else float("nan")
    )

    results = {
        "strict_k562_origin_mapped_n": len(strict_k562_mapped),
        "relaxed_k562_origin_mapped_n": len(relaxed_k562_mapped),
        "k562_growth_ratio": k562_growth_ratio,
        "strict_gm12878_origin_mapped_n": len(strict_gm_mapped),
        "relaxed_gm12878_origin_mapped_n": len(relaxed_gm_mapped),
        "gm12878_growth_ratio": gm_growth_ratio,
        "growth_asymmetry_ratio_k562_over_gm12878": growth_asymmetry_ratio,
        "flag": (
            "ASYMMETRIC GROWTH — K562-origin foreground grows disproportionately "
            "more than GM12878-origin under relaxation; possible definitional "
            "artifact specifically rescuing the underpowered side, not a "
            "symmetric broadening. Requires manual judgment, not auto-adopt."
            if (not pd.isna(growth_asymmetry_ratio) and growth_asymmetry_ratio > ASYMMETRY_FLAG_RATIO)
            else "GROWTH ROUGHLY SYMMETRIC — relaxation affects both origins "
                 "proportionally; less suggestive of a K562-specific artifact."
            if not pd.isna(growth_asymmetry_ratio)
            else "INDETERMINATE — one or both strict sets are empty, ratio undefined."
        ),
    }

    print("\n" + "=" * 70)
    print("COMPARISON TABLE (mapped = intersected with background symbol set)")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("=" * 70)

    pd.DataFrame([results]).to_csv(OUTPUT_CSV, index=False)
    print(f"\nComparison table written to: {OUTPUT_CSV}")
    print(
        "\nThis script reports diagnostics only -- it does not auto-decide "
        "ADOPT/REJECT. Write the unhedged verdict yourself in "
        "gene_set_definition_decision_v1.md, citing the actual "
        "k562_growth_ratio, gm12878_growth_ratio, and growth_asymmetry_ratio "
        "values above, informed by Block 1's Timmons et al. 2015 reading."
    )


if __name__ == "__main__":
    main()