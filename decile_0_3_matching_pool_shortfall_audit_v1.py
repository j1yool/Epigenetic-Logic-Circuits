"""
decile_0_3_matching_pool_shortfall_audit_v1.py

Day 65, Block 6 (LC track).

Purpose
-------
MASTER_STATUS.md documents a known, previously-flagged limitation: the
switching-gene matching pool is short in deciles 0-3 (decile 0 short by 275,
decile 1 by 228, decile 2 by 219, decile 3 by 105), and states this "should
be resolved before follow-up #2" -- the reconciliation rerun of the
deciles-0-3 pool-concentration comparison, originally scheduled for today.

This script does NOT fix anything. It:
  1. Reproduces the documented shortfall figures exactly, to confirm we're
     working from the same baseline before touching anything.
  2. Traces WHERE the shortfall originates -- specifically, whether it's a
     genuine supply constraint (too few stable/nonswitching genes exist in
     deciles 0-3 relative to switching genes needing a match) or an
     artifact of how match_by_decile() draws its pool (e.g. sampling
     without replacement against an exhausted pool, a silent dedup, or a
     decile-boundary edge case).

Working assumption, to be confirmed against match_by_decile()'s actual body
--------------------------------------------------------------------------
This script assumes 1:1 decile-matched sampling: for each decile d, every
switching gene in d needs one stable-gene match drawn from the stable pool
in the SAME decile d, without replacement. Under that assumption, shortfall
in decile d = max(0, n_switching_in_d - n_stable_in_d).

THIS IS AN ASSUMPTION, NOT A CONFIRMED FACT. match_by_decile()'s real
matching ratio and replacement policy have not been read yet (only
assign_deciles() and match_by_decile() are imported below as black boxes).
If the reproduction check in Block 6b fails to match the documented
275/228/219/105 figures, that is itself informative: it means the true
matching logic differs from the 1:1-no-replacement assumption, and
match_by_decile()'s actual source must be read before any further
conclusion is drawn. Per project convention: one diagnostic pass maximum,
report and stop rather than iterate blindly.

Hard requirements per project conventions
------------------------------------------
- Threading env vars set before any numeric imports.
- Cross-script imports via importlib.util.spec_from_file_location only.
- Schema confirmed against real files before data-handling code runs.
- Synthetic self-test with hand-derivable ground truth, must pass first.
- sys.exit(1) on self-test failure; sys.exit() with real info on schema
  or reproduction mismatch. No silent coercion.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import glob
import importlib.util

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG — confirmed sources from Day 65 Block 2 (gate files, switching sets)
# plus matched_nonswitching_v1.py's decile interface (from LC MASTER_STATUS.md
# "Key Files" list: assign_deciles(), match_by_decile(), _load_gate_module())
# =============================================================================

PROJECT_ROOT = r"C:\Users\jamoo\Downloads\LOGIC CIRCUITS"

MATCHED_NONSWITCHING_GLOB = os.path.join(PROJECT_ROOT, "**", "matched_nonswitching_v1.py")
SWITCHING_VARIABILITY_GLOB = os.path.join(
    PROJECT_ROOT, "**", "switching_gene_expression_variability_v1.py"
)

GM12878_GATE_FILE_GLOB = os.path.join(PROJECT_ROOT, "**", "gate_assignments_named.tsv")
K562_GATE_FILE_GLOB = os.path.join(PROJECT_ROOT, "**", "k562_gate_assignments_named.tsv")
GATE_COL = "gate_type"
GENE_ID_COL = "gene_id"

ASSIGN_DECILES_FUNC_NAME = "assign_deciles"
SWITCHING_LABEL_FUNC_NAME = "build_switching_sets"

K562_REPLICATES_GLOB = os.path.join(PROJECT_ROOT, "**", "encode_rnaseq_k562_replicates_v1.csv")
REPLICATE_COLS = ["rep1", "rep2"]  # confirmed working in Block 2's real run

# Documented shortfall figures to reproduce (MASTER_STATUS.md, LC, "Known
# limitation carried forward"). Deciles are 0-indexed per that doc.
DOCUMENTED_SHORTFALL = {0: 275, 1: 228, 2: 219, 3: 105}


# =============================================================================
# Shared helpers (same pattern as Block 2's script)
# =============================================================================

def _discover_file(glob_pattern, label):
    matches = sorted(glob.glob(glob_pattern, recursive=True))
    if len(matches) == 0:
        sys.exit(
            f"[SCHEMA HALT] No files matched for {label} using pattern:\n"
            f"  {glob_pattern}\nFix the glob pattern and rerun."
        )
    if len(matches) > 1:
        sys.exit(
            f"[SCHEMA HALT] Multiple files matched for {label}:\n  "
            + "\n  ".join(matches)
            + "\nNarrow the glob pattern and rerun."
        )
    return matches[0]


def _load_module(filepath, module_name):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_function_or_halt(module, func_name, module_path):
    if func_name not in dir(module) or not callable(getattr(module, func_name, None)):
        import inspect

        available = [
            name
            for name, obj in inspect.getmembers(module)
            if inspect.isfunction(obj) and obj.__module__ == module.__name__
        ]
        sys.exit(
            f"[SCHEMA HALT] Function '{func_name}' not found in {module_path}.\n"
            f"Functions actually defined:\n  "
            + "\n  ".join(available if available else ["(none found)"])
            + f"\nUpdate this script's config and rerun."
        )
    return getattr(module, func_name)


def _check_columns_or_halt(df, required_cols, filepath):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        sys.exit(
            f"[SCHEMA HALT] Missing column(s) {missing} in {filepath}.\n"
            f"Actual columns:\n  " + "\n  ".join(list(df.columns))
        )


# =============================================================================
# Self-test
# =============================================================================

def run_self_test():
    """
    Hand-constructed fixture: 4 deciles (0-3), known switching/stable counts
    per decile, known shortfall under the 1:1-no-replacement assumption.

    Fixture:
      decile 0: 10 switching, 6 stable  -> shortfall 4
      decile 1: 8 switching,  8 stable  -> shortfall 0
      decile 2: 5 switching,  9 stable  -> shortfall 0 (surplus, not deficit)
      decile 3: 12 switching, 3 stable  -> shortfall 9

    Expected shortfall dict: {0: 4, 1: 0, 2: 0, 3: 9}
    """
    rows = []
    decile_counts = {
        0: (10, 6),
        1: (8, 8),
        2: (5, 9),
        3: (12, 3),
    }
    gid = 0
    for decile, (n_switch, n_stable) in decile_counts.items():
        for _ in range(n_switch):
            rows.append({"gene_id": f"G{gid}", "decile": decile, "is_switching": True})
            gid += 1
        for _ in range(n_stable):
            rows.append({"gene_id": f"G{gid}", "decile": decile, "is_switching": False})
            gid += 1
    fixture = pd.DataFrame(rows)

    computed = compute_shortfall_by_decile(fixture, deciles=[0, 1, 2, 3])
    expected = {0: 4, 1: 0, 2: 0, 3: 9}

    mismatches = {k: (expected[k], computed[k]) for k in expected if expected[k] != computed[k]}
    if mismatches:
        print("[SELF-TEST FAILED]")
        for k, (exp, act) in mismatches.items():
            print(f"  decile {k}: expected shortfall {exp}, got {act}")
        sys.exit(1)

    print("[SELF-TEST PASSED]")
    print(f"  computed shortfall: {computed}")


def compute_shortfall_by_decile(df, deciles):
    """
    df: columns [gene_id, decile, is_switching]
    Returns {decile: max(0, n_switching - n_stable)} per the 1:1-no-replacement
    working assumption stated in the module docstring.
    """
    out = {}
    for d in deciles:
        sub = df[df["decile"] == d]
        n_switch = int(sub["is_switching"].sum())
        n_stable = int((~sub["is_switching"]).sum())
        out[d] = max(0, n_switch - n_stable)
    return out


# =============================================================================
# Real-data run
# =============================================================================

def run_real_data():
    # --- switching/stable labels (same path as Block 2) ---
    switching_script_path = _discover_file(
        SWITCHING_VARIABILITY_GLOB, "switching_gene_expression_variability_v1.py"
    )
    switching_module = _load_module(
        switching_script_path, "switching_gene_expression_variability_v1"
    )
    build_switching_sets = _get_function_or_halt(
        switching_module, SWITCHING_LABEL_FUNC_NAME, switching_script_path
    )

    gm_gate_path = _discover_file(GM12878_GATE_FILE_GLOB, "GM12878 gate assignments")
    k562_gate_path = _discover_file(K562_GATE_FILE_GLOB, "K562 gate assignments")
    gm_gate_df = pd.read_csv(gm_gate_path, sep="\t")
    k562_gate_df = pd.read_csv(k562_gate_path, sep="\t")
    _check_columns_or_halt(gm_gate_df, [GENE_ID_COL, GATE_COL], gm_gate_path)
    _check_columns_or_halt(k562_gate_df, [GENE_ID_COL, GATE_COL], k562_gate_path)

    switching_df, stable_df, _inconsistent_df = build_switching_sets(gm_gate_df, k562_gate_df)
    switching_ids = set(switching_df[GENE_ID_COL])
    stable_ids = set(stable_df[GENE_ID_COL])

    # --- decile assignment ---
    # Confirmed signature (inspect_assign_deciles_signature_v1.py output):
    #   assign_deciles(mean_tpm: pd.Series, edges: np.ndarray = None)
    #     -> (decile_labels: np.ndarray, edges: np.ndarray)
    # mean_tpm is a plain Series of TPM values; assign_deciles() does not
    # take or return gene_id -- decile_labels is positional, aligned to
    # mean_tpm's original order/index via np.log1p(mean_tpm.to_numpy()).
    # So gene_id alignment has to be handled by us: build mean_tpm indexed
    # by gene_id, then pair decile_labels back up by position.
    matched_script_path = _discover_file(
        MATCHED_NONSWITCHING_GLOB, "matched_nonswitching_v1.py"
    )
    matched_module = _load_module(matched_script_path, "matched_nonswitching_v1")
    assign_deciles = _get_function_or_halt(
        matched_module, ASSIGN_DECILES_FUNC_NAME, matched_script_path
    )

    k562_rep_path = _discover_file(K562_REPLICATES_GLOB, "K562 replicate expression")
    rep_df = pd.read_csv(k562_rep_path)
    _check_columns_or_halt(rep_df, [GENE_ID_COL] + REPLICATE_COLS, k562_rep_path)
    rep_df = rep_df.drop_duplicates(subset=GENE_ID_COL).reset_index(drop=True)
    mean_tpm = rep_df[REPLICATE_COLS].mean(axis=1)
    mean_tpm.index = rep_df[GENE_ID_COL].values  # index by gene_id, values still positional

    decile_labels, edges = assign_deciles(mean_tpm)
    if len(decile_labels) != len(rep_df):
        sys.exit(
            f"[SCHEMA HALT] assign_deciles() returned {len(decile_labels)} labels "
            f"for {len(rep_df)} input genes -- length mismatch, cannot align by "
            f"position. Inspect assign_deciles()'s return before proceeding."
        )

    decile_assignments = pd.DataFrame(
        {GENE_ID_COL: rep_df[GENE_ID_COL].values, "decile": decile_labels}
    )

    labeled_ids = switching_ids | stable_ids
    df = decile_assignments[decile_assignments[GENE_ID_COL].isin(labeled_ids)].copy()
    df["is_switching"] = df[GENE_ID_COL].isin(switching_ids)

    if len(df) == 0:
        sys.exit(
            "[DATA HALT] Zero genes overlap between assign_deciles() output and "
            "the labeled switching/stable set -- check gene_id format compatibility "
            "before proceeding."
        )

    deciles_present = sorted(df["decile"].unique())
    target_deciles = [d for d in [0, 1, 2, 3] if d in deciles_present]
    if len(target_deciles) < 4:
        print(
            f"[WARNING] Only deciles {target_deciles} present in decile column "
            f"among {deciles_present}; expected 0-3 to exist. Proceeding with "
            f"what's available -- reproduction check below will likely fail if "
            f"decile numbering/count doesn't match the documented figures."
        )

    computed_shortfall = compute_shortfall_by_decile(df, deciles=target_deciles)

    print("[REAL DATA — DECILE 0-3 SUPPLY/DEMAND]")
    for d in target_deciles:
        sub = df[df["decile"] == d]
        n_switch = int(sub["is_switching"].sum())
        n_stable = int((~sub["is_switching"]).sum())
        print(
            f"  decile {d}: n_switching={n_switch}, n_stable={n_stable}, "
            f"computed_shortfall={computed_shortfall[d]}, "
            f"documented_shortfall={DOCUMENTED_SHORTFALL.get(d, 'N/A')}"
        )

    # --- reproduction check against documented figures ---
    mismatches = {}
    for d, documented in DOCUMENTED_SHORTFALL.items():
        computed = computed_shortfall.get(d)
        if computed is None or computed != documented:
            mismatches[d] = (documented, computed)

    if mismatches:
        print("\n[REPRODUCTION CHECK: FAILED]")
        for d, (doc, comp) in mismatches.items():
            print(f"  decile {d}: documented={doc}, computed under 1:1 assumption={comp}")
        print(
            "\nThe 1:1-no-replacement matching assumption does NOT reproduce the "
            "documented shortfall figures. This means match_by_decile()'s real "
            "matching ratio or replacement policy differs from what this script "
            "assumed. DO NOT proceed to the reconciliation rerun on the strength "
            "of this script's shortfall model. Next step: read match_by_decile()'s "
            "actual body (matched_nonswitching_v1.py) at the function definition "
            "before drawing any conclusion about shortfall cause."
        )
        sys.exit(1)

    print("\n[REPRODUCTION CHECK: PASSED]")
    print(
        "1:1-no-replacement supply/demand exactly reproduces the documented "
        "shortfall. This confirms the shortfall is a genuine stable-gene supply "
        "constraint in deciles 0-3 (too few stable genes relative to switching "
        "genes needing a match in those deciles) -- not a bug in the matching "
        "code's sampling logic. Root cause is therefore data composition, not "
        "an implementation defect."
    )

    out_path = "decile_0_3_matching_pool_shortfall_audit_v1.csv"
    df[[GENE_ID_COL, "decile", "is_switching"]].to_csv(out_path, index=False)
    print(f"\noutput written: {out_path}")


if __name__ == "__main__":
    run_self_test()
    run_real_data()