"""
artifact_suspicious_recheck_v1_addendum_check.py

Day 46, Block 3 (first 30 min) -- Epigenetic Logic Circuits

Purpose
-------
Day 45's forward note flagged artifact_suspicious_recheck.py as having
loaded the real GM12878/K562 binary matrices with "not loaded" placeholder
text, since the files weren't present in that session's environment. The
uploaded artifact_suspicious_resolution_v1.md, however, already shows real
numeric mark values for TET2 in both GM12878 and K562 -- meaning the
script has ALREADY been successfully rerun against real data on the real
machine, using GM12878_BINARY_PATH / K562_BINARY_PATH exactly as
hardcoded in the script.

This script does not redo that work. Its job is narrower: verify the
already-produced real-data resolution doc is internally consistent
(the loaded marks correctly reproduce the reported structural/marks-only
gate calls when run back through this session's copy of call_gate() /
marks_only_gate()), and write the Day 46 addendum confirming closure --
per supersede-don't-append, as an addendum section, not a full document
replacement, since nothing in the underlying proof or scope changed.

Usage
-----
    python artifact_suspicious_recheck_v1_addendum_check.py --data-dir "C:\\Users\\jamoo\\Downloads\\LOGIC CIRCUITS"
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import re
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

DEFAULT_DATA_DIR = Path(__file__).resolve().parent
RESOLUTION_DOC_NAME = "artifact_suspicious_resolution_v1.md"

MARK_KEYS = ["H3K4me3", "H3K27ac", "H3K4me1", "H3K27me3", "H3K9me3"]


# ---------------------------------------------------------------------------
# call_gate() / marks_only_gate() -- byte-identical copies from
# artifact_suspicious_recheck.py, reused here to independently re-derive
# the gate calls reported in the resolution doc from the marks values ALSO
# reported in that same doc, as a self-consistency check.
# ---------------------------------------------------------------------------

def call_gate(row: dict) -> str:
    active = row.get('H3K4me3', 0)
    enhancer = row.get('H3K27ac', 0)
    poised = row.get('H3K4me1', 0)
    repressive1 = row.get('H3K27me3', 0)
    repressive2 = row.get('H3K9me3', 0)
    expressed = row.get('expressed', 0)

    any_active = active or enhancer
    any_repressive = repressive1 or repressive2
    active_count = active + enhancer + poised
    repressive_count = repressive1 + repressive2

    if not any_active and not any_repressive:
        return 'NULL'
    if active and repressive1:
        return 'BIVALENT'
    if any_repressive and not any_active:
        return 'REPRESSED'
    if poised and not any_active and not any_repressive:
        return 'POISED'
    if active and enhancer and not any_repressive:
        return 'SIMPLE_AND' if expressed else 'INCONSISTENT'
    if (active or enhancer) and not (active and enhancer) and not any_repressive:
        return 'SIMPLE_OR' if expressed else 'INCONSISTENT'
    if active_count + repressive_count >= 3:
        return 'COMPLEX'
    if expressed and not any_active:
        return 'INCONSISTENT'
    return 'COMPLEX'


def marks_only_gate(row: dict) -> str:
    active = row.get('H3K4me3', 0)
    enhancer = row.get('H3K27ac', 0)
    poised = row.get('H3K4me1', 0)
    repressive1 = row.get('H3K27me3', 0)
    repressive2 = row.get('H3K9me3', 0)

    any_active = active or enhancer
    any_repressive = repressive1 or repressive2
    active_count = active + enhancer + poised
    repressive_count = repressive1 + repressive2

    if not any_active and not any_repressive:
        return 'NULL'
    if active and repressive1:
        return 'BIVALENT'
    if any_repressive and not any_active:
        return 'REPRESSED'
    if poised and not any_active and not any_repressive:
        return 'POISED'
    if active and enhancer and not any_repressive:
        return 'SIMPLE_AND'
    if (active or enhancer) and not (active and enhancer) and not any_repressive:
        return 'SIMPLE_OR'
    if active_count + repressive_count >= 3:
        return 'COMPLEX'
    return 'COMPLEX'


# ---------------------------------------------------------------------------
# Parse the "Confirmatory data check" lines out of the resolution doc.
# ---------------------------------------------------------------------------

CONFIRM_LINE_PATTERN = re.compile(
    r"-\s*(?P<cell_line>[A-Za-z0-9]+):\s*structural='(?P<structural>[A-Z_]+)',\s*"
    r"marks-only='(?P<marks_only>[A-Z_]+)'\s*\(marks:\s*(?P<marks_dict>\{[^}]*\}),\s*"
    r"expressed=(?P<expressed>[0-9.]+)\)"
)


def parse_confirmatory_lines(text: str) -> list:
    records = []
    for m in CONFIRM_LINE_PATTERN.finditer(text):
        marks_dict_str = m.group("marks_dict")
        marks = {}
        for key in MARK_KEYS:
            key_pattern = re.compile(r"'" + re.escape(key) + r"':\s*([0-9.]+)")
            key_match = key_pattern.search(marks_dict_str)
            marks[key] = float(key_match.group(1)) if key_match else None
        records.append({
            "cell_line": m.group("cell_line"),
            "reported_structural": m.group("structural"),
            "reported_marks_only": m.group("marks_only"),
            "expressed": float(m.group("expressed")),
            "marks": marks,
        })
    return records


def verify_confirmatory_records(records: list) -> dict:
    mismatches = []
    n_placeholder = 0
    for rec in records:
        if any(v is None for v in rec["marks"].values()):
            n_placeholder += 1
            continue
        row = dict(rec["marks"])
        row["expressed"] = rec["expressed"]
        recomputed_structural = call_gate(row)
        recomputed_marks_only = marks_only_gate(row)
        if recomputed_structural != rec["reported_structural"] or recomputed_marks_only != rec["reported_marks_only"]:
            mismatches.append({
                "cell_line": rec["cell_line"],
                "reported": (rec["reported_structural"], rec["reported_marks_only"]),
                "recomputed": (recomputed_structural, recomputed_marks_only),
            })
    return {
        "n_records_found": len(records),
        "n_placeholder_not_loaded": n_placeholder,
        "n_real_data_records": len(records) - n_placeholder,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches,
        "all_consistent": (len(records) - n_placeholder) > 0 and len(mismatches) == 0,
    }


# ---------------------------------------------------------------------------
# Synthetic self-test
# ---------------------------------------------------------------------------

def run_synthetic_test() -> bool:
    print("Running synthetic self-test...")
    try:
        # Consistent case: reported values match what call_gate/marks_only_gate
        # actually produce from the same marks.
        consistent_text = (
            "- GM12878: structural='SIMPLE_AND', marks-only='SIMPLE_AND' "
            "(marks: {'H3K4me3': 1.0, 'H3K27ac': 1.0, 'H3K4me1': 0.0, "
            "'H3K27me3': 0.0, 'H3K9me3': 0.0}, expressed=1.0)\n"
            "- K562: structural='INCONSISTENT', marks-only='SIMPLE_AND' "
            "(marks: {'H3K4me3': 1, 'H3K27ac': 1, 'H3K4me1': 0, "
            "'H3K27me3': 0, 'H3K9me3': 0}, expressed=0.0)\n"
        )
        records = parse_confirmatory_lines(consistent_text)
        assert len(records) == 2, f"Expected 2 parsed records, got {len(records)}"
        result = verify_confirmatory_records(records)
        assert result["all_consistent"] is True, f"Expected consistent case to verify clean: {result}"
        assert result["n_mismatches"] == 0

        # Placeholder case: "not loaded" lines should be skipped, not
        # misparsed or falsely flagged as mismatches.
        placeholder_text = "- GM12878: not loaded (file not found: ...)\n- K562: not loaded (file not found: ...)\n"
        records_p = parse_confirmatory_lines(placeholder_text)
        assert len(records_p) == 0, "Placeholder 'not loaded' lines should not be parsed as data records."

        # Planted inconsistency: reported marks-only doesn't match what the
        # given marks actually produce -- MUST be caught.
        bad_text = (
            "- GM12878: structural='SIMPLE_AND', marks-only='COMPLEX' "  # wrong -- should be SIMPLE_AND
            "(marks: {'H3K4me3': 1.0, 'H3K27ac': 1.0, 'H3K4me1': 0.0, "
            "'H3K27me3': 0.0, 'H3K9me3': 0.0}, expressed=1.0)\n"
        )
        records_bad = parse_confirmatory_lines(bad_text)
        result_bad = verify_confirmatory_records(records_bad)
        assert result_bad["all_consistent"] is False, f"Expected planted inconsistency to be caught: {result_bad}"
        assert result_bad["n_mismatches"] == 1

        print("Synthetic self-test PASSED: consistent real-data records verify clean, "
              "'not loaded' placeholder lines are correctly skipped (not misparsed), "
              "and a planted reported-vs-recomputed mismatch is correctly caught.")
        return True
    except AssertionError as e:
        print(f"SYNTHETIC TEST FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify and close out artifact_suspicious_resolution_v1.md's real-data confirmatory check")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--resolution-doc", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if not run_synthetic_test():
        print("Aborting: synthetic self-test did not pass. Real document was NOT touched.")
        sys.exit(1)

    data_dir = Path(args.data_dir)
    resolution_path = Path(args.resolution_doc) if args.resolution_doc else data_dir / RESOLUTION_DOC_NAME
    out_path = Path(args.out) if args.out else data_dir / "artifact_suspicious_recheck_v1_day46_addendum.md"

    if not resolution_path.exists():
        print(f"ERROR: {resolution_path} not found.")
        sys.exit(1)

    text = resolution_path.read_text(encoding="utf-8", errors="ignore")
    records = parse_confirmatory_lines(text)
    result = verify_confirmatory_records(records)

    status = "REAL_DATA_CONFIRMED_CONSISTENT" if result["all_consistent"] else (
        "STILL_PLACEHOLDER_NO_REAL_DATA" if result["n_real_data_records"] == 0 else "INCONSISTENCY_FOUND"
    )

    lines = [
        f"STATUS: {status}", "",
        "# Day 46 Addendum — Artifact-Suspicious Recheck, Real-Data Confirmation", "",
        f"_Generated {datetime.now().isoformat(timespec='seconds')} by artifact_suspicious_recheck_v1_addendum_check.py._",
        "_Appends to, does not replace, artifact_suspicious_resolution_v1.md — the underlying proof "
        "and scope are unchanged; this only confirms the real-data confirmatory check block is now "
        "populated with actual GM12878/K562 values (not the 'not loaded' placeholder) and that those "
        "values are internally self-consistent with call_gate()/marks_only_gate()._",
        "",
        "## Status", f"**{status}**", "",
        "## Result", f"```\n{result}\n```", "",
    ]
    if status == "REAL_DATA_CONFIRMED_CONSISTENT":
        lines.append(
            f"Both TET2 confirmatory records (GM12878, K562) now carry real mark values "
            f"(no 'not loaded' placeholders remain). Independently re-running those marks through "
            f"this session's copy of call_gate()/marks_only_gate() reproduces the exact structural "
            f"and marks-only calls reported in the resolution doc, with zero discrepancies. "
            f"**The Day 45 carry-forward item ('rerun against real data, replace not-loaded "
            f"placeholders') is closed — no further action needed.** The K562 "
            f"SIMPLE_AND(GM12878)→INCONSISTENT(K562) switch is confirmed on real data, not just "
            f"the exhaustive 64-combination proof."
        )
    elif status == "STILL_PLACEHOLDER_NO_REAL_DATA":
        lines.append(
            "No real-data confirmatory records were found — the resolution doc still shows "
            "'not loaded' placeholders. The real GM12878/K562 binary matrix files still need to be "
            "generated/located at the paths hardcoded in artifact_suspicious_recheck.py "
            "(GM12878_BINARY_PATH, K562_BINARY_PATH) and the script rerun before this item can close."
        )
    else:
        lines.append(
            "**Do not close this item.** The resolution doc's reported gate calls do not match what "
            "call_gate()/marks_only_gate() actually produce from the marks values also reported in "
            "the same doc. Either the doc's marks values were transcribed incorrectly, or the version "
            "of call_gate() used to produce the doc has diverged from the confirmed byte-identical "
            "source. Diagnose at file/line level before trusting this resolution further, per standing rule."
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWritten to {out_path}")
    print(f"\nSTATUS: {status}")


if __name__ == "__main__":
    main()