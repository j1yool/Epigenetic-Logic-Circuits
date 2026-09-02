"""
trace_call_gate_dependency.py

Traces every point where the `expressed` label enters call_gate() across
the Logic Circuits codebase. Writes structured output to
call_gate_dependency_trace.txt.

Usage (run from project root, e.g. C:\\Users\\jamoo\\Downloads\\LOGIC CIRCUITS\\):
    conda activate genomics
    python trace_call_gate_dependency.py

Edit CANDIDATE_FILES below if call_gate() lives somewhere not listed.
"""

import ast
import os
import sys
from datetime import datetime

# ---- Configuration: edit paths if your repo layout differs ----
CANDIDATE_FILES = [
    "call_gate.py",
    "gate_calling.py",
    "logic_gates.py",
    "run_pipeline.py",
    "layer_feature_extraction.py",
    "layer_feature_extraction2.py",
]
SEARCH_ROOT = "."  # run this from the epigenetic-logic-circuits repo root
LABEL_VAR_NAMES = {"expressed", "is_expressed", "expression_label", "expr_label"}
OUTPUT_FILE = "call_gate_dependency_trace.txt"
# -----------------------------------------------------------------


def find_files(root, candidates):
    found = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn in candidates:
                found.append(os.path.join(dirpath, fn))
    return found


def get_function_source_lines(filepath, func_name="call_gate"):
    """Return (start_line, end_line, source_lines) for the named function, or None."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        source = f.read()
    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as e:
        return ("PARSE_ERROR", str(e), None)

    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            start = node.lineno
            end = getattr(node, "end_lineno", None)
            if end is None:
                # Fallback for older Python: walk to find max line in subtree
                end = max(
                    getattr(n, "lineno", start) for n in ast.walk(node)
                )
            return (start, end, lines[start - 1:end])
    return None


def trace_label_usage(func_lines, start_line):
    """Find every line inside the function body referencing a label variable."""
    hits = []
    for i, line in enumerate(func_lines):
        actual_line_no = start_line + i
        for var in LABEL_VAR_NAMES:
            if var in line:
                hits.append((actual_line_no, var, line.strip()))
    return hits


def classify_branch_context(func_lines, start_line, hit_line_no):
    """
    Naive branch-context extraction: walk backward from the hit to find the
    nearest enclosing if/elif/else header, to help identify which gate-type
    branch depends on the label.
    """
    idx = hit_line_no - start_line
    for j in range(idx, -1, -1):
        stripped = func_lines[j].strip()
        if stripped.startswith(("if ", "elif ", "else:", "else :")):
            return f"line {start_line + j}: {stripped}"
    return "(no enclosing if/elif found — possibly top-level or inside a different construct)"


def main():
    output_lines = []
    output_lines.append(f"call_gate() dependency trace")
    output_lines.append(f"Generated: {datetime.now().isoformat()}")
    output_lines.append(f"Label variable names searched: {sorted(LABEL_VAR_NAMES)}")
    output_lines.append("=" * 70)
    output_lines.append("")

    files_found = find_files(SEARCH_ROOT, CANDIDATE_FILES)

    if not files_found:
        output_lines.append(
            "No candidate files found under SEARCH_ROOT. "
            "Edit CANDIDATE_FILES or SEARCH_ROOT in this script and re-run."
        )
        write_output(output_lines)
        return

    any_function_found = False

    for filepath in sorted(files_found):
        output_lines.append(f"FILE: {filepath}")
        output_lines.append("-" * 70)

        result = get_function_source_lines(filepath, "call_gate")

        if result is None:
            output_lines.append("  call_gate() not defined in this file.")
            output_lines.append("")
            continue

        if result[0] == "PARSE_ERROR":
            output_lines.append(f"  SYNTAX ERROR while parsing file: {result[1]}")
            output_lines.append("")
            continue

        any_function_found = True
        start_line, end_line, func_lines = result
        output_lines.append(f"  call_gate() found: lines {start_line}-{end_line}")

        hits = trace_label_usage(func_lines, start_line)

        if not hits:
            output_lines.append("  No label-variable references found inside call_gate().")
            output_lines.append("  -> This file's call_gate() may be label-independent, "
                                 "or it delegates to a helper defined elsewhere (check imports).")
        else:
            output_lines.append(f"  {len(hits)} label reference(s) found:")
            for line_no, var, code in hits:
                branch = classify_branch_context(func_lines, start_line, line_no)
                output_lines.append(f"    Line {line_no} [{var}]: {code}")
                output_lines.append(f"        enclosing branch -> {branch}")

        output_lines.append("")

    if not any_function_found:
        output_lines.append(
            "call_gate() was not found in ANY candidate file. "
            "Confirm the function name/location and update CANDIDATE_FILES."
        )

    output_lines.append("=" * 70)
    output_lines.append("MANUAL FOLLOW-UP REQUIRED (not automatable):")
    output_lines.append(
        "For each branch flagged above, determine by reading context: "
        "does this branch assign a specific gate type (e.g. XOR, complex) "
        "that is UNREACHABLE without the label, or does the label just "
        "resolve an ambiguous case that could also be resolved from marks alone? "
        "That judgment call is Step 2 of Block 2 — this script only locates "
        "candidates, it doesn't make the fixable-vs-structural call."
    )

    write_output(output_lines)


def write_output(lines):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {OUTPUT_FILE} ({len(lines)} lines)")


if __name__ == "__main__":
    main()