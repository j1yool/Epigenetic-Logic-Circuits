"""
artifact_suspicious_recheck_v1.py

Day 45, Block 3 — Epigenetic Logic Circuits.

SCOPE (confirmed against real files this session):
switching_gene_pathway_context_v1.md contains one gene (TET2, verdict
"biologically plausible") and one explicit open item: the call_gate()
label-dependency caveat was not independently checked against TET2's
SIMPLE_AND (GM12878) -> INCONSISTENT (K562) call. There is no
artifact-suspicious list — that framing in the Day 44 forward note did
not match the document.

call_gate() ITSELF (CONFIRMED — verbatim from uploaded
Day_5_-_Gate-Calling_Algorithm and DAY_14_-_K562_Gate-Calling_Algorithm,
byte-identical in both files): see call_gate() below.

KEY FINDING, proven by exhaustive enumeration over all 64 binary
combinations of the 6 boolean inputs (H3K4me3, H3K27ac, H3K4me1,
H3K27me3, H3K9me3, expressed) — see run_synthetic_self_test():
  - call_gate() returns 'INCONSISTENT' in exactly 6 of 64 combinations.
  - In ALL 6 of those cases, the marks-only derivation below
    (marks_only_gate — call_gate with every `expressed`-conditional
    branch collapsed to its marks-implied label) returns SIMPLE_AND or
    SIMPLE_OR — never anything else.
  - Outside those 6 cases, call_gate() and marks_only_gate() agree with
    ZERO exceptions.
  - A third `expressed`-dependent branch in the original source
    (`if expressed and not any_active: return 'INCONSISTENT'`) is
    confirmed UNREACHABLE dead code: any row with `not any_active`
    already returns NULL or REPRESSED earlier in the function. Verified
    for all 64 combinations; never touched here, per project rule
    against fixing anything not directly in scope.

CONSEQUENCE: 'INCONSISTENT' can ONLY ever be produced by one of the two
live `expressed`-conditional branches. Therefore ANY gene whose
call_gate() output is 'INCONSISTENT' — including TET2's K562 call — is
PROVABLY explained by the documented structural label dependency
(call_gate_determinism_resolution_v1.md), independent of that gene's
specific mark values. This resolves the open item without needing the
real GM12878/K562 binary matrices, though this script still attempts to
load them (paths below are CONFIRMED, taken verbatim from the uploaded
scripts' own __main__ blocks) for an illustrative/confirmatory report on
TET2's actual mark values, non-blocking if the files aren't present in
this environment.

marks_only_gate() is a DERIVED function — not something that exists in
the uploaded scripts as a separate implementation. It has NOT been
verified against whatever script originally produced the locked
GM12878 0.7974 / K562 0.7531 marks-only AUC figures. If exact
reproduction of those numbers matters, rerun this derivation against the
full binary matrices and compare — the logical proof above holds
regardless, since it depends only on call_gate()'s confirmed structure.
"""

import os

# Windows threading fix — must precede any numeric imports.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import itertools
import re
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

SWITCHING_CONTEXT_PATH = DATA_DIR / "switching_gene_pathway_context_v1.md"
OUTPUT_PATH = DATA_DIR / ".." / "writing" / "artifact_suspicious_resolution_v1.md"

# CONFIRMED real paths — taken verbatim from the __main__ blocks of
# Day_5_-_Gate-Calling_Algorithm and DAY_14_-_K562_Gate-Calling_Algorithm.
GM12878_BINARY_PATH = Path(r"C:\Users\jamoo\Downloads\LOGIC CIRCUITS\data\binary_matrix.csv")
K562_BINARY_PATH = Path(r"C:\Users\jamoo\Downloads\LOGIC CIRCUITS\data\k562\k562_binary_matrix.csv")

MARKS_ONLY_AUC = {"GM12878": 0.7974, "K562": 0.7531}  # locked elsewhere, for report context only

TET2_ENSEMBL_ID = "ENSG00000168769"  # from switching_gene_pathway_context_v1.md


# ---------------------------------------------------------------------------
# call_gate() — CONFIRMED, verbatim from the two uploaded scripts.
# ---------------------------------------------------------------------------
def call_gate(row):
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
        if expressed:
            return 'SIMPLE_AND'
        else:
            return 'INCONSISTENT'
    if (active or enhancer) and not (active and enhancer) and not any_repressive:
        if expressed:
            return 'SIMPLE_OR'
        else:
            return 'INCONSISTENT'
    if active_count + repressive_count >= 3:
        return 'COMPLEX'
    if expressed and not any_active:
        return 'INCONSISTENT'  # confirmed unreachable — see module docstring
    return 'COMPLEX'


# ---------------------------------------------------------------------------
# marks_only_gate() — DERIVED from call_gate() by removing every
# `expressed`-conditional branch. See module docstring for verification.
# ---------------------------------------------------------------------------
def marks_only_gate(row):
    active = row.get('H3K4me3', 0)
    enhancer = row.get('H3K27ac', 0)
    poised = row.get('H3K4me1', 0)
    repressive1 = row.get('H3K27me3', 0)
    repressive2 = row.get('H3K9me3', 0)
    # `expressed` intentionally not read — this is the marks-only comparator.

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
        return 'SIMPLE_AND'  # was: if expressed ... else INCONSISTENT
    if (active or enhancer) and not (active and enhancer) and not any_repressive:
        return 'SIMPLE_OR'  # was: if expressed ... else INCONSISTENT
    if active_count + repressive_count >= 3:
        return 'COMPLEX'
    return 'COMPLEX'


# ---------------------------------------------------------------------------
# Self-test — EXHAUSTIVE, not sampled: only 64 combinations exist for 6
# binary inputs, so this proves the claim rather than merely suggesting it.
# ---------------------------------------------------------------------------
def run_synthetic_self_test() -> None:
    keys = ['H3K4me3', 'H3K27ac', 'H3K4me1', 'H3K27me3', 'H3K9me3', 'expressed']
    inconsistent_marks_only_values = set()
    mismatches_outside_inconsistent = []
    n_inconsistent = 0

    for combo in itertools.product([0, 1], repeat=6):
        row = dict(zip(keys, combo))
        g_struct = call_gate(row)
        g_marks = marks_only_gate(row)
        if g_struct == 'INCONSISTENT':
            n_inconsistent += 1
            inconsistent_marks_only_values.add(g_marks)
        elif g_struct != g_marks:
            mismatches_outside_inconsistent.append((row, g_struct, g_marks))

    assert n_inconsistent == 6, (
        f"SELF-TEST FAILED: expected exactly 6 of 64 binary combinations "
        f"to produce call_gate()=='INCONSISTENT', found {n_inconsistent}. "
        f"The embedded call_gate() source may have been altered from the "
        f"confirmed original — re-verify against the uploaded scripts "
        f"before trusting anything downstream."
    )
    assert inconsistent_marks_only_values == {"SIMPLE_AND", "SIMPLE_OR"}, (
        f"SELF-TEST FAILED: expected marks_only_gate() to return only "
        f"SIMPLE_AND/SIMPLE_OR for the 6 INCONSISTENT cases, got "
        f"{inconsistent_marks_only_values}. The core proof this script "
        f"relies on does not hold — do not trust the resolution report."
    )
    assert mismatches_outside_inconsistent == [], (
        f"SELF-TEST FAILED: found {len(mismatches_outside_inconsistent)} "
        f"case(s) where call_gate() and marks_only_gate() disagree "
        f"OUTSIDE of an INCONSISTENT call — the derivation is not a pure "
        f"expressed-branch removal as claimed. First mismatch: "
        f"{mismatches_outside_inconsistent[0]}"
    )

    print("[SELF-TEST PASSED] Exhaustive check over all 64 binary input "
          "combinations confirms:")
    print(f"  - call_gate() returns INCONSISTENT in exactly {n_inconsistent}/64 cases")
    print(f"  - marks_only_gate() for those cases is always in "
          f"{inconsistent_marks_only_values} (never anything else)")
    print(f"  - zero disagreements between the two functions outside "
          f"INCONSISTENT cases")
    print("  => any INCONSISTENT call is provably explained by the "
          "expressed-label dependency, independent of specific mark values.\n")


# ---------------------------------------------------------------------------
# Parse the Gene-level table from switching_gene_pathway_context_v1.md
# (unchanged from the previous corrected version — already verified
# against the real document this session).
# ---------------------------------------------------------------------------
def parse_gene_table(path: Path) -> list:
    if not path.exists():
        sys.exit(f"FATAL: {path} not found. Confirm SWITCHING_CONTEXT_PATH.")

    text = path.read_text(encoding="utf-8")
    header_pattern = re.compile(r"^#{1,4}\s*(.+)$", re.MULTILINE)
    headers = [(m.start(), m.group(1).strip()) for m in header_pattern.finditer(text)]

    target_idx = None
    for i, (_, title) in enumerate(headers):
        if "gene-level table" in title.lower():
            target_idx = i
            break

    if target_idx is None:
        sys.exit(
            f"FATAL: no header containing 'Gene-level table' found in "
            f"{path}. Headers present: {[h[1] for h in headers]}."
        )

    start = headers[target_idx][0]
    end = headers[target_idx + 1][0] if target_idx + 1 < len(headers) else len(text)
    section_text = text[start:end]

    table_rows = [
        line.strip() for line in section_text.splitlines()
        if line.strip().startswith("|") and line.strip().endswith("|")
    ]
    data_rows = [
        row for row in table_rows
        if not re.match(r"^\|[\s:|-]+\|$", row)
        and "gene" not in row.lower().split("|")[1].strip().lower()
    ]

    genes = []
    for row in data_rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        if len(cells) < 4:
            continue
        genes.append({
            "gene": cells[0], "direction": cells[1],
            "driver_status": cells[2], "verdict": cells[3],
        })

    if not genes:
        sys.exit(
            f"FATAL: 'Gene-level table' section found but zero rows "
            f"parsed. Section content:\n{section_text[:1000]}"
        )

    print(f"Parsed {len(genes)} gene(s): {[g['gene'] for g in genes]}\n")
    return genes


# ---------------------------------------------------------------------------
# Optional confirmatory data load — non-blocking if files are absent,
# since the core resolution below does not require them.
# ---------------------------------------------------------------------------
def try_load_gene_row(binary_path: Path, gene_symbol: str, ensembl_id: str):
    if not binary_path.exists():
        return None, f"file not found: {binary_path}"

    try:
        df = pd.read_csv(binary_path, index_col=0)
    except Exception as e:
        return None, f"could not read {binary_path}: {e}"

    if gene_symbol in df.index:
        return df.loc[gene_symbol], None
    if ensembl_id in df.index:
        return df.loc[ensembl_id], None
    return None, (
        f"neither '{gene_symbol}' nor '{ensembl_id}' found in "
        f"{binary_path.name} index (index appears to use a different "
        f"identifier scheme — first 3 index values: "
        f"{list(df.index[:3])})"
    )


# ---------------------------------------------------------------------------
# Resolution report
# ---------------------------------------------------------------------------
def run_resolution(gene_records: list) -> str:
    lines = ["# Artifact-Suspicious Gene Resolution — v1\n"]
    lines.append(
        "**Scope note:** no gene in switching_gene_pathway_context_v1.md "
        "currently carries an 'artifact-suspicious' verdict (n=1 gene, "
        "TET2, verdict='biologically plausible'). The real open item — "
        "per that document's own 'Open items carried forward' section — "
        "is verifying the call_gate() label-dependency caveat against "
        "TET2's specific SIMPLE_AND→INCONSISTENT call. This resolves "
        "that item directly, for TET2 and generically for any future "
        "gene in this table.\n"
    )
    lines.append(
        "**Method — proof, not simulation:** call_gate() (verbatim, "
        "Day 5 / Day 14 Gate-Calling Algorithm, byte-identical in both) "
        "was checked exhaustively over all 64 binary combinations of its "
        "6 boolean inputs. Result: 'INCONSISTENT' is produced in exactly "
        "6/64 cases, and in every one of those cases the marks-only "
        "derivation (call_gate with every `expressed`-conditional branch "
        "collapsed to its marks-implied label) returns SIMPLE_AND or "
        "SIMPLE_OR — never anything else. Outside INCONSISTENT cases the "
        "two functions agree with zero exceptions. A third "
        "`expressed`-dependent branch in the original source is confirmed "
        "unreachable dead code. **Consequence: any gene whose call_gate() "
        "output is 'INCONSISTENT' is provably explained by the "
        "documented structural label dependency "
        "(call_gate_determinism_resolution_v1.md), independent of that "
        "gene's specific mark values.**\n"
    )

    counts = {"resolved-by-known-cause": 0, "open-with-cause-still-unidentified": 0}

    for record in gene_records:
        gene = record["gene"]
        direction = record["direction"]
        lines.append(f"## {gene}\n")
        lines.append(
            f"Document verdict: **{record['verdict']}** | Reported "
            f"direction: {direction} | Driver status: "
            f"{record['driver_status']}\n"
        )

        sides = [s.strip() for s in re.split(r"→|->", direction)]
        resolved_sides = [s for s in sides if "INCONSISTENT" in s.upper()]

        if resolved_sides:
            lines.append(
                f"Reported direction contains 'INCONSISTENT' "
                f"({len(resolved_sides)} of {len(sides)} side(s)). By the "
                f"exhaustive proof above, this call is **provably "
                f"resolved-by-known-cause** — no other code path in "
                f"call_gate() can produce 'INCONSISTENT'. This "
                f"independently confirms the caveat flagged as unchecked "
                f"in switching_gene_pathway_context_v1.md's Open Items.\n"
            )
            counts["resolved-by-known-cause"] += 1
        else:
            lines.append(
                f"Reported direction does not contain 'INCONSISTENT' on "
                f"either side, so the exhaustive proof above does not "
                f"directly apply — this gene's call would need real mark "
                f"data to explain, and is not resolved by this method.\n"
            )
            counts["open-with-cause-still-unidentified"] += 1

        # Confirmatory, non-blocking: try to load and illustrate real marks.
        gm_row, gm_err = try_load_gene_row(GM12878_BINARY_PATH, gene, TET2_ENSEMBL_ID if gene == "TET2" else gene)
        k562_row, k562_err = try_load_gene_row(K562_BINARY_PATH, gene, TET2_ENSEMBL_ID if gene == "TET2" else gene)

        lines.append("**Confirmatory data check (optional, non-blocking):**\n")
        if gm_row is not None:
            g_struct = call_gate(gm_row)
            g_marks = marks_only_gate(gm_row)
            lines.append(
                f"- GM12878: structural='{g_struct}', marks-only='{g_marks}' "
                f"(marks: { {m: gm_row.get(m) for m in ['H3K4me3','H3K27ac','H3K4me1','H3K27me3','H3K9me3']} }, "
                f"expressed={gm_row.get('expressed')})\n"
            )
        else:
            lines.append(f"- GM12878: not loaded ({gm_err})\n")

        if k562_row is not None:
            k_struct = call_gate(k562_row)
            k_marks = marks_only_gate(k562_row)
            lines.append(
                f"- K562: structural='{k_struct}', marks-only='{k_marks}' "
                f"(marks: { {m: k562_row.get(m) for m in ['H3K4me3','H3K27ac','H3K4me1','H3K27me3','H3K9me3']} }, "
                f"expressed={k562_row.get('expressed')})\n"
            )
        else:
            lines.append(f"- K562: not loaded ({k562_err})\n")

        lines.append("")

    lines.append("## Summary counts\n")
    lines.append("| classification | count |")
    lines.append("|---|---|")
    for label, count in counts.items():
        lines.append(f"| {label} | {count} |")
    lines.append("")
    lines.append(
        f"\n{counts['open-with-cause-still-unidentified']} of "
        f"{len(gene_records)} genes remain open. Per standing rule, none "
        f"should be fixed or reframed until a cause is named at "
        f"file/line level."
    )

    return "\n".join(lines)


def main() -> None:
    print("Running exhaustive self-test...")
    run_synthetic_self_test()

    print("Parsing gene-level table...")
    genes = parse_gene_table(SWITCHING_CONTEXT_PATH)

    print("Building resolution report (real binary matrices loaded "
          "opportunistically, not required)...")
    report = run_resolution(genes)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"Results written to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()