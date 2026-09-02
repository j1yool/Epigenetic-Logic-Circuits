"""
compare_switching_gene_directionality.py

Uses the CONFIRMED schema from switching_gene_schema_check.txt:
  switching_gene_auc_comparison.csv: cell_line, model_type, subset_auc, genome_wide_auc, subset_n
  switching_genes_v1.tsv: gene_id, gene_name, gate_type_GM12878, gate_type_K562,
                          switch_direction, GM12878_expression, K562_expression, replicate_check

Question: does the switching-gene subset show a CONSISTENT or INCONSISTENT
direction of (subset_auc - genome_wide_auc) across cell lines, for the
marks_only model, now that genome_wide_auc reflects the reconciled benchmark?
"""

import os
import pandas as pd

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

REPO_DIR = r"C:\Users\jamoo\Downloads\LOGIC CIRCUITS"
CURRENT_CSV = os.path.join(REPO_DIR, "switching_gene_auc_comparison.csv")
CANDIDATE_TSV = os.path.join(REPO_DIR, "switching_genes_v1.tsv")
OLD_INTERPRETATION_DOC = os.path.join(REPO_DIR, "switching_gene_validation_interpretation_v1.md")

# Locked benchmarks from MASTER_STATUS.md, for a sanity cross-check only —
# never used to overwrite or adjust anything computed from the CSV.
LOCKED_BENCHMARKS = {
    ("GM12878", "gate_type"): 0.9864,
    ("GM12878", "marks_only"): 0.7974,
    ("K562", "gate_type"): 0.9986,
    ("K562", "marks_only"): 0.7531,
}

VERDICT_FILE = os.path.join(REPO_DIR, "switching_gene_directionality_verdict.txt")


def main():
    lines = ["=== Switching-Gene Directionality Verdict ===\n"]

    df = pd.read_csv(CURRENT_CSV)

    # --- Step 1: sanity-check genome_wide_auc against locked benchmarks ---
    lines.append("--- Sanity check: genome_wide_auc vs. locked MASTER_STATUS.md benchmarks ---")
    mismatch_found = False
    for _, row in df.iterrows():
        key = (row["cell_line"], row["model_type"])
        if key in LOCKED_BENCHMARKS:
            locked = LOCKED_BENCHMARKS[key]
            actual = row["genome_wide_auc"]
            diff = abs(locked - actual)
            flag = "OK" if diff < 0.001 else "MISMATCH — investigate before trusting delta below"
            lines.append(f"{key}: locked={locked}, csv={actual}, diff={diff:.4f} [{flag}]")
            if diff >= 0.001:
                mismatch_found = True
    lines.append("")

    # --- Step 2: compute subset vs genome-wide delta per cell line, marks_only model ---
    lines.append("--- Marks-only model: subset_auc vs. genome_wide_auc, by cell line ---")
    marks_only = df[df["model_type"] == "marks_only"].copy()
    marks_only["delta"] = marks_only["subset_auc"] - marks_only["genome_wide_auc"]
    lines.append(marks_only[["cell_line", "subset_auc", "genome_wide_auc", "delta"]].to_string(index=False))
    lines.append("")

    deltas = dict(zip(marks_only["cell_line"], marks_only["delta"]))
    signs = {cl: ("positive" if d > 0 else "negative") for cl, d in deltas.items()}
    lines.append(f"Signs: {signs}")

    directionally_consistent = len(set(signs.values())) == 1
    lines.append(f"Direction consistent across cell lines: {directionally_consistent}\n")

    # --- Step 3: same check for gate_type model, for completeness ---
    lines.append("--- Gate-type model: subset_auc vs. genome_wide_auc, by cell line (context only) ---")
    gate_type = df[df["model_type"] == "gate_type"].copy()
    gate_type["delta"] = gate_type["subset_auc"] - gate_type["genome_wide_auc"]
    lines.append(gate_type[["cell_line", "subset_auc", "genome_wide_auc", "delta"]].to_string(index=False))
    lines.append("")

    # --- Step 4: gene-level switch_direction breakdown, for interpretive context ---
    if os.path.exists(CANDIDATE_TSV):
        genes = pd.read_csv(CANDIDATE_TSV, sep="\t")
        lines.append("--- switch_direction distribution (switching_genes_v1.tsv) ---")
        lines.append(genes["switch_direction"].value_counts().to_string())
        lines.append("")
        lines.append(
            "Interpretive note: if one switch_direction category (e.g. AND\u2192INC) "
            "dominates the subset, an asymmetric shift toward INCONSISTENT-type calls "
            "in one cell line could mechanically explain a marks-only AUC drop in that "
            "cell line specifically, independent of any remaining model artifact."
        )
    else:
        lines.append(f"NOTE: {CANDIDATE_TSV} not found — skipping gene-level direction breakdown.")
    lines.append("")

    # --- Step 5: pull whatever the old provisional doc claimed, for before/after comparison ---
    lines.append("--- Prior provisional doc (~Day 25) for before/after comparison ---")
    if os.path.exists(OLD_INTERPRETATION_DOC):
        with open(OLD_INTERPRETATION_DOC, "r", encoding="utf-8", errors="ignore") as f:
            prior_text = f.read()
        prior_hits = [
            l for l in prior_text.splitlines()
            if any(kw in l.lower() for kw in ["direction", "inconsistent", "gm12878", "k562"])
        ]
        for l in prior_hits:
            lines.append(f"  > {l.strip()}")
        lines.append(
            "\nCompare these prior statements against the Step 2 deltas above: if the prior "
            "doc reported the SAME sign pattern under the old (unreconciled) model, that is "
            "modest evidence the inconsistency is not solely an artifact of the lbfgs/"
            "GM12878-only mismatch, since it reproduces post-fix. If the prior doc's numbers "
            "differ substantially in magnitude (not just precision) from Step 2, the "
            "reconciliation did change the picture and this needs care in how it's worded."
        )
    else:
        lines.append(f"NOT FOUND at {OLD_INTERPRETATION_DOC} — no prior text to compare against.")
    lines.append("")

    # --- Step 6: verdict ---
    lines.append("=== VERDICT ===")
    if mismatch_found:
        lines.append(
            "genome_wide_auc does NOT match locked MASTER_STATUS.md benchmarks. "
            "STOP — do not finalize a directionality verdict until this is resolved; "
            "the delta computed above may not reflect the reconciled model."
        )
    else:
        lines.append("genome_wide_auc confirmed to match locked benchmarks (within rounding).")
        if directionally_consistent:
            lines.append(
                "Delta sign is CONSISTENT across cell lines post-reconciliation → "
                "the inconsistency previously flagged was a symptom of the old model "
                "mismatch. Close as resolved artifact."
            )
        else:
            lines.append(
                "Delta sign is INCONSISTENT across cell lines post-reconciliation "
                f"(GM12878 delta={deltas.get('GM12878'):+.4f}, K562 delta={deltas.get('K562'):+.4f}). "
                "This survives the reconciliation and is a REAL finding, not a fixed artifact. "
                "It belongs in the Discussion section: switching genes are more marks-predictable "
                "than the genome-wide baseline in GM12878, but less marks-predictable than the "
                "genome-wide baseline in K562 — an asymmetry that itself may be biologically "
                "meaningful (e.g., different regulatory logic mechanisms per cell-line context) "
                "rather than noise."
            )

    with open(VERDICT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Verdict written to: {VERDICT_FILE}")


if __name__ == "__main__":
    main()