"""
check_switching_gene_csv_provenance.py

Verdict: was switching_gene_auc_comparison.csv regenerated AFTER the
Day 35 AUC-mismatch fix (solver="liblinear", both cell lines) landed
in block5_auc_comparison_both_lines.py?

Filesystem-mtime based (no git dependency).
"""

import os
import glob

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

REPO_DIR = r"C:\Users\jamoo\Downloads\LOGIC CIRCUITS"
CODE_DIR = REPO_DIR
DATA_DIR = REPO_DIR
FIX_SCRIPT_NAME = "block5_auc_comparison_both_lines.py"
TARGET_CSV_NAME = "switching_gene_auc_comparison.csv"
OUTPUT_FILE = os.path.join(REPO_DIR, "switching_gene_csv_provenance_verdict.txt")

def find_file(name, search_dirs):
    for d in search_dirs:
        candidate = os.path.join(d, name)
        if os.path.exists(candidate):
            return candidate
    return None


def find_csv_producer_scripts(code_dir, target_csv_name):
    matches = []
    for filepath in glob.glob(os.path.join(code_dir, "*.py")):
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception as e:
            matches.append((filepath, [f"__READ_ERROR__: {e}"]))
            continue
        hits = [f"L{i+1}: {line.strip()}" for i, line in enumerate(lines) if target_csv_name in line]
        if hits:
            matches.append((filepath, hits))
    return matches


def check_producer_model_definition(filepath):
    signals = {"uses_liblinear": False, "references_gm12878": False, "references_k562": False, "evidence": []}
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        signals["evidence"].append(f"__READ_ERROR__: {e}")
        return signals
    if "liblinear" in content:
        signals["uses_liblinear"] = True
        signals["evidence"].append("Contains solver='liblinear'.")
    if "GM12878" in content:
        signals["references_gm12878"] = True
    if "K562" in content:
        signals["references_k562"] = True
    return signals


def main():
    report = ["=== Switching-Gene CSV Provenance Check (mtime-based) ===\n"]

    fix_script_path = find_file(FIX_SCRIPT_NAME, [CODE_DIR, REPO_DIR])
    csv_path = find_file(TARGET_CSV_NAME, [DATA_DIR, CODE_DIR, REPO_DIR])

    if not fix_script_path:
        report.append(f"FIX SCRIPT NOT FOUND: {FIX_SCRIPT_NAME} not in {CODE_DIR} or {REPO_DIR}\n")
    if not csv_path:
        report.append(f"CSV NOT FOUND: {TARGET_CSV_NAME} not in {DATA_DIR}, {CODE_DIR}, or {REPO_DIR}\n")

    mtime_says_fresh = None
    if fix_script_path and csv_path:
        fix_mtime = os.path.getmtime(fix_script_path)
        csv_mtime = os.path.getmtime(csv_path)
        mtime_says_fresh = csv_mtime > fix_mtime
        report.append(f"Fix script path: {fix_script_path}")
        report.append(f"Fix script mtime: {fix_mtime}")
        report.append(f"CSV path: {csv_path}")
        report.append(f"CSV mtime: {csv_mtime}")
        report.append(f"CSV is {'NEWER' if mtime_says_fresh else 'OLDER/EQUAL'} than fix script by mtime.\n")
        report.append(
            "NOTE: mtime is a weaker signal than commit history — it can be reset by copies, "
            "pulls, or backups. Treat this as supporting evidence, not sole proof.\n"
        )

    report.append("--- Scripts referencing target CSV filename ---")
    producers = find_csv_producer_scripts(CODE_DIR, TARGET_CSV_NAME)
    if not producers:
        report.append(f"NO SCRIPT FOUND referencing '{TARGET_CSV_NAME}' in {CODE_DIR}.\n")
    else:
        for filepath, hits in producers:
            report.append(f"\nFile: {filepath}")
            for h in hits:
                report.append(f"  {h}")
        report.append("")

    report.append("--- Model-definition check on candidate producer(s) ---")
    any_confirmed_fixed = False
    for filepath, _ in producers:
        signals = check_producer_model_definition(filepath)
        report.append(f"\n{filepath}:")
        report.append(f"  uses_liblinear      : {signals['uses_liblinear']}")
        report.append(f"  references_GM12878  : {signals['references_gm12878']}")
        report.append(f"  references_K562     : {signals['references_k562']}")
        for e in signals["evidence"]:
            report.append(f"    evidence: {e}")
        if signals["uses_liblinear"] and signals["references_gm12878"] and signals["references_k562"]:
            any_confirmed_fixed = True

    report.append("\n=== VERDICT ===")
    if any_confirmed_fixed and (mtime_says_fresh is True or mtime_says_fresh is None):
        verdict = "CONFIRMED post-reconciliation, safe to use"
    elif any_confirmed_fixed and mtime_says_fresh is False:
        verdict = (
            "AMBIGUOUS: producer script uses fixed model definition (liblinear, both cell lines), "
            "but CSV mtime predates fix script mtime. Regenerate to be safe — mtime evidence alone "
            "is weak, but combined with an out-of-order timestamp it's not worth trusting silently."
        )
    else:
        verdict = "STALE, predates fix, must regenerate (producer script does not confirm fixed model definition)"

    report.append(f"\nFINAL VERDICT: {verdict}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"Provenance check complete. Verdict written to: {OUTPUT_FILE}")
    print(f"FINAL VERDICT: {verdict}")


if __name__ == "__main__":
    main()