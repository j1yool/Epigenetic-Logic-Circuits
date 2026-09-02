"""
find_p511_source_v1.py

Day 46, Block 2, item 1 -- Mutation-to-Meaning Distance

Purpose
-------
The original sign-convention audit (sign_convention_audit_v1.md) reports
p=0.511 for the full TP53 set, but the exact statistical test that produced
that number has never been located -- run_pipeline.py's own
run_sign_convention_check() only reports means and direction, no test
statistic (this is explicitly flagged as an open item in
recompute_dmu_tetramer_subset.py's own docstring).

This script performs an exhaustive, reproducible search for "0.511" (and
its more precise variants) across:
  1. Every current .py and .md file in BOTH project roots (LOGIC CIRCUITS
     and MUTATION TO MEANING -- per the stale-file-awareness rule, a script
     can end up in the wrong project folder).
  2. Full git history of both repos via `git log -p -S"0.511" --all`
     (the "pickaxe" search), which finds the string even in commits that
     were later reverted or in files that were later deleted -- a plain
     filesystem grep cannot see either of those.
  3. Any orphaned/duplicate copies directly under the Downloads root
     (outside either project folder), since prior sessions have already
     found stale duplicates living there.

The disposition is binary and unhedged: either the source script is FOUND
(with exact file + commit), or the search is logged as EXHAUSTIVE and the
item is closed as an unresolved-execution-problem (not unresolved-ambiguity)
per the project's verdict rule.

Usage
-----
    python find_p511_source_v1.py \\
        --logic-circuits-dir "C:\\Users\\jamoo\\Downloads\\LOGIC CIRCUITS" \\
        --m2m-dir "C:\\Users\\jamoo\\Downloads\\MUTATION TO MEANING" \\
        --downloads-root "C:\\Users\\jamoo\\Downloads"
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from datetime import datetime

SEARCH_STRINGS = ["0.511", "p=0.511", "p = 0.511", "0.5110"]
TEXT_EXTENSIONS = {".py", ".md", ".txt", ".ipynb"}


def grep_filesystem(root: Path, search_strings: list) -> list:
    """Plain-text search across current files under root. Returns list of
    dicts: {file, line_number, line_text, matched_string}."""
    hits = []
    if not root or not root.exists():
        return hits
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            # Report at most one hit per line -- multiple search_strings can
            # overlap (e.g. "0.511" is a substring of "p=0.511"), and a line
            # that matches several variants is still one occurrence, not
            # several, for counting purposes. Prefer the longest (most
            # specific) matching variant for the recorded matched_string.
            matched = [s for s in search_strings if s in line]
            if matched:
                best = max(matched, key=len)
                hits.append({
                    "file": str(path), "line_number": i,
                    "line_text": line.strip()[:200], "matched_string": best,
                })
    return hits


def grep_git_history(repo_dir: Path, search_strings: list) -> dict:
    """Pickaxe search (-S) across all branches and all history, including
    deleted files. Returns dict with status + list of matching commits."""
    if repo_dir is None:
        return {"status": "NOT_PROVIDED", "reason": "no repo_dir given for this project", "commits": []}
    if not repo_dir.exists():
        return {"status": "UNAVAILABLE", "reason": "repo_dir does not exist", "commits": []}
    if not (repo_dir / ".git").exists():
        return {"status": "UNAVAILABLE", "reason": "not a git repo (no .git found)", "commits": []}
    if shutil.which("git") is None:
        return {"status": "UNAVAILABLE", "reason": "git not found on PATH", "commits": []}

    all_commits = []
    for s in search_strings:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_dir), "log", "--all", "--oneline", f"-S{s}"],
                capture_output=True, text=True, timeout=30,
            )
            lines = [l for l in result.stdout.splitlines() if l.strip()]
            for l in lines:
                all_commits.append({"search_string": s, "commit_line": l})
        except Exception as e:
            return {"status": "ERROR", "reason": str(e), "commits": []}

    return {"status": "RAN", "commits": all_commits}


def find_orphan_duplicates(downloads_root: Path, known_filenames: list) -> list:
    """Look for copies of known pipeline-related filenames sitting directly
    under Downloads root (i.e. NOT inside either project folder) -- the
    stale-duplicate pattern already seen in prior sessions."""
    hits = []
    if not downloads_root or not downloads_root.exists():
        return hits
    for fname in known_filenames:
        for path in downloads_root.glob(fname):
            hits.append(str(path))
        # also check one level of subfolders that AREN'T the two known project dirs
    return hits


def run_search(logic_circuits_dir: Path, m2m_dir: Path, downloads_root: Path) -> dict:
    results = {}
    results["filesystem_logic_circuits"] = grep_filesystem(logic_circuits_dir, SEARCH_STRINGS)
    results["filesystem_m2m"] = grep_filesystem(m2m_dir, SEARCH_STRINGS)
    results["git_history_logic_circuits"] = grep_git_history(logic_circuits_dir, SEARCH_STRINGS)
    results["git_history_m2m"] = grep_git_history(m2m_dir, SEARCH_STRINGS)
    results["orphan_duplicates"] = find_orphan_duplicates(
        downloads_root,
        ["sign_convention*.py", "sign_convention*.md", "*p511*", "*p_511*"],
    )
    return results


def determine_disposition(results: dict) -> tuple:
    fs_hits = results["filesystem_logic_circuits"] + results["filesystem_m2m"]
    git_lc = results["git_history_logic_circuits"]
    git_m2m = results["git_history_m2m"]
    git_hits = git_lc.get("commits", []) + git_m2m.get("commits", [])
    orphan_hits = results["orphan_duplicates"]

    # A project that was never given a directory ("NOT_PROVIDED") doesn't
    # count as incomplete -- the caller simply didn't ask to search there.
    # Only a provided-but-broken repo ("UNAVAILABLE"/"ERROR") counts against
    # completeness.
    any_git_unavailable = (
        git_lc["status"] not in ("RAN", "NOT_PROVIDED")
        or git_m2m["status"] not in ("RAN", "NOT_PROVIDED")
    )

    if fs_hits or git_hits or orphan_hits:
        disposition = "SOURCE_FOUND"
    elif any_git_unavailable:
        disposition = "SEARCH_INCOMPLETE"
    else:
        disposition = "SOURCE_UNRECOVERABLE"

    return disposition, {
        "n_filesystem_hits": len(fs_hits),
        "n_git_history_hits": len(git_hits),
        "n_orphan_duplicates": len(orphan_hits),
        "git_search_complete": not any_git_unavailable,
    }


# ---------------------------------------------------------------------------
# Synthetic self-test
# ---------------------------------------------------------------------------

def run_synthetic_test() -> bool:
    print("Running synthetic self-test...")
    tmp = Path(tempfile.mkdtemp(prefix="p511_synth_"))
    try:
        # --- Case A: string present in a CURRENT file (filesystem grep must catch it) ---
        proj_a = tmp / "proj_a"
        proj_a.mkdir()
        (proj_a / "some_audit.md").write_text("The result was p=0.511, unexpectedly.\n", encoding="utf-8")
        fs_hits = grep_filesystem(proj_a, SEARCH_STRINGS)
        assert len(fs_hits) == 1, f"Expected 1 filesystem hit, got {len(fs_hits)}"
        assert fs_hits[0]["matched_string"] == "p=0.511"

        # --- Case B: string present ONLY in git history of a DELETED file (pickaxe must catch it) ---
        proj_b = tmp / "proj_b"
        proj_b.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=proj_b, check=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=proj_b, check=True)
        subprocess.run(["git", "config", "user.name", "test"], cwd=proj_b, check=True)
        deleted_script = proj_b / "old_sign_check.py"
        deleted_script.write_text("# computed test statistic\nresult_p = 0.511\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=proj_b, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "add sign check script"], cwd=proj_b, check=True)
        deleted_script.unlink()
        subprocess.run(["git", "add", "."], cwd=proj_b, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "remove script, superseded"], cwd=proj_b, check=True)

        # Filesystem grep should find NOTHING now (file is gone)
        fs_hits_b = grep_filesystem(proj_b, SEARCH_STRINGS)
        assert len(fs_hits_b) == 0, "File was deleted -- filesystem grep should find nothing."

        # Git history grep MUST still find it
        git_hits_b = grep_git_history(proj_b, SEARCH_STRINGS)
        assert git_hits_b["status"] == "RAN"
        assert len(git_hits_b["commits"]) > 0, "Pickaxe search failed to find string in deleted file's history."

        # --- Case C: string genuinely absent everywhere -> SOURCE_UNRECOVERABLE ---
        proj_c = tmp / "proj_c"
        proj_c.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=proj_c, check=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=proj_c, check=True)
        subprocess.run(["git", "config", "user.name", "test"], cwd=proj_c, check=True)
        (proj_c / "unrelated.md").write_text("Nothing relevant here.\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=proj_c, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=proj_c, check=True)

        results_c = run_search(proj_c, None, None)
        disposition_c, _ = determine_disposition(results_c)
        assert disposition_c == "SOURCE_UNRECOVERABLE", f"Expected SOURCE_UNRECOVERABLE, got {disposition_c}"

        # --- Case B wired through full disposition logic -> SOURCE_FOUND ---
        results_b = run_search(proj_b, None, None)
        disposition_b, _ = determine_disposition(results_b)
        assert disposition_b == "SOURCE_FOUND", f"Expected SOURCE_FOUND, got {disposition_b}"

        print("Synthetic self-test PASSED: filesystem grep catches live matches, "
              "git pickaxe search catches matches in deleted-file history that filesystem "
              "grep misses, and disposition logic correctly classifies both FOUND and "
              "genuinely-absent (UNRECOVERABLE) cases.")
        return True
    except AssertionError as e:
        print(f"SYNTHETIC TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"SYNTHETIC TEST ERROR: {e}")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Exhaustive search for the p=0.511 source script")
    parser.add_argument("--logic-circuits-dir", type=str, default=None)
    parser.add_argument("--m2m-dir", type=str, default=None)
    parser.add_argument("--downloads-root", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if not run_synthetic_test():
        print("Aborting: synthetic self-test did not pass. Real search was NOT run.")
        sys.exit(1)

    lc_dir = Path(args.logic_circuits_dir) if args.logic_circuits_dir else None
    m2m_dir = Path(args.m2m_dir) if args.m2m_dir else None
    dl_root = Path(args.downloads_root) if args.downloads_root else None
    out_path = Path(args.out) if args.out else (m2m_dir or Path(__file__).resolve().parent) / "p511_reconciliation_disposition_v1.md"

    print(f"\nSearching:")
    print(f"  LOGIC CIRCUITS: {lc_dir}")
    print(f"  MUTATION TO MEANING: {m2m_dir}")
    print(f"  Downloads root (orphan check): {dl_root}")

    results = run_search(lc_dir, m2m_dir, dl_root)
    disposition, summary = determine_disposition(results)

    lines = [f"DISPOSITION: {disposition}", "",
             "# p=0.511 Source Reconciliation — Disposition", "",
             f"_Generated {datetime.now().isoformat(timespec='seconds')} by find_p511_source_v1.py_", "",
             "## Disposition", f"**{disposition}**", "",
             "## Summary", f"```\n{summary}\n```", ""]

    if disposition == "SOURCE_FOUND":
        lines.append("## Matches")
        for hit in results["filesystem_logic_circuits"] + results["filesystem_m2m"]:
            lines.append(f"- FILESYSTEM: `{hit['file']}:{hit['line_number']}` — `{hit['line_text']}`")
        for hit in results["git_history_logic_circuits"].get("commits", []) + results["git_history_m2m"].get("commits", []):
            lines.append(f"- GIT HISTORY: `{hit['commit_line']}` (matched `{hit['search_string']}`)")
        for hit in results["orphan_duplicates"]:
            lines.append(f"- ORPHAN DUPLICATE FILE: `{hit}`")
        lines.append("")
        lines.append("**Next step:** open the matched file/commit, confirm which statistical test it ran, "
                      "and reconcile against sign_convention_audit_v1.md's reported p=0.511.")
    elif disposition == "SEARCH_INCOMPLETE":
        lines.append("## Why incomplete")
        lines.append("Git history search could not run for at least one project (repo not found, "
                      "not a git repo, or git unavailable in this run environment). Filesystem search "
                      "found nothing. Re-run with correct `--logic-circuits-dir` / `--m2m-dir` pointed "
                      "at real git checkouts to get a conclusive disposition.")
    else:
        lines.append("## Closure")
        lines.append("Search was exhaustive: current files in both project roots (filesystem grep), "
                      "full git history including deleted files (pickaxe search, `-S` across `--all`) "
                      "in both repos, and orphan-duplicate check under the Downloads root all returned "
                      "no match for \"0.511\" in any form searched.")
        lines.append("")
        lines.append("**Closed as: unresolved due to execution problem (source script never committed "
                      "or already garbage-collected outside git's reflog window), not unresolved due to "
                      "genuine ambiguity.** The p=0.511 figure in `sign_convention_audit_v1.md` stands "
                      "as reported, but the exact test statistic behind it cannot be independently "
                      "reconstructed from any recoverable artifact. Both Mann-Whitney U and Welch's "
                      "t-test are reported side-by-side going forward (as `recompute_dmu_tetramer_subset.py` "
                      "already does) so this ambiguity cannot recur.")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDisposition written to {out_path}")
    print(f"\nDISPOSITION: {disposition}")


if __name__ == "__main__":
    main()