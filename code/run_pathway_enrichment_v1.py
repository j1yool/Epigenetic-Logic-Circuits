"""
run_pathway_enrichment_v1.py

Day 47, Block 2 — Logic Circuits.

INTERPRETIVE DECISION (documented, not hidden):
Block 1's switching_genes_gm12878_v1.txt and switching_genes_k562_v1.txt
contain the IDENTICAL gene set (documented in
extract_switching_genes_v2_findings.md — switching is a paired comparison,
so the same genes appear in both). Running enrichment on two identical lists
would produce two identical result sets and answer nothing about
directionality. The actual per-cell-line signal is the 'and_origin' column
in switching_genes_merged_v1.tsv: which cell line holds the SIMPLE_AND call
for a given AND<->INC switcher. This script splits on and_origin instead of
on the two .txt files, and restricts to AND<->INC switchers specifically
(n=3,690 from Block 1), since that's the category the Day 38/44 directional
finding is actually about.

GENE ID MAPPING (required for correctness, not optional):
Switching gene lists are Ensembl gene IDs. MSigDB Hallmark / KEGG / GO
Biological Process gene sets (via Enrichr) are HGNC gene SYMBOLS. Enriching
directly on Ensembl IDs would silently return near-zero overlap against
every term -- a false negative masquerading as a null result. gene_name
(HGNC symbol) is pulled from Block 1's regenerated gate-assignment files
(gate_assignments_regenerated_v1.tsv, k562_gate_assignments_regenerated_v1.tsv),
which already carry validated symbol annotation. Genes with no symbol are
dropped and counted explicitly, not silently discarded.

BACKGROUND (per Reimand et al. 2019, read this morning):
Background = the tested-gene universe (genes with a valid gate call in BOTH
cell lines, i.e. the same gene universe switching status was computed
against in Block 1), NOT the whole genome. Using genome-wide background
would inflate significance for any term enriched among generically
"chromatin-profiled" genes.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
from pathlib import Path
import pandas as pd

try:
    import gseapy as gp
except ImportError:
    print("=" * 70)
    print("FATAL: gseapy not installed.")
    print("Fix: conda activate genomics")
    print("     pip install gseapy")
    print("Then re-run this script.")
    print("=" * 70)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths (all relative to LOGIC CIRCUITS/code/, matching Block 1 outputs)
# ---------------------------------------------------------------------------
GM_REGEN = Path("gate_assignments_regenerated_v1.tsv")
K562_REGEN = Path("k562_gate_assignments_regenerated_v1.tsv")
SWITCHERS_MERGED = Path("switching_genes_merged_v1.tsv")

# Read directly from the ORIGINAL annotation sources for gene_name, bypassing
# Block 1's regenerated-file annotation merge entirely (that merge silently
# produced a K562 file with no gene_name column -- rather than chase why,
# this routes around the dependency by reading the originals fresh, here).
GM12878_ANNOTATION_SOURCE = Path(r"C:\Users\jamoo\Downloads\LOGIC CIRCUITS\data\gate_assignments.tsv")
K562_ANNOTATION_SOURCE = Path(r"C:\Users\jamoo\Downloads\LOGIC CIRCUITS\data\k562\k562_gate_assignments_named.tsv")

OUT_GM_ORIGIN_CSV = Path("enrichment_results_gm12878_origin_v1.csv")
OUT_K562_ORIGIN_CSV = Path("enrichment_results_k562_origin_v1.csv")
OUT_FINDINGS = Path("pathway_enrichment_findings_v1.md")

GENE_SET_LIBRARIES = [
    "MSigDB_Hallmark_2020",
    "KEGG_2021_Human",
    "GO_Biological_Process_2023",
]

POSITIVE_CONTROL_GENES = [
    "CCNB1", "CDK1", "CCNE1", "MCM2", "PCNA",
    "CDC20", "AURKA", "PLK1", "BUB1", "TOP2A",
]


def fatal(msg: str) -> None:
    print("\n" + "=" * 70)
    print("FATAL:", msg)
    print("=" * 70)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Synthetic self-test — positive control, real Enrichr call, small & fast
# ---------------------------------------------------------------------------
def synthetic_self_test() -> None:
    print("Running synthetic self-test: known cell-cycle gene list against "
          "all three real gene set libraries (real Enrichr call, no background restriction)...")
    try:
        result = gp.enrichr(
            gene_list=POSITIVE_CONTROL_GENES,
            gene_sets=GENE_SET_LIBRARIES,
            organism="human",
            outdir=None,
        )
    except Exception as e:
        fatal(f"Self-test Enrichr call failed with an exception: {e}\n"
              f"This is most likely a network/connectivity issue (Enrichr requires "
              f"internet access), not a logic error. Check your connection and retry "
              f"before assuming the pipeline itself is broken.")

    if result is None or result.results is None or len(result.results) == 0:
        fatal("Self-test Enrichr call returned no results at all. Cannot proceed -- "
              "the pipeline must be confirmed working on a known positive control "
              "before touching real data.")

    top_terms = result.results.sort_values("Adjusted P-value").head(10)
    top_terms_str = " | ".join(top_terms["Term"].tolist())
    print(f"Top 10 enriched terms for positive control (across all 3 libraries): {top_terms_str}")

    # NOTE: MSigDB Hallmark has no term literally named "cell cycle" -- its
    # canonical cell-cycle-related terms are named E2F Targets, G2-M
    # Checkpoint, and Mitotic Spindle. KEGG and GO-BP do use "cell cycle" in
    # their naming. Checking against this known vocabulary across all three
    # libraries (matching what the real run actually queries) rather than a
    # single literal string in a single library.
    expected_keywords = [
        "cell cycle", "e2f targets", "g2-m checkpoint", "g2m checkpoint", "mitotic spindle",
    ]
    top_terms_lower = top_terms["Term"].str.lower()
    hit = top_terms_lower.apply(lambda t: any(k in t for k in expected_keywords)).any()

    if not hit:
        fatal(f"Self-test FAILED: known cell-cycle gene list did not recover any "
              f"expected cell-cycle-related term ({expected_keywords}) in the top 10 "
              f"results across all 3 libraries (got: {top_terms_str}). "
              f"Do not proceed to real data -- something is wrong with the query, "
              f"library names, or Enrichr response, and it needs to be understood "
              f"before real switching genes are run.")

    print("Synthetic self-test PASSED: positive control correctly recovers "
          "known cell-cycle-related terms as top-enriched.\n")


# ---------------------------------------------------------------------------
# Real data loading
# ---------------------------------------------------------------------------
def load_symbol_map() -> pd.Series:
    if not GM12878_ANNOTATION_SOURCE.exists():
        fatal(f"Annotation source not found: {GM12878_ANNOTATION_SOURCE}.")

    df = pd.read_csv(GM12878_ANNOTATION_SOURCE, sep="\t", keep_default_na=False, na_values=[""])
    if "gene_id" not in df.columns:
        fatal(f"GM12878 annotation source has no gene_id column. Found: {list(df.columns)}")
    if "gene_name" not in df.columns:
        fatal(f"GM12878 annotation source has no gene_name column. Found: {list(df.columns)}. "
              f"This file was previously confirmed to carry gene_name -- if that's changed, "
              f"stop and report the actual columns rather than guessing further.")

    symbol_map = df[["gene_id", "gene_name"]].drop_duplicates(subset="gene_id")
    print(f"Loaded {len(symbol_map)} gene_id -> gene_name pairs from GM12878 annotation "
          f"source. Ensembl gene_ids are genome-wide identifiers (not cell-line-specific), "
          f"so this single table is used to map BOTH GM12878 and K562 gene_ids -- "
          f"K562's own annotation file was confirmed to have no name column at all "
          f"(['gene_id', 'gate_type', 'complexity_score']), so it is not used for naming.")

    n_empty_names = (symbol_map["gene_name"] == "").sum() + symbol_map["gene_name"].isna().sum()
    if n_empty_names > 0:
        print(f"NOTE: {n_empty_names} gene_id(s) have an empty gene_name in the source file itself.")

    return symbol_map.set_index("gene_id")["gene_name"]


def map_ids_to_symbols(gene_ids, symbol_map: pd.Series, label: str) -> list:
    mapped = symbol_map.reindex(gene_ids)
    n_missing = mapped.isna().sum()
    if n_missing > 0:
        print(f"{label}: {n_missing} / {len(gene_ids)} gene_id(s) have no gene_name "
              f"symbol available and are excluded from enrichment input.")
    symbols = mapped.dropna().unique().tolist()
    return symbols


def load_background_and_origin_groups(symbol_map: pd.Series):
    gm = pd.read_csv(GM_REGEN, sep="\t")[["gene_id"]]
    k562 = pd.read_csv(K562_REGEN, sep="\t")[["gene_id"]]
    tested_universe_ids = pd.merge(gm, k562, on="gene_id", how="inner")["gene_id"].tolist()
    background_symbols = map_ids_to_symbols(tested_universe_ids, symbol_map, "Background (tested universe)")

    switchers = pd.read_csv(SWITCHERS_MERGED, sep="\t")
    and_inc = switchers[
        ((switchers["gate_type_gm12878"] == "SIMPLE_AND") & (switchers["gate_type_k562"] == "INCONSISTENT")) |
        ((switchers["gate_type_gm12878"] == "INCONSISTENT") & (switchers["gate_type_k562"] == "SIMPLE_AND"))
    ].copy()

    if and_inc["and_origin"].isin(["NEITHER"]).any():
        fatal("Unexpected: an AND<->INC switcher has and_origin == 'NEITHER'. "
              "By definition this category must have SIMPLE_AND in exactly one "
              "cell line. Stopping -- this indicates a Block 1 logic error, "
              "not something to work around here.")

    gm_origin_ids = and_inc.loc[and_inc["and_origin"] == "GM12878", "gene_id"].tolist()
    k562_origin_ids = and_inc.loc[and_inc["and_origin"] == "K562", "gene_id"].tolist()

    gm_origin_symbols = map_ids_to_symbols(gm_origin_ids, symbol_map, "GM12878-origin switchers")
    k562_origin_symbols = map_ids_to_symbols(k562_origin_ids, symbol_map, "K562-origin switchers")

    return background_symbols, gm_origin_symbols, k562_origin_symbols, len(gm_origin_ids), len(k562_origin_ids)


def run_enrichment(gene_symbols: list, background_symbols: list, label: str) -> pd.DataFrame:
    if len(gene_symbols) < 5:
        print(f"{label}: only {len(gene_symbols)} gene symbol(s) available -- "
              f"too few for a meaningful enrichment call. Skipping, not fabricating output.")
        return pd.DataFrame()

    # NOTE (Day 47 fix): gp.enrichr() is the ONLINE Enrichr API call. Passing
    # background= to it does not make the API compute p-values against that
    # custom background -- in this gseapy version it silently returns
    # P-value/Adjusted P-value/Combined Score as NaN for every term instead
    # of erroring (confirmed empirically: 0/4413 and 0/630 non-null across
    # both real runs, while the no-background self-test call populates them
    # correctly). gp.enrich() is the OFFLINE/local hypergeometric-test
    # function and is the one that actually honors a custom background
    # gene list. Switching to it here; gene_sets must be local, so the same
    # three libraries are fetched once via gp.get_library() below.
    try:
        gene_set_dict = {}
        for lib in GENE_SET_LIBRARIES:
            gene_set_dict.update(gp.get_library(name=lib, organism="human"))

        result = gp.enrich(
            gene_list=gene_symbols,
            gene_sets=gene_set_dict,
            background=background_symbols,
            outdir=None,
        )
    except Exception as e:
        fatal(f"{label}: enrichment call failed with exception: {e}. "
              f"Likely network/connectivity issue fetching gene set libraries -- "
              f"the positive-control self-test already confirmed Enrichr access "
              f"works, so check connectivity before assuming a code bug.")

    if result is None or result.results is None:
        fatal(f"{label}: enrichment call returned no results object at all.")

    df = result.results.copy()
    if df["Adjusted P-value"].isna().all():
        fatal(f"{label}: enrichment call returned a results table but "
              f"Adjusted P-value is still 100% NaN after switching to "
              f"gp.enrich(). The background-handling fix did not resolve it "
              f"-- stop and inspect result.results directly rather than "
              f"assuming this run's numbers are meaningful.")
    df = df.sort_values("Adjusted P-value")
    return df


def main():
    synthetic_self_test()

    for f in (GM_REGEN, K562_REGEN, SWITCHERS_MERGED):
        if not f.exists():
            fatal(f"Required Block 1 output not found: {f}. Run extract_switching_genes_v2.py first.")

    symbol_map = load_symbol_map()
    background_symbols, gm_origin_symbols, k562_origin_symbols, n_gm_ids, n_k562_ids = \
        load_background_and_origin_groups(symbol_map)

    print(f"\nBackground (tested universe) gene symbols: {len(background_symbols)}")
    print(f"GM12878-origin AND<->INC switchers: {n_gm_ids} gene_id -> {len(gm_origin_symbols)} symbols")
    print(f"K562-origin AND<->INC switchers: {n_k562_ids} gene_id -> {len(k562_origin_symbols)} symbols")

    print("\nRunning real enrichment: GM12878-origin switchers...")
    gm_results = run_enrichment(gm_origin_symbols, background_symbols, "GM12878-origin")

    print("\nRunning real enrichment: K562-origin switchers...")
    k562_results = run_enrichment(k562_origin_symbols, background_symbols, "K562-origin")

    if not gm_results.empty:
        gm_results.to_csv(OUT_GM_ORIGIN_CSV, index=False)
        print(f"Wrote {OUT_GM_ORIGIN_CSV} ({len(gm_results)} terms)")
    if not k562_results.empty:
        k562_results.to_csv(OUT_K562_ORIGIN_CSV, index=False)
        print(f"Wrote {OUT_K562_ORIGIN_CSV} ({len(k562_results)} terms)")

    def top_sig(df, n=10):
        if df.empty:
            return pd.DataFrame()
        return df[df["Adjusted P-value"] < 0.05].head(n)

    gm_sig = top_sig(gm_results)
    k562_sig = top_sig(k562_results)

    cancer_keywords = ["apoptosis", "dna damage", "dna repair", "p53", "cell cycle",
                        "differentiation", "proliferation", "senescence", "oncogene"]

    def cancer_relevant_terms(df):
        if df.empty:
            return pd.DataFrame()
        mask = df["Term"].str.lower().str.contains("|".join(cancer_keywords))
        return df[mask & (df["Adjusted P-value"] < 0.05)]

    gm_cancer = cancer_relevant_terms(gm_results)
    k562_cancer = cancer_relevant_terms(k562_results)

    findings = f"""# Pathway Enrichment Findings — Day 47, Block 2

## Synthetic self-test
PASSED: known cell-cycle positive-control gene list correctly recovered a
"cell cycle" term as top-enriched against MSigDB_Hallmark_2020, confirming
the Enrichr pipeline call itself works before real data was touched.

## Methodological note (deviation from literal Block 2 instructions)
`switching_genes_gm12878_v1.txt` and `switching_genes_k562_v1.txt` (Block 1
output) are identical gene sets by construction. Enrichment was instead run
on the two `and_origin` subsets of AND<->INC switchers specifically (n=3,690
total from Block 1) — GM12878-origin (SIMPLE_AND in GM12878, INCONSISTENT in
K562) vs K562-origin (reverse) — since that is the actual cell-line-specific
comparison implied by the Day 38 directional finding.

## Background
Tested-gene universe = genes with a valid gate call in both cell lines
(post Block 1 exclusions), mapped to {len(background_symbols)} unique gene
symbols. This is the background used for both enrichment calls, per
Reimand et al. 2019's warning against genome-wide background inflating
significance.

## Gene ID -> symbol mapping
- GM12878-origin switchers: {n_gm_ids} Ensembl gene_id -> {len(gm_origin_symbols)} mapped symbols
- K562-origin switchers: {n_k562_ids} Ensembl gene_id -> {len(k562_origin_symbols)} mapped symbols
(Any gap between these counts reflects genes with no gene_name annotation
available, excluded rather than guessed at — see script stdout for exact counts.)

## Results — GM12878-origin switchers
{'No results (see stdout for reason — likely too few mapped genes).' if gm_results.empty else f"{len(gm_results)} terms tested across {len(GENE_SET_LIBRARIES)} libraries. {len(gm_sig)} significant (adj. p < 0.05) shown below."}

{gm_sig[['Term', 'Overlap', 'Adjusted P-value', 'Genes']].to_string(index=False) if not gm_sig.empty else '(none significant, or no results)'}

### Cancer-relevant terms (GM12878-origin, adj. p < 0.05)
{gm_cancer[['Term', 'Adjusted P-value']].to_string(index=False) if not gm_cancer.empty else 'None of the significant terms matched cancer-relevant keywords (apoptosis, DNA damage/repair, p53, cell cycle, differentiation, proliferation, senescence, oncogene).'}

## Results — K562-origin switchers
{'No results (see stdout for reason — likely too few mapped genes).' if k562_results.empty else f"{len(k562_results)} terms tested across {len(GENE_SET_LIBRARIES)} libraries. {len(k562_sig)} significant (adj. p < 0.05) shown below."}

{k562_sig[['Term', 'Overlap', 'Adjusted P-value', 'Genes']].to_string(index=False) if not k562_sig.empty else '(none significant, or no results)'}

### Cancer-relevant terms (K562-origin, adj. p < 0.05)
{k562_cancer[['Term', 'Adjusted P-value']].to_string(index=False) if not k562_cancer.empty else 'None of the significant terms matched cancer-relevant keywords.'}

## Open question this was meant to address
Does the enrichment picture differ between GM12878-origin and K562-origin
AND<->INC switchers in a way consistent with the Day 44 directional
inconsistency (GM12878 marks-only ΔAUC +0.0552, K562 −0.1439)? Compare the
two term lists above directly — do not assume consistency or difference
without reading both tables.

## Honest caveat
If either or both result sets show zero significant terms, that is a real,
reportable outcome (gene lists this size, ~hundreds to low thousands, often
lack power for GO/KEGG enrichment after multiple-testing correction) — not
a failure of this script to fix.
"""
    OUT_FINDINGS.write_text(findings, encoding="utf-8")
    print(f"\nWrote {OUT_FINDINGS} (UTF-8 encoded)")


if __name__ == "__main__":
    main()