"""
finalize_enrichment_findings_v1.py

Regenerates pathway_enrichment_findings_v1.md from the enrichment CSVs that
already completed successfully (enrichment_results_gm12878_origin_v1.csv,
enrichment_results_k562_origin_v1.csv). Does NOT re-run Enrichr -- those
network calls already succeeded and the results are on disk.

Fixes the one real bug: the original write_text() call used Windows' default
cp1252 encoding, which can't represent the "\u0394" (Delta) character in the
discussion text. This version writes UTF-8 explicitly.
"""

from pathlib import Path
import pandas as pd

GM_CSV = Path("enrichment_results_gm12878_origin_v1.csv")
K562_CSV = Path("enrichment_results_k562_origin_v1.csv")
OUT_FINDINGS = Path("pathway_enrichment_findings_v1.md")

for f in (GM_CSV, K562_CSV):
    if not f.exists():
        raise SystemExit(f"FATAL: {f} not found. Enrichment must have completed "
                          f"and written this file already -- check you're in the "
                          f"right directory.")

gm_results = pd.read_csv(GM_CSV)
k562_results = pd.read_csv(K562_CSV)

print(f"Loaded {GM_CSV}: {len(gm_results)} terms")
print(f"Loaded {K562_CSV}: {len(k562_results)} terms")


def top_sig(df, n=10):
    if df.empty:
        return pd.DataFrame()
    return df[df["Adjusted P-value"] < 0.05].sort_values("Adjusted P-value").head(n)


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

print(f"\nGM12878-origin: {len(gm_sig)} significant terms (adj p<0.05), "
      f"{len(gm_cancer)} cancer-relevant")
print(f"K562-origin: {len(k562_sig)} significant terms (adj p<0.05), "
      f"{len(k562_cancer)} cancer-relevant")

findings = f"""# Pathway Enrichment Findings — Day 47, Block 2

## Synthetic self-test
PASSED: known cell-cycle positive-control gene list correctly recovered
"Cell cycle" (KEGG) and multiple Hallmark/GO cell-cycle terms as
top-enriched, confirming the Enrichr pipeline call worked before real data
was touched.

## Methodological note (deviation from literal Block 2 instructions)
`switching_genes_gm12878_v1.txt` and `switching_genes_k562_v1.txt` (Block 1
output) are identical gene sets by construction. Enrichment was instead run
on the two `and_origin` subsets of AND<->INC switchers specifically (n=3,690
total from Block 1) -- GM12878-origin (SIMPLE_AND in GM12878, INCONSISTENT in
K562) vs K562-origin (reverse).

## Notable asymmetry (worth flagging regardless of enrichment results)
GM12878-origin switchers: n=3,502. K562-origin switchers: n=188. Nearly a
19:1 imbalance -- the AND<->INC switching gene population is overwhelmingly
dominated by genes cleanly AND-gated specifically in GM12878. This is
directly consistent with the Day 44 directional finding (GM12878 marks-only
model delta-AUC +0.0552 vs K562 delta-AUC -0.1439): the switching subset is
much more marks-predictable in GM12878 in part because most of it IS
GM12878's clean-AND-gate population by construction.

## Gene ID -> symbol mapping
- GM12878-origin switchers: 3,502 Ensembl gene_id -> 3,119 mapped symbols (381 unmapped)
- K562-origin switchers: 188 Ensembl gene_id -> 170 mapped symbols (17 unmapped)
- Background (tested universe): 42,771 mapped symbols (12,828 unmapped)

## Results — GM12878-origin switchers (n=3,119 mapped genes)
{len(gm_results)} terms tested across 3 libraries (Hallmark, KEGG, GO-BP). {len(gm_sig)} significant (adj. p < 0.05).

Top significant terms:
{gm_sig[['Term', 'Overlap', 'Adjusted P-value', 'Genes']].to_string(index=False) if not gm_sig.empty else '(none significant)'}

### Cancer-relevant terms (GM12878-origin, adj. p < 0.05)
{gm_cancer[['Term', 'Adjusted P-value']].to_string(index=False) if not gm_cancer.empty else 'None of the significant terms matched cancer-relevant keywords (apoptosis, DNA damage/repair, p53, cell cycle, differentiation, proliferation, senescence, oncogene).'}

## Results — K562-origin switchers (n=170 mapped genes)
{len(k562_results)} terms tested across 3 libraries. {len(k562_sig)} significant (adj. p < 0.05).

Top significant terms:
{k562_sig[['Term', 'Overlap', 'Adjusted P-value', 'Genes']].to_string(index=False) if not k562_sig.empty else '(none significant)'}

### Cancer-relevant terms (K562-origin, adj. p < 0.05)
{k562_cancer[['Term', 'Adjusted P-value']].to_string(index=False) if not k562_cancer.empty else 'None of the significant terms matched cancer-relevant keywords.'}

## Open question this was meant to address
Does the enrichment picture differ between GM12878-origin and K562-origin
AND<->INC switchers in a way consistent with the Day 44 directional
inconsistency? Note the large sample-size asymmetry above (3,119 vs 170
mapped genes) when interpreting any difference in enriched term counts --
K562-origin's much smaller gene set will naturally have less power to reach
significance after multiple-testing correction, independent of any true
biological difference.

## Honest caveat
If K562-origin shows few or no significant terms while GM12878-origin shows
many, that could reflect either (a) a genuine biological difference, or
(b) the ~18x smaller K562-origin gene set having systematically less
statistical power. This script does not adjudicate between these -- reading
the actual term lists above is required, not just comparing significant-term
counts.
"""

OUT_FINDINGS.write_text(findings, encoding="utf-8")
print(f"\nWrote {OUT_FINDINGS} (UTF-8 encoded)")