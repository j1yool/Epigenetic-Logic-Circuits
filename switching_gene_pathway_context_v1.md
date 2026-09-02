This document does not close the switching-gene directionality investigation. It is scaffolding for the eventual paper's discussion section, not a revision of the finding in `switching_gene_validation_interpretation_v1.md`.

# Switching-Gene Pathway Context (v1)

## Scope and derivation

`switching_gene_validation_interpretation_v1.md` does not contain a per-gene list for the GM12878-vs-K562 directional inconsistency — that finding is reported at the cell-line/AUC-delta level (marks-only subset AUC: GM12878 +0.0552, K562 −0.1439), not attributed to individual genes. The document's own `[UNRESOLVED]` item — the per-cell-line split of the AND→INC category — was computed here as the basis for this file's gene list, per Edward's explicit direction.

**AND↔INC split result** (from `switching_genes_v1.tsv`, n=7,474 total switching genes):

- `switch_direction` is deterministically encoded GM12878→K562 for every row (0 exceptions).
- AND→INC (n=3,502): uniformly `SIMPLE_AND` in GM12878 → `INCONSISTENT` in K562.
- INC→AND (n=188): uniformly `INCONSISTENT` in GM12878 → `SIMPLE_AND` in K562.
- The 3,502-vs-188 asymmetry between these two directions is a candidate mechanical contributor to the marks-only AUC delta sign flip, independent of the AML question below. This asymmetry is noted here as context; it is not evaluated further in this file.

**AML driver cross-reference**: the 3,690 genes in the combined AND↔INC set (both directions) were checked against the static AML driver reference list (FLT3, NPM1, DNMT3A, IDH1, IDH2, TET2, RUNX1, CEBPA, KIT, WT1, ASXL1, NRAS, KRAS — not COSMIC Cancer Gene Census, per instruction to use a static list for this pass). One gene matched.

<!-- FLAG: This cross-reference used a 13-gene static AML driver list, not the COSMIC Cancer Gene Census snapshot already on file for Project VII (2020-12-03, 723 genes). A COSMIC-based pass would likely surface additional genes among the 3,690, including AML-adjacent (non-canonical-driver) pathway members not on the static list. Reserved for the designated AML extension phase (Day 76-90), not opened here. -->

## Gene-level table

| Gene | GM12878 → K562 | Driver status | Verdict |
|---|---|---|---|
| TET2 | SIMPLE_AND → INCONSISTENT | Known AML driver | Biologically plausible |

## Per-gene notes

### TET2 (ENSG00000168769)

**Driver status:** Known AML driver. TET2 is a canonical, frequently mutated gene in AML, MDS, and clonal hematopoiesis, and its native function — 5mC-to-5hmC conversion — makes it a direct epigenetic regulator rather than a downstream pathway member.

**Directional inconsistency assessment:** TET2 is called `SIMPLE_AND` (clean, mark-consistent logic) in GM12878, a lymphoblastoid line, and `INCONSISTENT` (mark/expression mismatch) in K562, a myeloid-lineage (CML-derived) line. TET2 dysregulation is specifically implicated in myeloid, not lymphoid, malignancy, so a call flip toward inconsistent regulatory logic in the myeloid-lineage line is directionally consistent with TET2's known disease biology — this is not the flip a technical artifact would be expected to produce at random. That said, `call_gate()`'s structural label-dependency issue (documented in `call_gate_determinism_resolution_v1.md`) applies to this call exactly as it applies to any other `INCONSISTENT` label, so the flip is consistent with a real biological signal but is not independently verified against that structural caveat here.

**Verdict: biologically plausible.**

## Open items carried forward

- [ ] AND↔INC 3,502-vs-188 asymmetry as a contributor to the marks-only AUC delta sign flip — noted, not evaluated.
- [ ] COSMIC Cancer Gene Census cross-reference against the full 3,690-gene AND↔INC set — reserved for AML extension phase (Day 76-90).
- [ ] `call_gate()` label-dependency caveat not independently re-checked against TET2's specific INCONSISTENT call.
