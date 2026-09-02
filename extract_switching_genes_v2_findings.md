# Switching Gene Extraction v2 — Findings (Day 47)

## Root cause (final)
`gate_assignments.tsv` had a genuinely empty `gate_type` cell on disk for
34,163 rows (confirmed by raw text inspection: `'ENSG00000000003\t\t-1.0...'`
— two adjacent tabs, not the text "NULL"). This is NOT a read-time parsing
issue in today's scripts. It is upstream corruption: some prior script read
this file with pandas' default NA handling (which treats the string "NULL"
as missing), then wrote the DataFrame back out with `to_csv()`, which
serializes NaN as an empty string by default. The original "NULL" text was
permanently lost from that file at that point.

## Fix
`gate_type` and `complexity_score` were regenerated directly from
`binary_matrix.csv` / `k562_binary_matrix.csv` (source of truth, confirmed
intact — no NaNs in mark columns) using `call_gate()` / 
`compute_complexity_score()` copied verbatim from the Day 5 / Day 14
scripts. Only `gene_name` / `gene_id_clean` annotation was borrowed from the
old TSVs. Result: 0 NaN gate_type values in either regenerated file.

## Output
- `gate_assignments_regenerated_v1.tsv`, `k562_gate_assignments_regenerated_v1.tsv`
  — these should replace the originals as the trusted source going forward.
- `switching_genes_gm12878_v1.txt`, `switching_genes_k562_v1.txt` — 17633 genes each
- `switching_genes_merged_v1.tsv` — full per-gene, per-cell-line gate_type + and_origin

## Excluded genes (unknown expression status)
- K562: 4436 genes excluded

## Counts vs. Day 38 document
Total switchers: 17633 (Day 38 doc: 6,962)
AND<->INC switchers: 3690 (Day 38 doc: 3,502)

If these don't match, the Day 38 figures should be treated as suspect and
re-derived from these clean regenerated files, not reconciled by adjusting
today's numbers to match.

## Action item for MASTER_STATUS.md / RESEARCH_LOG.md
gate_assignments.tsv and k562_gate_assignments_named.tsv are flagged
CORRUPTED (gate_type column) and should not be read directly by any future
script. Point all downstream Logic Circuits work at the regenerated files
above instead.
