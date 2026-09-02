"""
extract_genes_from_gtf_v1.py

PURPOSE (Day 60, LC, Block 1 -- prerequisite for the diff script)
-------------------------------------------------------------------
Parses the raw Ensembl GTF (Homo_sapiens_GRCh38_116_gtf.gz) into a clean,
diffable CSV: one row per gene-feature line, with gene_symbol / chr / start /
end / strand / biotype / gene_id columns matching the BioMart export schema.

The GTF is a 9-column tab-delimited file where feature-type rows include
"gene", "transcript", "exon", etc. Only "gene" rows are extracted here.
Gene-level metadata (biotype, name) lives in a semicolon-delimited
attribute string in column 9 (e.g. gene_id "ENSG..."; gene_biotype "...";
gene_name "...";) -- gene_name is NOT always present.

This script streams the file line-by-line rather than loading it into
memory at once (decompressed size is ~365MB), and logs every filtering
decision explicitly rather than silently dropping rows.

Output: full_genome_gene_pool_gtf_v1.csv
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import csv
import gzip
import re

GTF_PATH = r"C:\Users\jamoo\Downloads\LOGIC CIRCUITS\Homo_sapiens.GRCh38.116.gtf"
OUT_PATH = "full_genome_gene_pool_gtf_v1.csv"

GZIP_MAGIC = b"\x1f\x8b"


def open_gtf(path: str):
    """
    Open a GTF file for text-mode streaming, auto-detecting whether it's
    still gzip-compressed or has already been decompressed on disk --
    rather than assuming from the filename, since a .gtf file with no
    .gz extension (like this one) can still be gzipped, and vice versa.
    """
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == GZIP_MAGIC:
        return gzip.open(path, "rt")
    return open(path, "rt")

STANDARD_CHROMS = set([str(i) for i in range(1, 23)] + ["X", "Y", "MT"])

ATTR_RE = re.compile(r'(\w+) "([^"]*)"')


def parse_attributes(attr_field: str) -> dict:
    """Parse the semicolon-delimited GTF attribute string into a dict."""
    return dict(ATTR_RE.findall(attr_field))


def process_gene_line(fields: list) -> dict:
    """
    Given the 9 tab-split fields of a GTF line already confirmed to be a
    'gene' feature row, return a normalized row dict.
    Returns None if the chromosome is not in STANDARD_CHROMS (caller counts these).
    """
    chrom, source, feature, start, end, score, strand, frame, attr_field = fields
    if chrom not in STANDARD_CHROMS:
        return None
    attrs = parse_attributes(attr_field)
    return {
        "gene_symbol": attrs.get("gene_name", "").strip(),
        "chr": chrom,
        "start": int(start),
        "end": int(end),
        "strand": "1" if strand == "+" else "-1" if strand == "-" else strand,
        "biotype": attrs.get("gene_biotype", "").strip(),
        "gene_id": attrs.get("gene_id", "").strip(),
    }


def process_lines(lines):
    """
    Core extraction loop, factored out so synthetic_self_test can feed it
    an in-memory list of GTF-format strings instead of reading from disk.

    Returns (rows, n_gene_lines, n_nonstandard_chrom, n_missing_symbol)
    """
    rows = []
    n_gene_lines = 0
    n_nonstandard_chrom = 0
    n_missing_symbol = 0

    for line in lines:
        if not line or line.startswith("#"):
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) != 9:
            continue
        if fields[2] != "gene":
            continue
        n_gene_lines += 1
        row = process_gene_line(fields)
        if row is None:
            n_nonstandard_chrom += 1
            continue
        if row["gene_symbol"] == "":
            n_missing_symbol += 1
        rows.append(row)

    return rows, n_gene_lines, n_nonstandard_chrom, n_missing_symbol


def synthetic_self_test():
    """
    Known-ground-truth fixture covering:
      - 1 standard-chrom gene WITH gene_name (should extract cleanly)
      - 1 standard-chrom gene WITHOUT gene_name (should extract with empty symbol, counted)
      - 1 non-standard-chrom (patch) gene (should be excluded, counted)
      - 1 transcript line and 1 exon line (should be ignored entirely -- not gene features)
      - 1 comment line (should be ignored)
    """
    toy_lines = [
        "#!genome-build GRCh38.p14",
        '1\thavana\tgene\t1000\t2000\t.\t+\t.\tgene_id "ENSG00000000001"; gene_version "1"; gene_name "TESTGENE1"; gene_source "havana"; gene_biotype "protein_coding";',
        '1\thavana\ttranscript\t1000\t2000\t.\t+\t.\tgene_id "ENSG00000000001"; transcript_id "ENST00000000001"; gene_name "TESTGENE1"; gene_biotype "protein_coding"; transcript_biotype "protein_coding";',
        '1\thavana\texon\t1000\t1500\t.\t+\t.\tgene_id "ENSG00000000001"; transcript_id "ENST00000000001"; exon_number "1"; gene_name "TESTGENE1"; gene_biotype "protein_coding"; exon_id "ENSE00000000001";',
        '5\thavana\tgene\t3000\t4000\t.\t-\t.\tgene_id "ENSG00000000002"; gene_version "1"; gene_source "havana"; gene_biotype "lncRNA";',
        'HG1012_PATCH\thavana\tgene\t500\t900\t.\t+\t.\tgene_id "ENSG00000000003"; gene_version "1"; gene_name "TESTGENE3"; gene_source "havana"; gene_biotype "protein_coding";',
    ]

    rows, n_gene_lines, n_nonstandard, n_missing_symbol = process_lines(toy_lines)

    assert n_gene_lines == 3, f"[SELF-TEST FAILED] Expected 3 gene-feature lines, got {n_gene_lines}"
    assert n_nonstandard == 1, f"[SELF-TEST FAILED] Expected 1 non-standard-chrom exclusion, got {n_nonstandard}"
    assert n_missing_symbol == 1, f"[SELF-TEST FAILED] Expected 1 missing-symbol gene, got {n_missing_symbol}"
    assert len(rows) == 2, f"[SELF-TEST FAILED] Expected 2 rows in final output (patch excluded), got {len(rows)}"

    row1 = next(r for r in rows if r["gene_id"] == "ENSG00000000001")
    assert row1["gene_symbol"] == "TESTGENE1"
    assert row1["chr"] == "1"
    assert row1["start"] == 1000
    assert row1["end"] == 2000
    assert row1["strand"] == "1"
    assert row1["biotype"] == "protein_coding"

    row2 = next(r for r in rows if r["gene_id"] == "ENSG00000000002")
    assert row2["gene_symbol"] == "", "Missing gene_name should yield empty string, not a crash or a fabricated value"
    assert row2["strand"] == "-1"
    assert row2["biotype"] == "lncRNA"

    print("[SELF-TEST PASSED] GTF gene-line extraction, standard-chrom filtering, and missing-symbol handling all correct.")


def main():
    synthetic_self_test()

    if not os.path.exists(GTF_PATH):
        sys.exit(
            f"[FILE NOT FOUND] {GTF_PATH}\n"
            f"Update GTF_PATH at the top of this script to the actual path of "
            f"Homo_sapiens_GRCh38_116_gtf.gz on your machine."
        )

    print(f"\nStreaming {GTF_PATH} (this is a ~365MB file uncompressed -- expect it to take a minute or two)...")

    with open_gtf(GTF_PATH) as f:
        rows, n_gene_lines, n_nonstandard, n_missing_symbol = process_lines(f)

    print(f"\nTotal gene-feature lines in GTF: {n_gene_lines}")
    print(f"Excluded (non-standard chromosome / scaffold / patch): {n_nonstandard}")
    print(f"Retained (standard chromosomes 1-22, X, Y, MT): {len(rows)}")
    print(f"  Of those, missing gene_name (empty gene_symbol): {n_missing_symbol}")

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["gene_symbol", "chr", "start", "end", "strand", "biotype", "gene_id"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {OUT_PATH} ({len(rows)} rows)")


if __name__ == "__main__":
    main()