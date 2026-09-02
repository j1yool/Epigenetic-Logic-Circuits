"""
fetch_genome_pool_from_gtf_v1.py

Replaces Step 4 of build_clustering_control_v2.py. NAMED CAUSE: Ensembl's
BioMart martservice web endpoint returned 405 on the first attempt (wrong
HTTP method -- fixed in v2) and then "Service unavailable" (Ensembl's own
status page) on the second -- a live infrastructure problem with the web
service itself, not a request-shape bug this script can work around.
BioMart's web service has a known reputation for exactly this kind of
flakiness under scripted/bulk access.

FIX: read Ensembl's GRCh38 GTF annotation file directly -- a static,
versioned download, not a live query service. Far more robust for a
one-time bulk pull of ~20,000 gene records. Edward downloads the file by
hand from https://ftp.ensembl.org/pub/current_gtf/homo_sapiens/ (that
domain is unreachable from this sandbox too, and "current release number"
changes over time, so this has to be a real download rather than a
hardcoded URL).

INPUT: any Homo_sapiens.GRCh38.<release>.gtf.gz file in data/ (auto-
discovered by pattern -- the exact release number doesn't matter to this
script and isn't hardcoded anywhere).

GTF SCHEMA (confirmed against Ensembl's documented GTF2.2 spec, and
self-tested against a small hand-built mock file before touching the real
one): tab-separated columns
    seqname, source, feature, start, end, score, strand, frame, attribute
The attribute field is semicolon-separated `key "value"` pairs, e.g.:
    gene_id "ENSG00000141510"; gene_name "TP53"; gene_biotype "protein_coding";
This script extracts rows where feature == "gene" and gene_biotype ==
"protein_coding", restricted to real chromosomes (1-22, X, Y -- GTF
scaffold/patch contigs like "GL000009.2" or "KI270728.1" are excluded,
confirmed via the same VALID_CHROMS set used in build_clustering_control_v2.py
for the KRAB-ZNF coordinate table, so both gene sets are filtered
identically). Strand is converted from GTF's '+'/'-' to the same 1/-1
numeric convention already used in krab_znf_coordinates_v1.csv.

OUTPUT: data/full_genome_gene_pool_v1.csv
    columns: gene_symbol, chr, start, end, strand
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import glob
import gzip
import re
import pandas as pd

DATA_DIR = "data"
OUTPUT_PATH = os.path.join(DATA_DIR, "full_genome_gene_pool_v1.csv")
VALID_CHROMS = {str(i) for i in range(1, 23)} | {"X", "Y"}

ATTR_PATTERN = re.compile(r'(\w+)\s+"([^"]*)"')


def find_gtf_file():
    matches = sorted(glob.glob(os.path.join(DATA_DIR, "Homo_sapiens.GRCh38.*.gtf.gz")))
    # Explicitly exclude the larger variants we don't want, in case one
    # was downloaded by mistake alongside or instead of the plain file.
    matches = [m for m in matches if ".chr." not in m and "abinitio" not in m]
    if not matches:
        sys.exit(
            f"FATAL: no file matching 'Homo_sapiens.GRCh38.*.gtf.gz' found in {DATA_DIR}/.\n"
            f"Browse https://ftp.ensembl.org/pub/, open the highest-numbered release-NNN/ "
            f"folder, then gtf/homo_sapiens/, and download the plain "
            f"Homo_sapiens.GRCh38.<NNN>.gtf.gz file (not '.chr.' or 'abinitio'). "
            f"Place it in {DATA_DIR}/."
        )
    if len(matches) > 1:
        print(f"NOTE: multiple GTF files found: {matches}. Using {matches[-1]}.")
    return matches[-1]


def parse_attributes(attr_str):
    """Parses GTF's semicolon-separated key "value" attribute field into a
    dict. Confirmed against Ensembl's GTF2.2 spec and the mock self-test
    file before being used on the real download."""
    return dict(ATTR_PATTERN.findall(attr_str))


MOCK_GTF_CONTENT = """##description: fake tiny GTF for self-test only, not real data, embedded directly in this script
1	havana	gene	1000000	1050000	.	+	.	gene_id "ENSG00000000001"; gene_version "1"; gene_name "FAKEGENE1"; gene_source "havana"; gene_biotype "protein_coding";
1	havana	transcript	1000000	1050000	.	+	.	gene_id "ENSG00000000001"; transcript_id "ENST00000000001";
19	ensembl	gene	2000000	2010000	.	-	.	gene_id "ENSG00000000002"; gene_version "1"; gene_name "FAKEGENE2"; gene_source "ensembl"; gene_biotype "protein_coding";
19	ensembl	gene	3000000	3010000	.	+	.	gene_id "ENSG00000000003"; gene_version "1"; gene_name "FAKEGENE3"; gene_source "ensembl"; gene_biotype "lncRNA";
GL000009.2	havana	gene	500	600	.	+	.	gene_id "ENSG00000000004"; gene_version "1"; gene_name "FAKEGENE4"; gene_source "havana"; gene_biotype "protein_coding";
"""


def parse_gtf_lines(lines):
    """Same logic as parse_gtf(), but takes an iterable of lines directly
    instead of a file path -- lets the self-test run against an in-memory
    string with zero external file dependency."""
    records = []
    n_total_gene_rows = 0
    n_missing_attrs = 0
    for line in lines:
        if not line or line.startswith("#"):
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) != 9:
            continue
        seqname, source, feature, start, end, score, strand, frame, attr_str = fields
        if feature != "gene":
            continue
        n_total_gene_rows += 1
        attrs = parse_attributes(attr_str)
        gene_name = attrs.get("gene_name")
        gene_biotype = attrs.get("gene_biotype")
        if gene_name is None or gene_biotype is None:
            n_missing_attrs += 1
            continue
        if gene_biotype != "protein_coding":
            continue
        if seqname not in VALID_CHROMS:
            continue
        records.append({
            "gene_symbol": gene_name,
            "chr": seqname,
            "start": int(start),
            "end": int(end),
            "strand": 1 if strand == "+" else -1 if strand == "-" else 0,
        })
    if n_missing_attrs:
        print(f"NOTE: {n_missing_attrs}/{n_total_gene_rows} 'gene' rows missing "
              f"gene_name or gene_biotype attribute -- excluded, not silently kept.")
    return pd.DataFrame.from_records(records), n_total_gene_rows


def parse_gtf(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return parse_gtf_lines(f)


# =============================================================================
# SELF-TEST against the mock GTF (built to exercise every filter branch:
# a real protein-coding gene, a non-'gene' feature row, a non-protein-coding
# biotype, and a scaffold contig -- all must be excluded except the two
# real protein-coding gene rows).
# =============================================================================

def self_test_gtf_parsing():
    df, n_total = parse_gtf_lines(MOCK_GTF_CONTENT.splitlines())

    assert n_total == 4, f"FATAL: expected 4 'gene'-feature rows in mock, found {n_total}"
    assert len(df) == 2, f"FATAL: expected 2 rows to survive filtering (protein_coding, real chrom), got {len(df)}"
    assert set(df["gene_symbol"]) == {"FAKEGENE1", "FAKEGENE2"}, (
        f"FATAL: wrong genes survived filtering. Expected FAKEGENE1/FAKEGENE2, got {set(df['gene_symbol'])}. "
        f"(FAKEGENE3 is lncRNA, must be excluded; FAKEGENE4 is on scaffold GL000009.2, must be excluded.)"
    )
    row1 = df[df["gene_symbol"] == "FAKEGENE1"].iloc[0]
    assert row1["chr"] == "1" and row1["start"] == 1000000 and row1["end"] == 1050000 and row1["strand"] == 1, (
        f"FATAL: FAKEGENE1 fields wrong: {row1.to_dict()}"
    )
    row2 = df[df["gene_symbol"] == "FAKEGENE2"].iloc[0]
    assert row2["chr"] == "19" and row2["strand"] == -1, f"FATAL: FAKEGENE2 fields wrong: {row2.to_dict()}"

    print("Self-test PASSED (GTF parsing): correctly keeps protein-coding genes on real "
          "chromosomes, correctly excludes non-'gene' rows, non-protein-coding biotypes, "
          "and scaffold contigs. Strand conversion (+/- -> 1/-1) verified.")


def run_real_data():
    gtf_path = find_gtf_file()
    print(f"Parsing {gtf_path} ...")
    df, n_total_gene_rows = parse_gtf(gtf_path)
    print(f"Total 'gene'-feature rows in GTF: {n_total_gene_rows}")
    print(f"Protein-coding, real-chromosome genes retained: {len(df)}")

    if len(df) < 15000 or len(df) > 25000:
        print(f"\n*** NAMED ANOMALY, NOT SILENTLY ACCEPTED ***")
        print(f"Retained gene count ({len(df)}) is outside the expected range [15000, 25000] "
              f"for GRCh38 protein-coding genes (~19,000-20,000 typically annotated). "
              f"Not proceeding -- inspect the GTF file and this script's filtering logic "
              f"before trusting the output.")
        sys.exit(1)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nGene count within expected range. Wrote {OUTPUT_PATH}.")

    # Quick cross-check: how many of the 373 real KRAB-ZNF gene symbols
    # appear in this pool? They SHOULD mostly overlap (KRAB-ZNFs are
    # protein-coding), which is a useful sanity signal that this table and
    # the HGNC-derived KRAB-ZNF table are using compatible gene symbols.
    krab_path = os.path.join(DATA_DIR, "krab_znf_coordinates_v1.csv")
    if os.path.exists(krab_path):
        krab_symbols = set(pd.read_csv(krab_path)["gene_symbol"])
        overlap = krab_symbols & set(df["gene_symbol"])
        print(f"\nCross-check: {len(overlap)}/{len(krab_symbols)} KRAB-ZNF gene symbols "
              f"found in this genome-wide pool (expect most, since KRAB-ZNFs are "
              f"protein-coding; a low overlap would indicate a symbol-convention "
              f"mismatch between the HGNC-derived and GTF-derived tables).")
        missing = krab_symbols - set(df["gene_symbol"])
        if missing:
            print(f"  {len(missing)} KRAB-ZNF symbols NOT found in the genome pool "
                  f"(logged, not silently dropped): {sorted(missing)[:20]}"
                  f"{' ...' if len(missing) > 20 else ''}")


if __name__ == "__main__":
    self_test_gtf_parsing()
    print("\nSelf-test passed. Proceeding to real-data run.\n")
    run_real_data()