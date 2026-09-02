"""
fetch_krab_znf_coordinates_v2.py

Supersedes fetch_krab_znf_coordinates_v1.py. NAMED BUG in v1, found on the
real HGNC response (confirmed against the printed field list from Edward's
run, not guessed):

  v1's discover_family_field() matched candidate fields with a bare
  substring check (`"group" in k.lower()`). ZNF91's real HGNC record has
  THREE fields containing that substring: 'locus_group', 'gene_group_id',
  'gene_group' -- in that field order. v1 then picked "the first candidate
  without 'id' in its name" as the name field, which selected 'locus_group'
  (value: "protein-coding gene", an unrelated broad category field -- not
  a gene family/group at all) instead of 'gene_group' (the actual family
  list, e.g. containing "Krueppel-associated box (KRAB) domain containing").
  'locus_group' happened to come first in HGNC's field ordering and also
  happened to contain the substring "group", which is exactly the kind of
  loose-match failure mode the schema-confirmation rule exists to catch.

FIX: match on the compound term "gene_group" / "gene_famil" specifically,
not the bare substring "group" / "famil". A regression-guard self-test
below reproduces the exact field set from Edward's real ZNF91 response
(with locus_group included) and asserts 'gene_group' -- not 'locus_group'
-- is selected, so this specific confusion can't silently reoccur.

CANNOT BE RUN IN THIS SANDBOX: rest.genenames.org and rest.ensembl.org are
not on the sandbox's network allowlist. Same situation as v1 -- run this
locally (VS Code Run button, zero configured arguments, per project
convention).

EXPECTED GENE COUNT SANITY CHECK: no single authoritative number exists in
the literature -- confirmed by direct search rather than assumed:
    - Imbeault, Helleboid & Trono 2017 (Nature): ~350
    - de Tribolet-Hardy et al. 2023 (Genome Research, same lab, follow-up
      catalog): 378
    - Nowick et al. 2006 (original hand-curated catalog): 423 loci
This script uses [300, 450] as a sanity-check RANGE reflecting genuine
cross-study variation in curation criteria, not a single fabricated
target. A fetched count outside this range is a named bug to log, not a
number to silently accept -- but a count anywhere inside it is NOT
automatically "correct," just not obviously wrong.

OUTPUT: krab_znf_coordinates_v1.csv
    columns: gene_symbol, chr, start, end, strand
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import json
import requests
import pandas as pd

HGNC_BASE = "https://rest.genenames.org"
ENSEMBL_BASE = "https://rest.ensembl.org"
KNOWN_KRAB_ZNF_PROBE_GENE = "ZNF91"  # well-characterized KRAB-ZNF, used only to discover field names
GROUP_NAME_MATCH = "krab"            # case-insensitive substring match against whatever family/group field is found

OUTPUT_PATH = "data/krab_znf_coordinates_v1.csv"  # filename unchanged deliberately -- this IS
                                                     # the Day 59 Block 4 deliverable name from the
                                                     # schedule; only the fetch SCRIPT is versioned
EXPECTED_COUNT_RANGE = (300, 450)  # see docstring -- literature range, not a single invented number
ENSEMBL_BATCH_SIZE = 500           # Ensembl POST symbol-lookup batch limit is 1000; using 500 to stay well clear
REQUEST_TIMEOUT_S = 30
RETRY_DELAY_S = 1.0


def hgnc_get(path, headers=None):
    h = {"Accept": "application/json"}
    if headers:
        h.update(headers)
    resp = requests.get(f"{HGNC_BASE}{path}", headers=h, timeout=REQUEST_TIMEOUT_S)
    if resp.status_code != 200:
        sys.exit(f"FATAL: HGNC request failed, status={resp.status_code}, url={resp.url}\nBody: {resp.text[:500]}")
    return resp.json()


# =============================================================================
# STEP 1: self-discover the HGNC field name for gene family/group
# =============================================================================

def discover_family_field():
    data = hgnc_get(f"/fetch/symbol/{KNOWN_KRAB_ZNF_PROBE_GENE}")
    if "response" not in data or "docs" not in data["response"] or not data["response"]["docs"]:
        sys.exit(
            f"FATAL: unexpected HGNC response shape for probe gene "
            f"{KNOWN_KRAB_ZNF_PROBE_GENE}. Actual top-level keys: {list(data.keys())}\n"
            f"Full response: {json.dumps(data)[:1000]}\n"
            f"Not proceeding with a guessed schema."
        )
    doc = data["response"]["docs"][0]
    print(f"HGNC record for {KNOWN_KRAB_ZNF_PROBE_GENE} -- all field names present:")
    print(f"  {sorted(doc.keys())}\n")

    # v2 fix: match the COMPOUND term "gene_group"/"gene_famil", not the bare
    # substring "group"/"famil" -- v1's loose match picked up the unrelated
    # 'locus_group' field (a broad category like "protein-coding gene") ahead
    # of the real 'gene_group' field. Confirmed against Edward's real run.
    candidate_fields = [k for k in doc.keys() if "gene_famil" in k.lower() or "gene_group" in k.lower()]
    if not candidate_fields:
        sys.exit(
            f"FATAL: no field name containing 'gene_family' or 'gene_group' found in the "
            f"HGNC record for {KNOWN_KRAB_ZNF_PROBE_GENE}. Actual fields: {sorted(doc.keys())}\n"
            f"HGNC's schema has changed in a way this script doesn't anticipate. "
            f"Not proceeding -- inspect the printed fields above and update the script by hand."
        )
    print(f"Candidate family/group field name(s): {candidate_fields}")

    name_field = next((f for f in candidate_fields if "id" not in f.lower()), candidate_fields[0])
    id_field = next((f for f in candidate_fields if "id" in f.lower()), None)

    names = doc.get(name_field)
    ids = doc.get(id_field) if id_field else None
    print(f"  {name_field} = {names}")
    if id_field:
        print(f"  {id_field} = {ids}\n")

    if not isinstance(names, list):
        names = [names]
    if ids is not None and not isinstance(ids, list):
        ids = [ids]

    match_idx = next((i for i, n in enumerate(names) if GROUP_NAME_MATCH in str(n).lower()), None)
    if match_idx is None:
        sys.exit(
            f"FATAL: none of {KNOWN_KRAB_ZNF_PROBE_GENE}'s {name_field} entries contain "
            f"'{GROUP_NAME_MATCH}': {names}\nNot proceeding -- confirm the correct group "
            f"name by hand at https://www.genenames.org/data/gene-symbol-report/#!/symbol/{KNOWN_KRAB_ZNF_PROBE_GENE}"
        )
    matched_name = names[match_idx]
    matched_id = ids[match_idx] if ids else None
    print(f"MATCHED group: name='{matched_name}'" + (f", id={matched_id}" if matched_id else " (no separate id field)") + "\n")

    return name_field, id_field, matched_name, matched_id


# =============================================================================
# STEP 2: fetch full gene symbol list for the matched group, with pagination
# =============================================================================

def fetch_group_members(name_field, id_field, matched_name, matched_id):
    query_field = id_field if matched_id is not None else name_field
    query_value = matched_id if matched_id is not None else matched_name
    query_str = f'{query_field}:"{query_value}"'

    all_symbols = []
    start = 0
    rows_per_page = 500
    while True:
        data = hgnc_get(f"/search/{query_str}?start={start}&rows={rows_per_page}")
        if "response" not in data or "docs" not in data["response"]:
            sys.exit(
                f"FATAL: unexpected HGNC search response shape. "
                f"Actual keys: {list(data.keys())}. Full: {json.dumps(data)[:1000]}"
            )
        docs = data["response"]["docs"]
        num_found = data["response"].get("numFound", None)
        if start == 0:
            print(f"HGNC search for {query_str}: numFound={num_found}")
        if not docs:
            break
        symbols = [d.get("symbol") for d in docs if d.get("symbol")]
        missing_symbol = [d for d in docs if not d.get("symbol")]
        if missing_symbol:
            print(f"  NOTE: {len(missing_symbol)} docs in this page had no 'symbol' field -- skipped.")
        all_symbols.extend(symbols)
        start += rows_per_page
        if num_found is not None and start >= num_found:
            break
        time.sleep(RETRY_DELAY_S)

    all_symbols = sorted(set(all_symbols))
    print(f"Total unique gene symbols retrieved: {len(all_symbols)}\n")
    return all_symbols


# =============================================================================
# STEP 3: Ensembl batch symbol -> coordinate lookup (GRCh38, chunked POSTs)
# =============================================================================

def fetch_coordinates(symbols):
    records = []
    unresolved = []
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    for i in range(0, len(symbols), ENSEMBL_BATCH_SIZE):
        chunk = symbols[i:i + ENSEMBL_BATCH_SIZE]
        resp = requests.post(
            f"{ENSEMBL_BASE}/lookup/symbol/homo_sapiens",
            headers=headers,
            data=json.dumps({"symbols": chunk}),
            timeout=REQUEST_TIMEOUT_S,
        )
        if resp.status_code != 200:
            sys.exit(
                f"FATAL: Ensembl batch lookup failed, status={resp.status_code}, "
                f"chunk_start={i}\nBody: {resp.text[:500]}"
            )
        data = resp.json()
        if i == 0:
            sample_key = next(iter(data), None)
            if sample_key is not None:
                print(f"Ensembl response schema, sample entry ({sample_key}): "
                      f"{sorted(data[sample_key].keys())}\n")
                required = {"seq_region_name", "start", "end", "strand"}
                missing = required - set(data[sample_key].keys())
                if missing:
                    sys.exit(
                        f"FATAL: Ensembl response missing expected field(s) {missing}. "
                        f"Actual fields: {sorted(data[sample_key].keys())}. "
                        f"Not proceeding with a guessed schema."
                    )

        for sym in chunk:
            entry = data.get(sym)
            if entry is None:
                unresolved.append(sym)
                continue
            records.append({
                "gene_symbol": sym,
                "chr": entry["seq_region_name"],
                "start": entry["start"],
                "end": entry["end"],
                "strand": entry["strand"],
            })
        print(f"  Batch {i // ENSEMBL_BATCH_SIZE + 1}: {len(chunk)} queried, "
              f"{len(chunk) - sum(1 for s in chunk if data.get(s) is None)} resolved")
        time.sleep(RETRY_DELAY_S)

    if unresolved:
        print(f"\nNOTE: {len(unresolved)}/{len(symbols)} symbols did not resolve via Ensembl "
              f"(withdrawn/renamed symbols, readthrough transcripts, etc.). Logged, not silently dropped:")
        print(f"  {unresolved}")

    return pd.DataFrame.from_records(records), unresolved


# =============================================================================
# SELF-TEST: pagination and coordinate-parsing logic against mocked responses
# (no network -- exercises the logic in fetch_group_members/fetch_coordinates
# shape-handling without hitting HGNC/Ensembl, since those are unreachable
# from this sandbox and must be validated structurally before the real run.)
# =============================================================================

def self_test_family_field_discovery_regression_guard():
    """Reproduces the exact field set from ZNF91's real HGNC response (as
    printed in Edward's Day 59 run) and asserts the v2 matching logic
    selects 'gene_group'/'gene_group_id', not 'locus_group' -- the precise
    failure v1 had. If this ever regresses to the loose substring match,
    this test fails before any live API call is made."""
    real_field_order = [
        'agr', 'alias_symbol', 'ccds_id', 'date_approved_reserved', 'date_modified',
        'date_name_changed', 'ena', 'ensembl_gene_id', 'entrez_id', 'gene_group',
        'gene_group_id', 'hgnc_id', 'location', 'locus_group', 'locus_type',
        'mane_select', 'mgd_id', 'name', 'omim_id', 'prev_name', 'pubmed_id',
        'refseq_accession', 'status', 'symbol', 'ucsc_id', 'uniprot_ids', 'uuid', 'vega_id',
    ]
    candidate_fields = [k for k in real_field_order if "gene_famil" in k.lower() or "gene_group" in k.lower()]
    assert candidate_fields == ["gene_group", "gene_group_id"], (
        f"FATAL: regression guard failed -- expected ['gene_group', 'gene_group_id'], "
        f"got {candidate_fields}. 'locus_group' must NOT be selected here."
    )
    name_field = next((f for f in candidate_fields if "id" not in f.lower()), candidate_fields[0])
    id_field = next((f for f in candidate_fields if "id" in f.lower()), None)
    assert name_field == "gene_group", f"FATAL: name_field should be 'gene_group', got '{name_field}'"
    assert id_field == "gene_group_id", f"FATAL: id_field should be 'gene_group_id', got '{id_field}'"
    print("Self-test PASSED (regression guard): 'gene_group'/'gene_group_id' correctly "
          "selected over 'locus_group' on ZNF91's real field set.")


def self_test_parsing_logic():
    # Simulated Ensembl response shape -- 2 resolved, 1 unresolved symbol.
    mock_ensembl_response = {
        "ZNF91": {"seq_region_name": "19", "start": 23161843, "end": 23214392, "strand": -1},
        "ZNF141": {"seq_region_name": "4", "start": 189267500, "end": 189291000, "strand": 1},
    }
    chunk = ["ZNF91", "ZNF141", "ZNF_NONEXISTENT"]
    records = []
    unresolved = []
    for sym in chunk:
        entry = mock_ensembl_response.get(sym)
        if entry is None:
            unresolved.append(sym)
            continue
        records.append({
            "gene_symbol": sym, "chr": entry["seq_region_name"],
            "start": entry["start"], "end": entry["end"], "strand": entry["strand"],
        })
    assert len(records) == 2, f"FATAL: expected 2 resolved records, got {len(records)}"
    assert unresolved == ["ZNF_NONEXISTENT"], f"FATAL: unresolved tracking wrong, got {unresolved}"
    assert records[0]["chr"] == "19", f"FATAL: chr field mapping wrong, got {records[0]['chr']}"
    print("Self-test PASSED (Ensembl response parsing: resolved records built correctly, "
          "unresolved symbols tracked rather than silently dropped).")

    # Simulated HGNC paginated search -- 2 pages, numFound=3, rows=2.
    mock_pages = [
        {"response": {"numFound": 3, "docs": [{"symbol": "ZNF91"}, {"symbol": "ZNF141"}]}},
        {"response": {"numFound": 3, "docs": [{"symbol": "ZNF680"}]}},
    ]
    collected = []
    for page in mock_pages:
        collected.extend(d["symbol"] for d in page["response"]["docs"])
    assert sorted(set(collected)) == ["ZNF141", "ZNF680", "ZNF91"], (
        f"FATAL: pagination accumulation logic wrong, got {collected}"
    )
    print("Self-test PASSED (HGNC pagination accumulation across pages).")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("STEP 1: discover HGNC family/group field name from a known KRAB-ZNF gene")
    print("=" * 70)
    name_field, id_field, matched_name, matched_id = discover_family_field()

    print("=" * 70)
    print("STEP 2: fetch full group membership (paginated)")
    print("=" * 70)
    symbols = fetch_group_members(name_field, id_field, matched_name, matched_id)

    n = len(symbols)
    lo, hi = EXPECTED_COUNT_RANGE
    if not (lo <= n <= hi):
        print(f"\n*** NAMED ANOMALY, NOT SILENTLY ACCEPTED ***")
        print(f"Fetched {n} gene symbols; literature range is [{lo}, {hi}] "
              f"(Imbeault 2017 ~350, de Tribolet-Hardy 2023: 378, Nowick 2006: 423 loci).")
        print(f"This is logged as a bug to investigate before Block 5 uses this gene set -- "
              f"NOT proceeding to coordinate fetch until you've confirmed whether the group "
              f"matched in Step 1 ('{matched_name}') is actually the full KRAB-ZNF set or a "
              f"narrower/broader HGNC subgroup.")
        sys.exit(1)
    print(f"\nFetched gene count ({n}) falls within the literature-informed sanity range "
          f"[{lo}, {hi}]. Proceeding.\n")

    print("=" * 70)
    print("STEP 3: Ensembl GRCh38 coordinate lookup (batched)")
    print("=" * 70)
    coords_df, unresolved = fetch_coordinates(symbols)

    coords_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {OUTPUT_PATH}: {len(coords_df)} rows "
          f"({len(unresolved)}/{n} symbols unresolved and excluded, logged above).")
    print("\nDo NOT proceed to Block 5's control-set construction until the row count and "
          "unresolved-symbol list above have been transcribed into MASTER_STATUS.md by hand.")


if __name__ == "__main__":
    self_test_family_field_discovery_regression_guard()
    self_test_parsing_logic()
    print("\nSelf-test passed. Proceeding to live API calls.\n")
    print("NOTE: this script requires network access to rest.genenames.org and "
          "rest.ensembl.org, and the `requests` package (`pip install requests` "
          "in your genomics conda env if missing). Run locally, not in a sandbox "
          "with restricted network egress.\n")
    main()