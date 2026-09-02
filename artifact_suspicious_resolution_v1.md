# Artifact-Suspicious Gene Resolution — v1

**Scope note:** no gene in switching_gene_pathway_context_v1.md currently carries an 'artifact-suspicious' verdict (n=1 gene, TET2, verdict='biologically plausible'). The real open item — per that document's own 'Open items carried forward' section — is verifying the call_gate() label-dependency caveat against TET2's specific SIMPLE_AND→INCONSISTENT call. This resolves that item directly, for TET2 and generically for any future gene in this table.

**Method — proof, not simulation:** call_gate() (verbatim, Day 5 / Day 14 Gate-Calling Algorithm, byte-identical in both) was checked exhaustively over all 64 binary combinations of its 6 boolean inputs. Result: 'INCONSISTENT' is produced in exactly 6/64 cases, and in every one of those cases the marks-only derivation (call_gate with every `expressed`-conditional branch collapsed to its marks-implied label) returns SIMPLE_AND or SIMPLE_OR — never anything else. Outside INCONSISTENT cases the two functions agree with zero exceptions. A third `expressed`-dependent branch in the original source is confirmed unreachable dead code. **Consequence: any gene whose call_gate() output is 'INCONSISTENT' is provably explained by the documented structural label dependency (call_gate_determinism_resolution_v1.md), independent of that gene's specific mark values.**

## TET2

Document verdict: **Biologically plausible** | Reported direction: SIMPLE_AND → INCONSISTENT | Driver status: Known AML driver

Reported direction contains 'INCONSISTENT' (1 of 2 side(s)). By the exhaustive proof above, this call is **provably resolved-by-known-cause** — no other code path in call_gate() can produce 'INCONSISTENT'. This independently confirms the caveat flagged as unchecked in switching_gene_pathway_context_v1.md's Open Items.

**Confirmatory data check (optional, non-blocking):**

- GM12878: structural='SIMPLE_AND', marks-only='SIMPLE_AND' (marks: {'H3K4me3': 1.0, 'H3K27ac': 1.0, 'H3K4me1': 0.0, 'H3K27me3': 0.0, 'H3K9me3': 0.0}, expressed=1.0)

- K562: structural='INCONSISTENT', marks-only='SIMPLE_AND' (marks: {'H3K4me3': 1, 'H3K27ac': 1, 'H3K4me1': 0, 'H3K27me3': 0, 'H3K9me3': 0}, expressed=0.0)


## Summary counts

| classification | count |
|---|---|
| resolved-by-known-cause | 1 |
| open-with-cause-still-unidentified | 0 |


0 of 1 genes remain open. Per standing rule, none should be fixed or reframed until a cause is named at file/line level.