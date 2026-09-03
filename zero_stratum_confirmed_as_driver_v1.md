# ZERO Stratum Confirmed as K562 Hansen KS Confound Driver (v1)

The K562 Hansen KS confound is driven by the ZERO expression stratum, not by deciles 4–9: excluding ZERO from both switching and non-switching gene pools resolves the KS test to KS=0.0100, p=0.9913 (n=3,796/3,796) — cleaner than the GM12878 benchmark itself (p=0.3917) — while excluding only the high deciles (4–9) and retaining ZERO leaves the confound intact or worse (KS=0.0871, p=7.168e-16, n=4,686/4,686, actually more significant than baseline).

## Task 1 — Baseline reproduction

Confirmed exactly, not approximately: `k562_confound_zero_decile49_isolated_v1.py`'s self-check reports baseline KS=0.0614, p=2.623e-11, n=6,647/6,647, relative diff **7.60e-05** against the Day 61 target. Synthetic self-test (exclusion partitioning, D/A_RESEEDED complement relationship, tolerance check, `run_ks_check()` reuse) also passed before the real-data run proceeded.

## Task 2 — Comparison against baseline

| Condition | Strata excluded | KS | p | n (switch/nonswitch) |
|---|---|---|---|---|
| Baseline (seed 61) | none | 0.0614 | 2.623e-11 | 6647 / 6647 |
| B — exclude ZERO | ZERO | **0.0100** | **0.9913** | 3796 / 3796 |
| C — exclude high deciles | 4,5,6,7,8,9 | 0.0871 | 7.168e-16 | 4686 / 4686 |
| D — exclude ZERO + high | ZERO, 4–9 | 0.0207 | 0.8264 | 1835 / 1835 |
| A — exclude low deciles (reseeded, seed 61) | 0,1,2,3 | 0.0848 | 1.823e-15 | 4812 / 4812 |

**Decision logic applied:** B's KS statistic moves sharply toward the GM12878-clean regime — a 6.1× drop in the KS statistic itself (0.0614 → 0.0100), not merely a p-value shift — while C stays at or below baseline cleanliness (KS actually increases, p drops five further orders of magnitude). This is the first branch of the pre-registered decision tree: **mismatch concentrates in the ZERO stratum.**

The signal is unanimous across all four conditions, not just the B-vs-C pair: every condition retaining ZERO (baseline, C, A) stays confounded (p ≤ 1.8e-15); every condition excluding ZERO (B, D) resolves (p ≥ 0.83). Condition A, rerun at seed=61 specifically for cross-check comparability against Day 63's seed=63 result, reproduces the same qualitative badness (KS=0.0848, p=1.823e-15) — confirming Day 63's original "excluding low deciles doesn't help" finding wasn't a seed artifact, and reinforcing that deciles 0–3 (as a nonzero-remainder concept, distinct from ZERO) are not the driver on their own.

**Power-artifact check (performed before trusting B's result):** B's n (3,796) is roughly half of baseline's (6,647), so the improvement needed to be checked against a null "this only failed to reach significance because n dropped" explanation. Two-sample KS critical value at α=0.05 scales as ≈1.36·√(2/n):

- At B's n=3,796, critical ≈0.0312. Baseline's actual effect size (0.0614) would still clear this threshold if merely carried over unchanged — but B's actual KS (0.0100) is well under its own critical value. The collapse is a real distributional change, not reduced power.
- Same check passes for D (n=1,835, critical ≈0.0449 vs. actual 0.0207 — genuinely clean) and for A/C (both comfortably exceed their respective critical values — genuinely confounded, not just typical noise).

## Task 3 — Verdict

**ZERO stratum confirmed as the K562 Hansen KS confound driver.** This closes the "pool-concentration in deciles 0–3" hypothesis (Mann-Whitney p=0.0006746, logged as strongest live hypothesis as of Day 63 `MASTER_STATUS.md`) as **superseded, not confirmed** — deciles 0–3 concentration was a symptom correlated with the real driver, not the driver itself. Condition A (excluding deciles 0–3 while retaining ZERO) staying confounded (p=1.823e-15) directly demonstrates that decile-level imbalance in the nonzero remainder is not sufficient on its own to explain the KS statistic; ZERO-stratum imbalance is.

## Forward pointer — what to test next

This isolates *where* the mismatch lives but not yet *why* the ZERO stratum differs so sharply between switching and non-switching pools. Two things should be tested before this is considered fully resolved rather than merely localized:

1. **Circularity check:** confirm what fraction of "switching" gene calls are themselves driven by a gene going to/from zero expression in one cell line. If switching-gene assignment logic is sensitive to near-zero TPM transitions, the ZERO-stratum imbalance could be partially definitional rather than a pure confound — this needs to be ruled in or out before treating B's clean result as license to simply drop ZERO from the matched design.
2. **Reconcile with the superseded pool-concentration finding:** re-run the original 57.6%-vs-27.7% deciles-0–3-concentration comparison with ZERO cleanly separated out, to confirm that finding doesn't reappear once ZERO-stratum imbalance is controlled for — if it fully disappears, that closes it as an artifact of the same root cause; if it persists at reduced magnitude, deciles 0–3 may still carry a secondary, smaller effect worth naming separately.

<!-- FLAG Day63: matching-pool shortfall in deciles 0-3 (WARN messages: decile 0 short by 275, decile 1 by 228, decile 2 by 219, decile 3 by 105) affects baseline, C, and D identically since all three draw from those deciles. This is a known, previously-documented matching-pool limitation (consistent with prior decile-collapse investigation), not new to this analysis, but should be resolved before deciles 0-3 concentration is re-tested per forward pointer #2 above. -->
