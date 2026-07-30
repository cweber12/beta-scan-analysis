# ADR 0009 — the evaluation schema is frozen for a baseline cycle, and every pooled number states its basis

- Status: Accepted
- Date: 2026-07-30
- Upholds: the issue #89 evidence-generation precedent — a pooled number carries its own
  provenance rather than relying on the reader to establish it
- Issue: [#131](https://github.com/cweber12/beta-scan-analysis/issues/131),
  slice of PRD [#129](https://github.com/cweber12/beta-scan-analysis/issues/129)

## Context

The evaluation schema moved **v8 → v11 → v12 → v13 → v14 in about two weeks**. Each bump
was individually justified — sustained-run `heldPose` (v8), attempt evidence (v9/v10),
non-conformance cause (v11), `truthPresent` (v12), `missReason` (v13), absence provenance
(v14, ADR 0008). None was gratuitous.

The cumulative effect was still that **no two baselines were ever scored on the same
basis**. "Improvement" and "regression" across batches were largely uninterpretable,
because a metric that moved between two batches could have moved because the scanner
changed, because the corpus changed, or because the thing being counted changed
definition underneath it.

That was not a theoretical cost. The miss split **"88% no-candidates / 12%
identity-gated"** survived four baselines, was used to argue the direction of the
scanner's search ladder, and then turned out to be a pooling artifact: the run-unit median
is 99.0% no-candidates, and 82% of all identity-gated frames come from just 2 runs. Four
consecutive baselines agreed with each other and all four were wrong, and nothing in the
report made that checkable.

The records *do* stamp `schemaVersion`, so the basis was always recorded. What was missing
is that nothing stated it next to the numbers, and nothing stopped a sweep being scored,
re-scored under a different basis, and the two being compared.

## Decision

**Two parts: freeze the basis for a cycle, and make every pooled number state the basis it
rests on.**

### 1. `BASELINE_CYCLE_SCHEMA` declares the frozen basis

A constant in `evaluate.py` beside `SCHEMA_VERSION`, frozen at **v14** as of 2026-07-29 on
the post-reset sweep scored in PR #128. It holds for one full cycle — collect → score →
analyse → act — rather than moving whenever a bump is convenient.

**A mid-cycle bump is permitted but never silent.** `SCHEMA_VERSION` moving while
`BASELINE_CYCLE_SCHEMA` stays put is a legible state, not an error: a real contract change
can force one. While the two differ, every pooled section of the report carries a re-score
demand, because scoring only the new batch leaves the compared population straddling two
bases. The fix is `evaluate --mode all` over the **whole** compared population.

Making the bump an error was rejected — it would push the next contract change into
working around the check rather than declaring it.

### 2. Every pooled section states its schema versions and build set

`_measurement_basis` mirrors the shape of issue #89's `_evidence_generation_summary`: one
summary per named pool, rendered into the section rather than once at the top. The three
pools that already carry an evidence line now carry a basis line beside it — trusted
pooled metrics, the per-frame/attempt pools, and the attempt funnel.

The basis is two things, because comparability needs both:

| Half | Read from | Why it matters |
|---|---|---|
| schema version(s) | each record's `schemaVersion` | what was counted |
| build set | `(appVersion, detectorCodeHash)` per run (#130) | what was measured |

**Mixture is flagged, not refused.** A corpus mid-migration legitimately spans bases, and
refusing to report would destroy the accounting that shows what the mixture *is*. This
follows the fail-open-and-name-it convention the rest of the pipeline uses: the failure
being prevented is a silent blend, not a blend. A mixed-schema pool renders as
`basis: MIXED SCHEMA`, in a visually distinct block, naming how many records are off-basis
and what to run.

**An unstamped record reads as `unknown`, never as the frozen version.** Collapsing the two
would let the exact contamination this slice exists to surface read as clean. `unknown`
sorts last in the version list — it is not a version and must not sort among them.

## Consequences

- Two report sections resting on different bases can no longer be read against each other
  without the reader being told. That is the whole deliverable.
- The freeze is a **process** commitment enforced by visibility, not by a lock. Nothing
  prevents a bump; the report just stops claiming comparability across one.
- This slice changes no data and re-collects nothing. Records already on disk are read
  exactly as written.
- The next schema bump has a defined cost attached: re-score the whole compared
  population, or accept a loudly-flagged mixed pool. Previously the cost was zero at the
  point of the bump and paid later by whoever read the numbers.
- The build half will sharpen as `detectorCodeHash` coverage grows. Only 4 of 499 pose runs
  carry it today (#130 landed late in the corpus's life), so most build sets are still
  named by commit stamp alone — which is exactly the "usable, stamp-suspect" tier that the
  retention slice (#135) exists to record.

## Alternatives considered

- **Refuse to report a mixed-schema pool.** Rejected: it destroys the accounting that
  makes the mixture legible, and a mid-migration corpus is a normal state. Naming beats
  refusing when the reader can act on the name.
- **Make a mid-cycle bump a hard error.** Rejected: contract changes are sometimes forced,
  and an error would be worked around rather than declared. The demand for a full re-score
  achieves the same protection without inviting a bypass.
- **State the basis once at the top of the report instead of per section.** Rejected for
  the reason #89 already established: a number read out of the middle of the report has to
  carry its own provenance, and the pools genuinely differ — the trusted pool quarantines
  where the failure-mode pools do not (#132), so one top-level statement would be wrong for
  some sections. A top-level declaration is rendered *in addition*, to name the cycle.
- **Migrate every old record forward to the frozen schema.** Rejected as out of scope and
  dishonest in the general case: a v12 record has no `missReason` to backfill, and
  synthesising one would fabricate the basis rather than record it. Re-scoring
  (`--mode all`) is the honest path where the inputs are still on disk.
