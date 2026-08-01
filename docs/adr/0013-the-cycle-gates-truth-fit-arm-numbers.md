# ADR 0013 — the Cycle gates the truth-fit arm numbers and covaries the per-Bundle table

- Status: Accepted
- Date: 2026-08-01
- Relates to: issue #176 (this decision), #168 (built the Cycle), #164 (built the arm
  comparison), #132 (the gate-versus-covariate precedent), #131 / ADR 0009 (measurement basis)

## Context

#168 built the **Cycle**: a comparison group opened before the first batch and closed after
the last, with a determinism canary and a truth-hash snapshot spanning it. At close it
writes `comparableBundles` — the Bundles whose truth, setup and crop trajectory were
byte-identical at both ends. #164 built the arm-versus-arm reporting. **Nothing connected
them**: `comparableBundles` was read by nothing under `analysis_pipeline/`, and the arm
comparison pooled every harness run on disk regardless of when it ran.

That produced no wrong number while the arms were single-Bundle probes. It becomes
load-bearing the moment a real mode-major sweep runs, which is the next step — batches are
mode-major *precisely so that* drift between the first batch and the last is confounded
with mode, and the Cycle is the only thing that catches it. An 8.8 h sweep read through an
ungated report throws that away silently.

The question the issue posed was binary — hard gate or covariate — and the answer is
neither alone. #132 already split exactly this question for the #15 conformance gate, and
the split turns on one test: **is the gate criterion correlated with the thing being
measured?**

- **Truth-fit metrics** (PCK and everything derived) are scored *against* Ground Truth. If
  the truth doesn't fit, the number isn't noisy — it is meaningless. → gate.
- **Failure-mode metrics** (what the detector did, before truth is consulted) would be
  *selecting on the outcome* if gated: the runs failing the conformance gate are
  disproportionately the runs where detection went badly, which is the population being
  studied. → covariate.

Applying that test to the Cycle changes the answer, because the Cycle's criterion is
different **in kind**. `comparableBundles` asks *"did this Bundle's inputs move between
Cycle open and close?"* — a truth re-seed, a recalibration, a rebuilt crop trajectory,
a model or module change. Those are **operational events, not detector outcomes**. A Bundle
is not dropped from a Cycle because detection went badly; it is dropped because a human
re-seeded its truth. So #132's selection-on-outcome hazard is much weaker here, which
argues for gating more freely than #132 does. (The one residual path — a crop trajectory
rebuilt *because* detection was failing — is why the per-Bundle detail must stay visible
rather than vanish.)

There is also existing precedent one layer down: `app.py` reports the enclosing `cycleId`
in the batch 202 and deliberately **does not gate** on it, because a batch outside a Cycle
is legitimate. Gating at the *analysis* layer while reporting at the *batch* layer is a
considered choice, not an accident: the batch is allowed to run anything, and the report is
what refuses to pool it.

## Decision

**Four postures, and the report states which one it applied.** No section may be read as if
it were another — the #132 precedent — so the posture is declared before the first table
rather than inferred from which rows are missing.

1. **`certified` → gate.** The pooled arm summary, the deltas and the comparison reach are
   computed over `comparableBundles` only. These are truth-fit numbers; a Bundle whose
   truth moved mid-Cycle yields a delta that silently contains a truth change, which is the
   confound PRD #156 exists to escape.
2. **The per-Bundle table is the covariate.** Every Bundle the arms ran on stays in it,
   marked with its comparability state and the reason. The gate removes rows from the
   pooled lines, **never from the evidence** — the #15/#88 precedent that a Bundle dropped
   from a comparison is never silently dropped.
3. **`failed` / `refused` → refuse.** No pooled comparison is published: no arm ranking, no
   Bundle × arm matrix, no deltas. `close_cycle` writes `comparableBundles` even when it
   fails and logs *"The arms in this cycle are NOT comparable to each other. Do not publish
   a comparison over them."* — so the gate keys on **`certified`**, never on the presence of
   that list. The runs are still listed as evidence, deliberately *not* in a matrix layout.
4. **`open` → in flight.** `comparableBundles` exists only at close, so there is nothing to
   gate on. The comparison renders as provisional and never as certified.
5. **No Cycle → label, don't gate.** The entire pre-#168 corpus and any probe run outside a
   Cycle. Nothing to gate against, so the comparison renders exactly as it did before, plus
   an explicit *not drift-checked* marker. Never silence, and never something that reads as
   certified.

**`newlyEligible` is its own state, not a kind of exclusion.** Those Bundles were never
snapshotted, so they did not fail anything — they were never in the Cycle rather than
dropped from it. Letting them read as failures would put a Bundle on a repair worklist for
the crime of having been created.

**The window scopes the run population, and every run it drops is named.** Nothing durable
stamps a Run with its `cycleId`, so a Run is placed by the base timestamp in its
`exp-<ts>-<arm8>-p<n>` id (#160) against `(openedRunTs, closedRunTs)` — the same join
`cycle_integrity.collect_cycle_runs` uses, matched pattern-for-pattern so the report cannot
pool runs the Cycle's own census never counted. A timestamp window is a weaker join than a
stamp, which is exactly why the exclusions are listed rather than merely subtracted from a
count. A Run predating the `exp-` convention is reported `unplaceable`, which is a
different statement from out-of-window.

**The pipeline reads the artifact; it does not import `cycle_integrity`.** Importing it
would drag `mediapipe_job` → `youtube_core` → `yt_dlp` and `vitpose_job` → `video_stats`
into the `analysis_pipeline` import graph, which ADR 0003 and ADR 0012 exist to keep out.
`analysis_pipeline/cycles.py` reads `analysis/cycles/*.json` as JSON, and
`test_cycle_integrity.py` — the one file that may import both halves — asserts the reader
agrees with the writer field by field over an artifact the guard actually produced. This is
the same trade `cycle_integrity.truth_identity` already makes in the other direction.

**The Cycle joins the measurement basis (#131).** `moduleVersion`, `sampleCoefficient` and
the pinned model shas sit outside both the record stamp and the pose envelope, and could
move between the first batch of a sweep and the last with nothing to say so. #168 records
them on the Cycle precisely for that reason; the arm section's basis line is where a reader
meets them.

## Consequences

- On the corpus as of this ADR, the certified Cycle scopes the comparison from **23 harness
  runs to 15**, and from **6 arms to 5** — the sixth arm (`crop:none`) ran hours before the
  Cycle opened and is now named as out-of-window rather than pooled with it.
- A failed Cycle produces **no publishable arm result at all**. That is the point, and it
  means an expensive sweep can end with nothing to publish; the runs and the canary diff
  stay in the report so the failure is itself the finding.
- Two Cycles are two comparison groups and are **never merged**. The arm section resolves
  one — the open Cycle if any, else the most recently closed — and names the others.
- The batch sidecar now records the `cycleId` the sweep ran inside. This is operator
  visibility only: it is a single corpus-level file overwritten by each batch, so it is not
  a durable per-Run join and the window remains the join the analysis uses.
- Stamping `cycleId` into the Run itself was **rejected for now**: it cannot go in the
  `configHash` block without making the same arm in two Cycles two different arms, and the
  evaluation schema is frozen for this cycle (ADR 0009). If the window join proves too weak
  in practice, a `diagnostics`-level stamp outside the hash is the next step.

## Alternatives considered

- **Covariate everywhere, gate nothing** (the #132 treatment applied uniformly). Rejected:
  it would put a delta containing a truth change into the pooled line with a footnote, and
  the whole reason the Cycle exists is that footnotes do not stop a clean-looking table from
  being read.
- **Gate everywhere, including the per-Bundle table.** Rejected: the excluded Bundles would
  disappear from the report entirely, which is the silent-drop failure #15 and #88 already
  ruled out — and it would hide the one residual correlation path (a trajectory rebuilt
  because detection was failing) rather than leave it visible.
- **Key the gate on the presence of `comparableBundles`.** Rejected, and it is the specific
  trap this ADR exists to close: a failed Cycle carries a populated list, so keying on it
  publishes exactly what the artifact forbids.
- **Refuse to render anything when there is no Cycle.** Rejected: that is the entire
  pre-#168 corpus, and refusing would destroy the accounting rather than qualify it.
