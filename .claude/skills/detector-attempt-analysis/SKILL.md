---
name: detector-attempt-analysis
description: Analyze the detector-attempt evidence corpus (schema-v8 evaluation records + detectorAttempts streams) and produce grounded scanner-improvement suggestions. Use when asked to analyze a new batch analysis run, assess detection quality, investigate misses/rejections/hallucinations, or recommend frontend detection changes.
---

# Detector-attempt corpus analysis

Turns the `analysis/` bundle corpus into a diagnosis of *why* pose detection
fails and *what to change in the scanner*. Built for the detectorAttempts
evidence contract (PRD "Backend Analysis Evidence Payload", issues #68/#73/#74/#75).

## Before judging anything: regenerate

Evaluation records are derived artifacts. If new detection runs landed under
`analysis/` (or `evaluate.py` changed), records on disk are stale until you re-run:

```
python -m analysis_pipeline evaluate --mode all --prune
python -m analysis_pipeline analysis -o reports --no-decode
```

- Commit new raw detections and regenerated records as separate `data:` commits
  (see CLAUDE.md commit conventions).
- The full-decode `analysis` run (without `--no-decode`) adds video-derived
  condition predictors but takes 10+ minutes; only needed for condition-band
  report sections, not for attempt analysis — attempts carry their own
  `searchConditions`.

## The standard deep-dive

```
python .claude/skills/detector-attempt-analysis/scripts/attempt_deep_dive.py
```

Prints: corpus health, attempt funnel, conditions by status, crop areas,
flip-rejection correctness, and a per-run table. Optionally filter the raw-stream
sections to one batch: `--runs-prefix 20260724-16`.

## Reading the evidence (schema v8)

Per run×truth pairing: `analysis/<route>/<video>/evaluations/<run_ts>_vs_<truthhash>.json`.

- `frameQuality.detectorEvidence` — `"attempts"` (trust for detector claims) vs
  `"legacy-frames"` (post-processed playback; detector claims are proxies only).
  A missing attempt stream is **unknown**, never raw success.
- `detectorAttemptStatusCounts` — `accepted / missing / flipRejected /
  qualityRejected` over truth-matched frames.
- `heldPose` — sustained (>= 3) near-identical run, non-anchor frames only;
  neutral diagnostic. `frozenStale` — heldPose **and** raw detector output
  (attempt-backed, or legacy `source == "raw"`). Never treat heldPose alone as a
  scanner failure.
- `conformance.conforms` (#15 gate) — when false, keep the run out of pooled
  metrics. `conformance.cause` (#88) says which failure it was: `sparse-match`
  (the detector supplied too little to fit — a detector problem tripping a truth
  gate) or `suspected-mistrack` (ample accepted detections, fit still off
  identity — the only class worth re-seeding truth for). Read the annotation;
  don't re-derive it from `miss%`. `conformance.causeEvidence` carries the
  matched-present frame count and accepted share it was decided from.
- Raw streams: `detections/<run_ts>_pose.json` → `data.detectorAttempts[]` with
  `status`, `rawKeypoints` (pre-mutation selected pose), `acceptedKeypoints`
  (accepted only), `initialSearchRegion`/`detectionRegion` (normalized;
  `{0,0,1,1}` = full frame, `null` = unknown), `reacquireAttempted`/`reacquired`,
  `searchConditions.{overall,climber,wall,flags}`, `candidateCount`,
  `rejectedCandidateCount`, `selectionMethod` (`tap|tracked|strongest`).

## Interpretation rules

1. **Ground Truth owns expected presence/pose; attempts own scanner behavior.**
   Score scanner decisions *against* truth (e.g. was a flipRejected raw pose
   actually near truth?) — never trust setup.json labels for evaluation.
2. **Split the failure funnel before recommending.** A low detect rate can be
   crop loss (missing with reacquire failure), over-rejection (flipRejected near
   truth), or conditions (flags on the searched region). Each implies a different
   scanner change; pooled rates alone don't.
3. **Beware pseudo-replication.** Frames within a run are correlated; never quote
   pooled per-frame CIs as if frames were independent. Compare per-run
   distributions (median/p90 across runs) instead.
4. **Watch for bimodality.** Corpus medians hide catastrophic runs. Always report
   "runs > 50% missing" style tail counts alongside medians.
5. **Legacy runs stay comparable, not equivalent.** Same-day legacy batches
   (e.g. 20260724 morning) duplicate videos under different evidence; exclude
   them when computing attempt-era rates.

## Baseline reference (2026-07-24 batch, 68 runs / 14 routes)

Use these to judge whether a new batch improved or regressed:

| metric | baseline |
|---|---|
| conforming runs | 48/68 |
| pooled PCK@0.5-torso | 0.60 (conforming runs mostly 0.75–0.95) |
| detect rate on truth-present | 74% |
| accepted / missing / flipRejected / qualityRejected | 68% / 25% / 7% / 0.1% |
| hallucination on truth-absent frames | 46% |
| reacquire success | 4.3% |
| flipRejected raw pose within 0.10 of truth | ~71% of truth-checkable (over-rejection) |
| heldPose (= frozenStale) on attempts | 5.8% |
| frozenStale on legacy source-covered runs | 0.5% |
| runs > 50% missing | 11/68 |
| underexposed flag rate, missing vs accepted | 7.7% vs 0.7% |

## Suggestion quality bar

A good suggestion names: the failure mode (funnel stage), the evidence (metric +
which runs), the proposed scanner change, and the metric expected to move. Example:
"Full-frame reacquire succeeds 4.3% of the time and missing runs have median
length 3 (max 1564): after N consecutive misses, reset the adaptive crop with an
expanding search ladder instead of a single full-frame retry; expect per-run
missing p90 to drop." Weak suggestions cite pooled percentages without a
mechanism or a target metric.
