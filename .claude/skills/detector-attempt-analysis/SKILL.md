---
name: detector-attempt-analysis
description: Analyze the detector-attempt evidence corpus (schema-v8+ evaluation records + detectorAttempts streams) and produce grounded scanner-improvement suggestions. Use when asked to analyze a new batch analysis run, assess detection quality, investigate misses/rejections/hallucinations, or recommend frontend detection changes.
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

Prints six sections: corpus health, attempt funnel, crop quality, rejection gates,
raw-stream extras, and a per-run table. `--runs-prefix 20260724-16` filters every
section to one batch.

The first four and the last read **record fields** — since #85/#86/#88 the records
carry rejection verdicts, crop IoU / containment / miss causes and non-conformance
causes, scored under thresholds each record stamps on itself, so the script no longer
re-derives them. Only "raw-stream extras" walks `detections/*_pose.json`, for the
measures no record carries: `selectionMethod`, reacquire *success*, absolute
luma/sharpness of the searched region, status run-lengths, and full-frame crop counts.

Record sections score a run's **truth-matched** attempts (42663 of 45468 in the
2026-07-24 batch); the raw-stream section scores all of them. The status mix therefore
differs by about a point between the two — a population difference, not a disagreement.

The standing report carries the full-stream funnel too (#87) — "Detector Attempt funnel
(run unit)" in `report.html`, with `eval_attempt_funnel_{status,runs,run_stats,flags}.csv`
beside it. Prefer those when a report run already exists: they cover every attempt, not
just the truth-matched ones. Either way read the run-unit columns, which the pooled
percentages alone will mislead you on (pooled missing is 26% while the median run misses
9% — a handful of collapsed runs carry the pool).

## Reading the evidence (schema v12)

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
- `frameQuality.hallucinationSplit` (#69) — `hallucination-fp` split by whether the
  Climber was in the frame at all (`truthPresent` on each `frameQuality.frames[]` entry).
  `truth-absent` is a real false positive → **presence gating**; `truth-present` is a
  tracking miss → **tracking robustness**. Never pitch one fix for the whole class
  without reading the split. Today it runs 100% truth-absent, because the auto
  classifier only sets the class in the `not truth.present` branch — the truth-present
  side needs human detection annotations (#45), which the corpus has none of yet. A
  pre-v12 record carries no `truthPresent`; that is **unknown**, never absent.
- `frameQuality.rejectionCorrectness` (#85) — the scanner's flip / quality gates
  second-guessed against truth. Verdicts are `goodPoseRejected` (the gate threw away a
  pose that agreed with truth), `badPoseRejected`, `truthUnknown`, split by gate under
  `byStatus`. Judge the gate's geometry on `overRejectionRateTruthPresent`: rejections
  on Climber-absent frames are correct by construction and only dilute the plain
  `overRejectionRate`. Per-frame verdicts and reasons sit on each
  `frameQuality.frames[]` entry (`rejectionVerdict`, `rejectionReason`, …). Don't
  re-derive this from `centroidDist` — centroid alone passes a pose whose joints scatter.
- `cropQuality` (#86) — one entry per **matched attempt of every status**, unlike
  `frameQuality`, which only holds frames where a pose was emitted. Carries the truth
  bbox, `initialSearchRegionIou` / `detectionRegionIou` (did the crop *frame* the
  Climber), `initialCropContainment` / `cropContainedTruth` (did it *cover* them at all —
  a large but well-placed crop scores badly on IoU and perfectly here), the fired search
  flags, and per missing attempt a `missCause` of `climber-absent` / `crop-misplaced` /
  `adverse-conditions` / `unexplained`. `crop-misplaced` is deliberately withheld when a
  full-frame reacquire also ran and failed: the Climber was searched for everywhere, so
  the crop cannot be what lost them. Thresholds are provisional and echoed in
  `cropQuality.thresholds`.
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
   distributions (median/p90 across runs) instead. The condition-band tables
   (`eval_condition_bands.csv`, `eval_frame_quality_condition_bands.csv`) already
   do this for you (#70): their CI is a cluster bootstrap over runs and they carry
   `n_runs` + `run_rate_median` / `run_rate_p90`. Judge a band by `n_runs`, not by
   its frame count — a band with 100k frames from 5 runs is 5 observations.
4. **Watch for bimodality.** Corpus medians hide catastrophic runs. Always report
   "runs > 50% missing" style tail counts alongside medians.
5. **Legacy runs stay comparable, not equivalent.** Same-day legacy batches
   (e.g. 20260724 morning) duplicate videos under different evidence. The
   pipeline now drops them for you (#89): when a video+truth pairing carries both
   generations, only the attempt-backed record pools, and the dropped ones are
   listed in `eval_superseded_records.csv` plus the report's shame lists. Working
   off records directly? Apply the same rule yourself — and check the pooled
   section's stated evidence generation before comparing across batches.

## Baseline reference (2026-07-24 batch, 68 attempt-backed runs / 14 routes)

Re-derived 2026-07-25 from the full schema-v12 regen (`evaluate --mode all --prune`),
so every row below comes off the current record fields. Corpus scale for orientation:
281 evaluation records on disk, 68 of them attempt-backed; after the #89
evidence-generation dedup 85 records pool and 59 of those pass the #15 gate (the
report's trusted pool). Use these to judge whether a new batch improved or regressed:

| metric | baseline |
|---|---|
| conforming runs | 48/68 (non-conforming: 12 sparse-match, 8 suspected-mistrack) |
| pooled PCK@0.5-torso | 0.60 (conforming runs: median 0.87, 83% within 0.75–0.95) |
| detect rate on truth-present | 74.1% |
| hallucination on truth-absent frames | 46.5% |
| `hallucination-fp` frames | 16.7% of detected (5264/31535), **100% truth-absent** — all real FPs, no tracking-miss half yet (#69) |
| accepted / missing / flipRejected / qualityRejected | 66.8% / 26.1% / 7.0% / 0.1% (truth-matched; all-stream 67.9 / 25.0 / 7.0 / 0.1) |
| per-run missing share | median 9.2%, p90 64.3%, 12 runs > 50% |
| flip-gate over-rejection | 76.7% of truth-present checkable rejections (1337 good poses of 1744); 46.5% counting Climber-absent ones |
| quality-gate over-rejection | 0% — 56 rejections all batch, 29 checkable, none good |
| miss causes | unexplained 50.5% / climber-absent 44.1% / adverse-conditions 5.4% / crop-misplaced 0% |
| initial crop IoU vs truth bbox | median 0.289 (accepted 0.307, missing 0.000) |
| crop contained truth bbox | 79.1% of scored attempts |
| reacquire success | 4.3% (513 / 11859) |
| heldPose (= frozenStale) on attempts | 5.8% |
| frozenStale on legacy-frames records | 0.5% (heldPose 63.7% — the gap *is* the post-processing) |
| underexposed flag rate, missing vs accepted | 7.9% vs 0.8% |

Two of these are new gates worth watching rather than settled facts. `crop-misplaced` is
0% only because every missing attempt in this batch ran a full-frame reacquire, which
bars the causal claim (`_miss_cause`) — 3850 of the 5585 containment-scored `unexplained`
misses still had a crop that excluded the truth bbox, so crop placement is bad, it just
isn't provably *the* cause. And the crop IoU / containment thresholds are provisional
(#86), not yet fit.

## Suggestion quality bar

A good suggestion names: the failure mode (funnel stage), the evidence (metric +
which runs), the proposed scanner change, and the metric expected to move. Example:
"Full-frame reacquire succeeds 4.3% of the time and missing runs have median
length 3 (max 1564): after N consecutive misses, reset the adaptive crop with an
expanding search ladder instead of a single full-frame retry; expect per-run
missing p90 to drop." Weak suggestions cite pooled percentages without a
mechanism or a target metric.
