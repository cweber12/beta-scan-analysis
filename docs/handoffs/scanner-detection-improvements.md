# Handoff: detection-logic improvements from the first attempt-backed corpus

**Audience:** an agent working in the **Beta Scanner** repo (the Next.js pose/ORB
app). The harness measured the scanner's first full `detectorAttempts[]` corpus
(2026-07-24, 68 runs / 14 routes, schema-v8 evaluation) and this handoff turns
those findings into ordered scanner changes. The harness owns the metrics; the
scanner owns the detection behavior. Evidence fields referenced here are defined
in [scanner-detector-attempt-evidence.md](scanner-detector-attempt-evidence.md).

## Baseline you are improving against

Attempt funnel over truth-matched frames: **68% accepted / 25% missing /
7% flipRejected / 0.1% qualityRejected**. Detect rate on truth-present frames
74%; pooled PCK@0.5-torso 0.60 (conforming runs 0.75–0.95). Each change below
names the metric the harness will re-measure on the next batch; the
`detector-attempt-analysis` skill in the harness repo prints all of them.

## 1. Replace single-shot full-frame reacquire with an expanding search ladder

**Evidence.** Reacquire runs on every miss and succeeds **4.3%** of the time
(11,859 attempts → 513 rescues). Missing comes in sustained runs (median 3
frames, max 1,564 — a run that never re-finds the climber), and it is bimodal:
the median run misses 9.2% of frames but **11 of 68 runs miss >50%**. Missing
frames' `initialSearchRegion` is *larger* (median area 0.20) than accepted
frames' (0.15) — the adaptive crop is following a lost track, not cropping too
tight.

**Change.** After N consecutive misses (suggest N = 2–3):
1. reset the adaptive crop rather than letting it keep drifting;
2. search an expanding ladder seeded at the last confident position, scaled by
   recent track velocity — last-known box × 2, × 4, then full frame — instead of
   jumping straight to one full-frame attempt;
3. export each ladder step in the attempt row (`reacquireSteps[]`, see contract
   addendum) so the harness can measure which rung rescues.

**Target metrics.** Reacquire success rate (4.3% baseline), per-run missing p90
(69%), max missing run length (1,564), runs >50% missing (11/68).

## 2. De-latch the flip-rejection gate

**Evidence.** Of flip-rejected frames where truth exists, **~71%** had a raw pose
within 0.10 normalized centroid distance of Ground Truth — plausibly good poses
discarded. The gate also latches: consecutive flip-rejection runs reach **398
frames**, and individual runs lose 25–60% of all frames to it (e.g. planet-x
`R0Z6c1zlic0` 61%, rug-rat `H4ZHP3-EoqA` 58%).

**Change.**
1. Require sustained flip evidence — 2–3 consecutive flip verdicts — before
   rejecting, instead of rejecting on a single-frame verdict.
2. Cap consecutive rejections: after K rejections in a row (suggest K ≈ 5),
   accept-with-flag rather than discard, since the stream is otherwise lost for
   the whole stretch.
3. Keep exporting the rejected raw pose either way — the harness scores whether
   the rejection was correct.

**Target metrics.** flipRejected share (7% → ~2%), over-rejection share among
truth-checkable rejections (71%), max flip-rejection run length (398).

## 3. Gate acceptance when the track is stale (hallucination suppression)

**Evidence.** **46.5%** of truth-absent matched frames carry an accepted pose,
and `selectionMethod` is `tracked` on 99.5% of attempts — when the climber
leaves the frame, tracked selection latches onto spectators, pads, or wall
features. This is the harness's #1 failure class since the 2026-07-23 baseline
(hallucination-fp, 16.7% of detected frames in the attempt corpus).

**Change.** When the tracked subject was recently lost (recovering from missing)
or exited the frame edge, raise the acceptance bar for a "new" subject: minimum
keypoint-score floor plus size/position consistency with the last confident
track before re-latching. A candidate failing the bar becomes `missing` (or a
rejected candidate), not an accepted pose.

**Target metrics.** Hallucination-on-absent rate (46.5%), hallucination-fp class
share (16.7%), while holding detect-rate-on-present (74%) steady — the gate must
not trade real detections away.

## 4. Exposure-compensate the search region before inference

**Evidence.** Failure conditions live in the flags, not the medians: missing
frames are `isUnderexposed` 7.7% and `isBacklit` 5.5% of the time vs 0.7% /
0.6% for accepted; flip-rejected frames are underexposed 9.4%. Roughly a 10×
odds shift on the same corpus.

**Change.** When `searchConditions.flags` fire on the crop (underexposed,
backlit, low-contrast), apply cheap local compensation — histogram equalization
or gamma correction on the search region — before the MediaPipe pass. At dev
Analyze stride this is affordable; if it proves out, consider it for user scans.

**Target metrics.** Missing/flipRejected rate on flag-firing frames vs
flag-quiet frames (the flag odds ratio should compress toward 1).

## 5. Audit dead evidence paths

- `selectionMethod: "strongest"` never appears in 45k+ attempts (only `tracked`
  45,225 and `tap` 243). Either the path is dead or it is mislabeled — verify.
- `qualityRejected` fired 56 times in 45k attempts (0.1%). Confirm the
  quality/filtering pass is actually wired into dev Analyze classification, or
  its thresholds are so loose the status is meaningless.
- `searchConditions.wall` is always `null` — populate it (contract addendum).

**Target metric.** Funnel completeness: every attempt status and selection
method observed in data at plausible rates.

## Sequencing

Ship 1 and 2 first (largest recoverable frame loss, both purely detection-path),
then 3 (needs the score evidence from the contract addendum to tune its floor),
then 4–5. Re-run harness Batch Analyze after each ship so the version-delta
report attributes movement to one change at a time.

## Acceptance checklist

- Each shipped change re-measured on a fresh full-corpus Batch Analyze against
  the baseline table above.
- No user-facing scan behavior change beyond the intended detection fixes;
  Detection Preview unchanged.
- New evidence fields (ladder steps, score evidence, wall conditions) land per
  the contract addendum in
  [scanner-detector-attempt-evidence.md](scanner-detector-attempt-evidence.md).
