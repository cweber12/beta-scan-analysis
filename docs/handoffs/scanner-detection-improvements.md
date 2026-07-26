# Handoff: detection-logic improvements from the first attempt-backed corpus

**Audience:** an agent working in the **Beta Scanner** repo (the Next.js pose/ORB
app). The harness measured the scanner's first full `detectorAttempts[]` corpus
(2026-07-24, 68 runs / 14 routes) and this handoff turns those findings into
ordered scanner changes. The harness owns the metrics; the scanner owns the
detection behavior. Evidence fields referenced here are defined in
[scanner-detector-attempt-evidence.md](scanner-detector-attempt-evidence.md).

> **Revised 2026-07-25** against the schema-v11 corpus regen (harness PRD #84).
> Same runs, better instruments: rejection correctness, crop IoU / containment and
> miss causes are now *scored into the evaluation records* rather than derived by
> ad-hoc script. Two figures moved as a result and the old ones should not be
> quoted — the flip over-rejection rate (§2) and the crop evidence in §1.

## Baseline you are improving against

Attempt funnel over truth-matched frames: **66.8% accepted / 26.1% missing /
7.0% flipRejected / 0.1% qualityRejected**. Detect rate on truth-present frames
74.1%; pooled PCK@0.5-torso 0.60 (conforming runs: median 0.87, 83% within
0.75–0.95).

Each change below names the metric the harness re-measures on the next batch, and
where that metric now lives. Two ways to read them:

- **Report CSVs** (`python -m analysis_pipeline analysis -o reports`) —
  `eval_attempt_funnel_{status,runs,run_stats,flags}.csv`,
  `eval_crop_quality_{attempts,miss_causes}.csv`, and the rejection columns on
  `eval_detection_error_attempt_runs.csv`.
- **The `detector-attempt-analysis` skill's deep-dive script**, which prints all of
  them in six sections.

Population note, because the two disagree by about a point and it is not a bug:
record-backed sections score a run's **truth-matched** attempts (42,663 of 45,468
in this corpus); the funnel CSVs cover the full stream. Compare like with like.

## 1. Replace single-shot full-frame reacquire with an expanding search ladder

**Evidence.** Reacquire runs on every miss and succeeds **4.3%** of the time
(11,859 attempts → 513 rescues). Missing comes in sustained runs (median 3
frames, max 1,564 — a run that never re-finds the climber), and it is bimodal:
the median run misses 9.2% of its truth-matched attempts but **12 of 68 runs
miss >50%**.

The crop evidence is now measured against the Ground Truth bbox rather than
inferred from region size, and it is damning: on truth-present misses the
searched region contained the Climber only **31.4%** of the time (1,945 of 6,189
scored), against **90.2%** on accepted attempts. Median crop-vs-truth IoU on
missing attempts is **0.000** — the crop is not merely loose, it is frequently
somewhere else. The adaptive crop is following a lost track.

*(This supersedes the earlier "missing regions are larger, median area 0.20 vs
0.15" framing. Region area was always a poor proxy: a large, correctly-placed
crop and a small, misplaced one are opposite failures and area cannot tell them
apart. IoU and containment can.)*

**Why the harness cannot yet tell you which misses the crop caused.** Miss causes
classify as **50.5% `unexplained` / 44.1% `climber-absent` / 5.4%
`adverse-conditions` / 0% `crop-misplaced`**. That 0% is not a clean bill — it is
the classifier refusing an unprovable claim. Because a full-frame reacquire ran on
*every* miss, the Climber was nominally searched for everywhere, so a misplaced
crop cannot be shown to be what lost them. Hence half the misses land in
`unexplained`, and the harness cannot rank crop placement against detector
weakness until it can see **what region the reacquire actually searched** — a
missing attempt currently reports no `detectionRegion` at all. Item 3 below is
what unblocks that.

**Change.** After N consecutive misses (suggest N = 2–3):
1. reset the adaptive crop rather than letting it keep drifting;
2. search an expanding ladder seeded at the last confident position, scaled by
   recent track velocity — last-known box × 2, × 4, then full frame — instead of
   jumping straight to one full-frame attempt;
3. export each ladder step in the attempt row (`reacquireSteps[]`, see contract
   addendum) so the harness can measure which rung rescues — and, critically, so
   missing attempts stop being causally opaque.

**Target metrics.** Reacquire success rate (4.3% baseline), per-run missing p90
(64.3%), max missing run length (1,564), runs >50% missing (12/68), crop
containment on truth-present misses (31.4%), and the `unexplained` miss share
(50.5%, which should fall as `reacquireSteps[]` lets causes be assigned).
Read from `eval_crop_quality_miss_causes.csv` and
`eval_attempt_funnel_run_stats.csv`.

## 2. De-latch the flip-rejection gate

**Evidence.** The harness now scores every rejected raw pose against Ground Truth
and issues a verdict. **76.7%** of flip rejections on frames where the Climber was
actually present threw away a pose that agreed with truth — **1,337 good poses
discarded out of 1,744 checkable**. Counting the Climber-absent rejections too
(correct by construction, so they only dilute) the rate is 46.5%. The gate also
latches: consecutive flip-rejection runs reach **398 frames**, and individual runs
lose 25–60% of all frames to it (e.g. planet-x `R0Z6c1zlic0` 61%, rug-rat
`H4ZHP3-EoqA` 58%).

*(This supersedes the earlier "~71% within 0.10 normalized centroid distance"
figure, which came from an ad-hoc proxy. Centroid distance alone passes a pose
whose joints scatter around the right centre; the record's verdict applies the
harness's real geometry classifier plus a majority-of-joints agreement floor. The
new number is both higher and trustworthy — do not quote the old one.)*

The **quality** gate, by contrast, is clean: 56 rejections in the whole corpus, 29
of them truth-checkable, **none** wrongly rejected. Whatever is wrong here is
specific to the flip gate.

**Change.**
1. Require sustained flip evidence — 2–3 consecutive flip verdicts — before
   rejecting, instead of rejecting on a single-frame verdict.
2. Cap consecutive rejections: after K rejections in a row (suggest K ≈ 5),
   accept-with-flag rather than discard, since the stream is otherwise lost for
   the whole stretch.
3. Keep exporting the rejected raw pose either way — the harness scores whether
   the rejection was correct.

**Target metrics.** flipRejected share (7.0% → ~2%), over-rejection on
truth-present rejections (**76.7%** — `over_rejection_rate_truth_present` on
`eval_detection_error_attempt_runs.csv`, or `rejectionCorrectness` on the record),
max flip-rejection run length (398). Watch that the two rates move together: a
"fix" that only stops rejecting on Climber-absent frames would move the pooled
46.5% without touching the gate's geometry judgement.

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
share (16.7%), while holding detect-rate-on-present (74.1%) steady — the gate must
not trade real detections away. Note that 44.1% of all misses are already
`climber-absent`, i.e. correct: suppressing hallucination will push that share up,
and that is the intended direction, not a regression.

## 4. Exposure-compensate the search region before inference

**Evidence.** Failure conditions live in the flags, not the medians: missing
frames are `isUnderexposed` 7.9% and `isBacklit` 5.6% of the time vs 0.8% /
0.3% for accepted; flip-rejected frames are underexposed 10.0%. Roughly a 10×
odds shift on the same corpus. Absolute luma and sharpness medians barely differ
by status (accepted overall luma 119.8 vs missing 120.0) — the flags carry the
signal, the levels do not.

Note the ceiling, though: only 5.4% of misses classify as `adverse-conditions`.
Flag-firing frames fail much more often, but most failures happen on frames where
no flag fired at all. This is a real effect on a minority of frames, which is why
it sits at #4 and not higher.

**Change.** When `searchConditions.flags` fire on the crop (underexposed,
backlit, low-contrast), apply cheap local compensation — histogram equalization
or gamma correction on the search region — before the MediaPipe pass. At dev
Analyze stride this is affordable; if it proves out, consider it for user scans.

**Target metrics.** Missing/flipRejected rate on flag-firing frames vs
flag-quiet frames (the flag odds ratio should compress toward 1), and the
`adverse-conditions` miss share (5.4%). Read from
`eval_attempt_funnel_flags.csv`, which carries the per-run distribution beside
each pooled rate — flags cluster hard within a run, so one dark video can flood
the pool.

## 5. Audit dead evidence paths

- `selectionMethod: "strongest"` never appears in 45k+ attempts (only `tracked`
  45,225 and `tap` 243). Either the path is dead or it is mislabeled — verify.
- `qualityRejected` fired 56 times in 45k attempts (0.1%), and every one the
  harness could check was a *correct* rejection. A gate that never fires and is
  never wrong is indistinguishable from a gate that is not wired in. Confirm the
  quality/filtering pass actually runs in dev Analyze classification, rather than
  its thresholds being so loose the status is meaningless.
- `searchConditions.wall` is always `null` — populate it (contract addendum).

**Target metric.** Funnel completeness: every attempt status and selection
method observed in data at plausible rates.

## Sequencing

Ship 1 and 2 first (largest recoverable frame loss, both purely detection-path),
then 3 (needs the score evidence from the contract addendum to tune its floor),
then 4–5. Re-run harness Batch Analyze after each ship so the version-delta
report attributes movement to one change at a time.

If you want to ship a single small thing first, ship **`reacquireSteps[]` (item
1.3) on its own**. It is the cheapest change on this list and it is the one
blocking the harness: half of all misses currently classify as `unexplained`
purely because a missing attempt reports nothing about where the reacquire looked.
With ladder steps exported, the harness can assign causes to that half and rank
crop placement against detector weakness — which tells you whether items 1 and 3
are correctly ordered. Right now that ordering is an inference, not a measurement.

## Acceptance checklist

- Each shipped change re-measured on a fresh full-corpus Batch Analyze against
  the baseline table above.
- No user-facing scan behavior change beyond the intended detection fixes;
  Detection Preview unchanged.
- New evidence fields (ladder steps, score evidence, wall conditions) land per
  the contract addendum in
  [scanner-detector-attempt-evidence.md](scanner-detector-attempt-evidence.md).
