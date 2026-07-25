# Context — analysis harness glossary

The canonical vocabulary for the Beta Scanner **analysis harness**. Use these
terms exactly in issues, code, hypotheses, and reports. This file is a glossary,
not a spec — it carries no implementation detail.

## Core objects

- **Bundle** — a self-contained per-video record at
  `analysis/<route_folder>/<video_key>/`: the video, `final_frame.png`,
  `metadata.json`, `setup.json`, and timestamped detection files. May also carry a
  `vitpose.json` **scaffold** (below). The unit of ingest.
- **ViTPose scaffold** (`vitpose.json`) — per-frame ViTPose++ Climber keypoints the
  downloader writes to seed beta-scanner's human-authored Ground Truth. A *seed, not
  truth*: the human still corrects and owns it. It is **not** a detection Run — no
  `detections/*_pose.json` is produced. Emitted by `POST /api/vitpose`; see
  `docs/adr/0003`. The seed request is the **decoupled seed contract** (below).
- **Decoupled seed contract** — the `POST /api/vitpose` seed request of record:
  **`seed_tap`** (the tap anchoring **Climber Identity**, with an optional `t` that
  picks the nearest tapped frame) plus **`seed_region`** (the **seed gate** deciding
  which track is the climber). `seed_region` is *decoupled from the Climber Crop* — the
  crop is a Video Stats input, not the gate. Legacy `climber_point` / `climber_crop`
  are backward-compatible aliases; the new fields win when both are present. Scanners
  gate on `GET /api/contract` → `capabilities.decoupledSeed`. See `docs/adr/0006`.
- **Ground Truth** (`ground-truth.json`) — beta-scanner's per-frame pose truth
  artifact, authored from the ViTPose scaffold plus human flags. New artifacts carry
  top-level `setupHash` and per-frame `review` provenance. `review: "auto"` is
  agreement-tier evidence only; human-flagged frames are the accuracy-tier evidence.
  See `docs/adr/0004`.
- **Video Stats** (`video-stats.json` + `metadata.json.video_stats`) — computed
  image-statistic Predictors, two-phased: whole-frame *source stats* stamped into
  `metadata.json` at download/import (never stale), and crop-aware *region stats*
  in `video-stats.json` stamped with the `setupHash` they were computed under
  (stale exactly like Ground Truth when recalibration mints a new hash). Emitted
  by `POST /api/video-stats`; also carries the ViTPose-derived `cameraAngle`
  estimate. Continuous stats are Predictors; the *suggested labels* derived from
  them only prefill the hand labels. The saved `analysisInputs` layer is advisory
  metadata with per-label provenance (auto-accepted vs human-overridden) recorded
  by the scanner; it is not Ground Truth and must not be treated as the authority
  for main detector scoring.
- **Route** — a physical climb, identified by its `route_folder`. Multiple
  **Videos** of the same Route are the norm (different sessions/angles/lighting).
- **Run** — one detection execution on one Video, recorded as a paired
  `<run_ts>_pose.json` + `<run_ts>_orb.json`. **The Run is the unit of
  statistical inference** — coefficients are summarized across Runs, not pooled
  across frames.
- **Detector Attempt** — one scanner-owned MediaPipe attempt on the sampled
  100 ms analysis timeline: the initial search region, whether full-frame
  reacquire ran, the selected raw Climber pose when MediaPipe returned one, the
  accepted pose when the scanner kept it, the rejection/missing status, compact
  candidate-selection metadata, and scanner-computed pixel conditions for the
  searched region. Detector Attempts are evidence, not recommendations; the
  harness joins them to Ground Truth to derive Detection Errors.
- **Rejection Verdict** — the harness's judgement of one *rejected* Detector
  Attempt: was the scanner's flip/quality gate right to discard that raw pose?
  Only the harness can answer it, because only the harness holds Ground Truth.
  One of `goodPoseRejected` (the discarded pose agreed with truth — the gate
  over-rejected), `badPoseRejected` (the pose diverged from truth, or landed on a
  Climber-absent frame where no pose belongs), or `truthUnknown` (no raw pose, or
  no usable truth geometry to check against).
- **Truth bbox** — the padded extent of a truth frame's core joints, computed
  backend-side. The padding exists because 13 joints do not span a Climber's
  silhouette (no crown of the head, no hands past the wrists, no feet past the
  ankles). It is what the scanner's Adaptive Crop *should* have covered, so it is
  the reference for every crop-quality measure.
- **Crop containment** vs **crop IoU** — two different questions about one
  Adaptive Crop, deliberately not merged. Containment is the share of the Truth
  bbox inside the searched region ("did we look where the Climber was"); IoU also
  penalises a region far larger than the Climber ("did we look *tightly*"). A
  correctly-placed but oversized crop scores perfectly on the first and poorly on
  the second, so reporting only IoU would read crop *size* as crop *error*.
- **Miss Cause** — why one missing Detector Attempt found no Climber:
  `climber-absent` (truth says nobody is there — a correct miss),
  `crop-misplaced` (the crop excluded the Climber *and* was the only place
  searched), `adverse-conditions` (everything was searched and condition flags
  fired), `unexplained` (everything searched, conditions clean, still lost).
  `crop-misplaced` is a causal claim, so it requires that no full-frame reacquire
  ran: when the scanner also searched the whole frame and still failed, the crop
  cannot be what lost the Climber, however badly placed it was. Crop placement is
  measured on every miss regardless — the two facts are reported side by side and
  neither may be inferred from the other.
- **Over-rejection rate** — the share of *truth-checkable* rejections whose
  verdict is `goodPoseRejected`. Reported per Run so scanner flip-gate changes are
  measurable batch-over-batch, and under two denominators: over all checkable
  rejections, and over Climber-present ones only. Climber-absent rejections are
  correct by construction, so including them measures how much of the scanner's
  rejecting is aimed at empty frames rather than how well the gate judges a pose.
- **Non-conformance cause** — why a bundle failed the conformance gate (the
  near-identity fit of scanner coordinates onto Ground Truth that quarantines a
  bundle from pooled metrics), which the gate's own pass/fail verdict cannot
  say: `sparse-match` (the detector supplied too little
  to fit — too few matched-present frames, or too small a share of present
  Detector Attempts accepted) or `suspected-mistrack` (ample accepted detections
  and the fit still misses identity — the appearance-stitch signature the gate was
  built for). The verdict is unchanged by the split; the cause only routes the
  record. Only `suspected-mistrack` reaches the **truth-repair worklist**,
  because re-seeding Ground Truth for a run whose detector found almost nothing
  repairs nothing.
- **Evidence generation** — which detector evidence an evaluation record was
  scored from: `attempts` (the canonical **Detector Attempt** stream) or
  `legacy-frames` (the dense playback `frames[]` fallback used before the scanner
  exported attempts). Records written before the marker existed read as
  `unknown` — not attempt-backed, and never claimed as such. When one
  **video + truth revision** pairing carries both generations, only the
  attempt-backed record pools: the pairing is counted once and the two
  generations never blend (the legacy record's appVersion differs, so a
  generation change would otherwise read as a scanner change). **Superseded**
  records are held out of aggregation only — they stay on disk, readable, and are
  listed by name in the report. Every pooled report section states the
  generation(s) it aggregates, because a pool can still legitimately span
  generations across *different* videos.

## The condition → detection vocabulary

- **Predictor** — a *condition* of the video that might drive detection quality:
  a computed image stat (reference/per-frame luma mean, stdDev, Laplacian
  **sharpness**), motion magnitude, climber coverage, or a **hand label**
  (route orientation, camera angle, occlusion, camera stability, …). Hand labels
  are written by the scanner at calibration into `setup.json.analysisInputs`
  (snake_case keys matching `runs.LABEL_KEYS`) and are advisory context only.
  They may stratify reports or preserve human notes, but computed pixel
  conditions and Detector Attempts are the primary analysis predictors. The
  harness upload no longer collects these labels.
- **Outcome** — a measure of *how good detection actually was*. The trusted pose
  Outcome is **`overlayQuality`** (the scanner's end-to-end 0..1 verdict) plus
  **`badStretches`** (spans the overlay was visibly wrong). The ORB Outcome is
  **cross-match separation** (below). An Outcome is validated against human
  judgment, never assumed.
- **Symptom** — a self-reported detector *reaction* that is often mistaken for an
  Outcome but is partly circular: `detectionRate`, `flipRate`, `confidence`,
  `gapsRefined`, `limbExpandedFrames`. Symptoms are Predictors of interest, not
  ground truth. (A high `flipRate` is the flip detector firing hard, not proof the
  pose was wrong.)
- **Proxy** — the per-frame `kp_count` / `mean_score` derived from *exported*
  frames. Because exported frames are already interpolated / gap-filled /
  smoothed, the Proxy is **not raw detector output**. Distinguished from
  raw-detect success, which comes from per-frame **provenance**
  (`source: raw | interpolated | filled | flipDiscarded | limbExpanded`).
- **Detection Error** — a per-frame discrepancy derived by the harness after
  joining scanner evidence to Ground Truth: missing Climber, hallucinated pose on
  an absent frame, wrong subject, flip/rotation, distortion, drift, or stale raw
  detector output. Causes are discovered by correlating Detection Errors against
  Detector Attempts, crops, and computed pixel conditions, not by accepting
  hand-authored frame metadata as truth.

## ORB cross-match

- **Cross-match** — matching one Video's wall-crop features (train) against
  another Video's `final_frame.png` (query), over all ordered pairs.
- **Cross-match ground truth** — a pair is a **same-route** positive iff the two
  Videos share `route_folder`, else a **cross-route** negative.
- **Route-ID separation** — the gap between the same-route and cross-route
  inlier-ratio distributions. Wide separation = ORB robustly identifies a wall
  under real condition variation; narrow = it doesn't. The headline ORB Outcome.
