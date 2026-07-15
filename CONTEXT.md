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
  `docs/adr/0003`.
- **Ground Truth** (`ground-truth.json`) — beta-scanner's per-frame pose truth
  artifact, authored from the ViTPose scaffold plus human flags. New artifacts carry
  top-level `setupHash` and per-frame `review` provenance. `review: "auto"` is
  agreement-tier evidence only; human-flagged frames are the accuracy-tier evidence.
  See `docs/adr/0004`.
- **Route** — a physical climb, identified by its `route_folder`. Multiple
  **Videos** of the same Route are the norm (different sessions/angles/lighting).
- **Run** — one detection execution on one Video, recorded as a paired
  `<run_ts>_pose.json` + `<run_ts>_orb.json`. **The Run is the unit of
  statistical inference** — coefficients are summarized across Runs, not pooled
  across frames.

## The condition → detection vocabulary

- **Predictor** — a *condition* of the video that might drive detection quality:
  a computed image stat (reference/per-frame luma mean, stdDev, Laplacian
  **sharpness**), motion magnitude, climber coverage, or a **hand label**
  (route orientation, camera angle, occlusion, camera stability, …). Hand labels
  are written by the scanner at calibration into `setup.json.analysisInputs`
  (snake_case keys matching `runs.LABEL_KEYS`); the harness upload no longer
  collects them.
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

## ORB cross-match

- **Cross-match** — matching one Video's wall-crop features (train) against
  another Video's `final_frame.png` (query), over all ordered pairs.
- **Cross-match ground truth** — a pair is a **same-route** positive iff the two
  Videos share `route_folder`, else a **cross-route** negative.
- **Route-ID separation** — the gap between the same-route and cross-route
  inlier-ratio distributions. Wide separation = ORB robustly identifies a wall
  under real condition variation; narrow = it doesn't. The headline ORB Outcome.
