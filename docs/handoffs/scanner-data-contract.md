# Handoff: scanner data contract for the analysis harness

**Audience:** an agent working in the **Beta Scanner** repo (the Next.js pose/ORB
app). **You do not need the analysis harness repo open to do this work** — it only
consumes what you produce. This doc tells you exactly what to emit and how the
harness will read it.

## Why

The harness (`youtube-downloader`) collects climbing videos into per-route
*bundles* and correlates video conditions against your detector's diagnostics to
find what makes pose/ORB detection fail. Two gaps block that today:

1. Your pose diagnostics ship `overlayQuality: null` and (usually)
   `badStretches: []`, and per-frame keypoints are already post-processed — so the
   harness has **no non-circular quality outcome** and its per-frame table is a
   proxy, not raw detector behavior.
2. ORB diagnostics record only reference-feature *richness*
   (`refKeypointCount`, `keyframeCount`) — **never how well features matched**, so
   "improve ORB" has no outcome to optimize.

This handoff has two phases. **Phase 1 is runnable today** on data already on disk
and delivers the first real ORB outcome. **Phase 2** enriches what every future
scan exports.

---

## The harness bundle layout you read/write against

Everything lives under the harness's local `analysis/` tree (same machine; the
`.mp4` binaries are present locally even though they're git-ignored):

```
analysis/<route_folder>/<video_key>/
    <video_key>.mp4        # the video (local only)
    final_frame.png        # last frame — the ORB query image (committed)
    metadata.json          # analysis_inputs (hand labels) + source_video (w/h/fps/duration)
    setup.json             # climberCrop, wallCrop, climberPoint, panning, qualityTier (normalized [0,1] boxes)
    detections/
        <run_ts>_pose.json # envelope: { video_key, run_ts, ..., data:{ diagnostics, frames[] } }
        <run_ts>_orb.json  # envelope: { ..., data:{ referenceFrameMeta, summary } }
```

`route_folder` = the folder two levels up from the video. Two videos share a route
iff they share `route_folder`. **This is the cross-match ground truth.**

---

## Phase 1 — all-pairs ORB cross-match (do this first)

### Goal
For every ordered pair of videos `(R, Q)`, measure how well `R`'s wall features
match `Q`'s final frame. Same-route pairs *should* match; cross-route pairs
*should not*. The spread between those two distributions is the ORB outcome.

### Algorithm
```
videos = every <route>/<video_key> dir under analysis/ that has final_frame.png
for R in videos:
    refImg   = decode R's reference frame          # fixed capture: frame 0;
                                                    # panning: first keyframe (v1)
    wallCrop = R.setup.json.wallCrop                # normalized {x,y,w,h}
    refFeat  = extractFeaturesFromCrop(cv, refImg, wallCropInPixels(wallCrop, refImg))
for Q in videos:
    queryFeat[Q] = extractFeatures(cv, decode(Q.final_frame.png))   # whole frame

for R in videos:
    matcher = createQueryMatcher over ... (or matchOrbFeatures per pair)
    for Q in videos:
        matches = matchOrbFeatures(cv, refFeat[R], queryFeat[Q])    # Lowe + Hamming cap
        H, inliers, reproj = computeHomography(cv, matches, refFeat[R], queryFeat[Q], gate)
        record row(R, Q)
```

**Reuse existing scanner code — do not reimplement:**
- `extractFeatures`, `extractFeaturesFromCrop` (`orbDetector.ts`)
- `matchOrbFeatures` / `createQueryMatcher` (`orbDetector.ts`) — already applies
  the Lowe ratio (0.75) + Hamming cap (64)
- `computeHomography` + `applyHomographyMatrix` (`matching/homography`)
- constants `ORB_FEATURES=3000`, `LOWE_RATIO=0.75`, `HAMMING_MAX_DISTANCE=64`

`wallCropInPixels`: multiply the normalized `{x,y,w,h}` by the reference frame's
pixel width/height to get the `OrbCropBox` (`extractFeaturesFromCrop` already
offsets keypoints back to full-frame coordinates).

Keep the diagonal (`R == Q`) as an upper-bound control — same session, so it
should score near-perfect.

### Output — one file, `reports/orb_match_matrix.json`
```json
{
  "generatedAt": "2026-07-11T15:00:00Z",
  "appVersion": "<git short sha>",
  "config": { "orbFeatures": 3000, "loweRatio": 0.75, "hammingMax": 64,
              "ransacReprojThreshold": <value used> },
  "pairs": [
    {
      "trainKey": "Bishop_The_Mandala_V12_20260710-142554",
      "trainRoute": "mandala-the",
      "queryKey":  "The_Mandala__v12__Bishop__CA_20260711-141532",
      "queryRoute": "mandala-the",
      "sameRoute": true,
      "matches": 148,
      "inliers": 96,
      "inlierRatio": 0.649,
      "homographyValid": true,
      "reprojErrorPx": 3.1
    }
  ]
}
```
Rules:
- One row per ordered `(R, Q)` pair (N² rows). Include the diagonal.
- `inlierRatio = inliers / matches` (0 when `matches == 0`).
- `homographyValid` = did `computeHomography` return a matrix that passed your
  existing validity gate (non-degenerate, within the reprojection threshold)?
  `false` when it returned null.
- `reprojErrorPx` = mean symmetric reprojection error over inliers; `null` when no
  valid homography.
- Write to the harness repo's `reports/` dir (git-ignored there) or print the path
  so it can be pointed at with `--matrix`.

### Environment note (decide this first)
Your app runs OpenCV as WASM on the browser main thread. A batch script over the
filesystem needs a headless CV runtime. Pick one and note it in the script's
README:
- **Node + `@techstark/opencv-js`** (same WASM API you already call — least code
  churn), or
- **Node + `opencv4nodejs`/native**, or
- a **headless-browser** runner that loads your existing bundle.

Frame decoding (reference frame + final_frame.png) can use `ffmpeg`/`sharp` or the
CV runtime's `imread` — the reference frame comes from the `.mp4`, the query from
the committed PNG.

### Phase 1 acceptance
- `reports/orb_match_matrix.json` exists with N² rows for N videos.
- Sanity: mean `inlierRatio` for `sameRoute:true` (off-diagonal) is clearly above
  the `sameRoute:false` mean. If not, the crop→feature direction or the
  route-folder grouping is wrong — stop and re-check before scaling up.

---

## Phase 2 — enrich what every scan exports (requires re-scanning videos)

All additions must be **back-compatible**: the harness treats any missing field as
`null`, so partial rollout is safe.

### 2.1 Populate the pose quality outcome
In the pose diagnostics `result` block (currently `overlayQuality: null`,
`badStretches: []`):
- **`overlayQuality`**: your existing end-to-end pose-quality score in `[0,1]`
  (whatever the field was designed to hold — wire it up, don't invent a new one).
- **`badStretches`**: `[{ "startSec": number, "endSec": number, "reason": string }]`
  — contiguous spans the overlay was visibly wrong/absent (lost track, sustained
  flip run, long gap-fill). `reason` is a short slug.

### 2.2 Per-frame provenance + per-frame conditions
Each element of `data.frames[]` currently carries `{ timestamp, keypoints[] }`.
Add:
- **`source`**: how this frame's pose was obtained —
  `"raw" | "interpolated" | "filled" | "flipDiscarded" | "limbExpanded"`.
  (`raw` = the detector actually detected it this frame; the others come from
  `interpolatePoseFrames` / `estimateMissingLandmarks` / `fillPersistentGaps` /
  the flip walk / limb-reach expansion.) This is the single most valuable field —
  it turns the harness's proxy table into real raw-vs-filled analysis.
- **`climber` / `wall`**: `{ "mean": n, "stdDev": n, "sharpness": n }` for this
  frame's climber-crop and wall-crop regions — the *same* luma/Laplacian
  computation you already run for the reference frame, applied per sampled frame.
  This lets the harness join conditions to raw-detect success **without decoding
  the git-ignored video**.

### 2.3 ORB per-keyframe match stats (panning captures)
For panning captures you already store ordered keyframes. Add consecutive-keyframe
match stats (keyframe *i* → *i+1*), same fields as the Phase 1 pair schema, under
`data.keyframeMatches[]`. This gives an in-scan ORB outcome that doesn't need the
cross-match batch.

### 2.4 Emit the reference frame
Write **`reference_frame.png`** into each bundle at scan time (the exact frame your
`wallCrop`/`climberCrop` were drawn on). This makes the Phase 1 train side
reproducible from the committed record — no local `.mp4` needed. (Alternative:
serialize the wall-crop ORB descriptors into the orb envelope; the PNG is simpler
and also feeds the harness's per-video cards.)

### Phase 2 acceptance
- A fresh scan produces `overlayQuality ∈ [0,1]`, a `badStretches` array,
  per-frame `source` + `climber`/`wall` stats, and `reference_frame.png`.
- Old bundles remain readable (fields simply absent).

---

## What the harness does with all this (for your context, no action needed)
- `overlayQuality`/`badStretches` become the **pose outcome**; detectionRate /
  flipRate / confidence are demoted to predictors/symptoms.
- Per-frame `source` + region stats drive a **per-frame failure timeline** and
  raw-detect-success correlations.
- `orb_match_matrix.json` renders an **NxN inlier heatmap** + same/cross-route
  separation and route-ID precision/recall.
- `final_frame.png` (+ `reference_frame.png`) show as thumbnails on per-video
  failure cards.
