# Handoff: detector-attempt evidence for backend analysis

**Audience:** an agent working in the **Beta Scanner** repo (the Next.js pose/ORB
app). This is the scanner-side contract for the next analysis-harness evidence
stream. The scanner emits evidence; the harness owns Ground Truth comparison,
Detection Error derivation, report wording, and recommendations.

## Why

Issue #68 showed that dense exported pose frames are the wrong source for detector
failure analysis. After the scanner reposted per-frame `source` provenance on
2026-07-24, most held poses were continuity output (`interpolated` or `filled`),
while only 467 held frames were actionable raw stale evidence. The current posted
pose frames still do not carry per-frame `climber`/`wall` region stats, so the
harness cannot correlate failures with the pixels the detector actually saw
without decoding the local video again.

The fix is to post one canonical `detectorAttempts[]` stream for dev Analyze.
Each row records one MediaPipe attempt and the scanner decisions around it before
playback interpolation, filling, smoothing, or constraints can obscure detector
behavior.

## Payload authority

- **Ground Truth** owns expected Climber presence and pose.
- **`detectorAttempts[]`** owns scanner detector evidence: attempts, raw selected
  keypoints, accepted keypoints, crop/reacquire behavior, and candidate metadata.
- **Computed pixel conditions** are primary predictors for analysis. They should
  describe the region searched on that attempt.
- **`setup.json.analysisInputs`** are advisory metadata with provenance. Keep the
  entry UI and payload, but do not treat those labels as truth for main scoring.
- Scanner-side scoring, if still posted for the dev UI, is preview evidence only;
  backend evaluation remains authoritative.

## Dev Analyze behavior

For harness Analyze and Batch Analyze, run the production detector and Adaptive
Crop logic at analysis cadence:

- force `frameStep = 1` on the sampled 100 ms timeline;
- keep Adaptive Crop and full-frame reacquire behavior;
- disable Adaptive Refinement because stride 1 already visits the analysis grid;
- keep dense playback frames available for the harness view if needed;
- exclude dense playback frames from current detector evidence.

This is intentionally not "saved Quality Tier cadence." It is production detector
logic observed on every analysis grid frame.

## Pose payload

Post the pose body append-only as today, stamped with the existing setup, app, and
Ground Truth identifiers. Add:

```json
{
  "detectorAttempts": [
    {
      "timestamp": 12.3,
      "status": "accepted",
      "initialSearchRegion": { "x": 0.1, "y": 0.2, "w": 0.4, "h": 0.5 },
      "detectionRegion": { "x": 0.1, "y": 0.2, "w": 0.4, "h": 0.5 },
      "reacquireAttempted": false,
      "reacquired": false,
      "rawKeypoints": [],
      "acceptedKeypoints": [],
      "searchConditions": null,
      "reacquireConditions": null,
      "candidateCount": 1,
      "rejectedCandidateCount": 0,
      "selectionMethod": "tracked"
    }
  ]
}
```

### Status

Use exactly:

- `accepted` - scanner accepted a selected Climber pose.
- `missing` - no selected Climber pose was available after the scanner's search
  and any full-frame reacquire.
- `flipRejected` - a selected pose was rejected by the Landmark Flip gate.
- `qualityRejected` - a selected pose survived flip handling but was removed by
  the scanner's quality/filtering pass.

`limbExpanded` is an accepted pose source, not a rejection status.

### Keypoints

- `rawKeypoints` is the selected MediaPipe Climber pose mapped to full-frame
  normalized coordinates before flip rejection, filtering, interpolation,
  smoothing, or constraints. Include it whenever MediaPipe returned a selected
  pose, including rejected attempts.
- `acceptedKeypoints` exists only for `status: "accepted"` and carries the
  scanner-accepted keypoints for that attempt.

### Regions

- `initialSearchRegion` is the normalized rectangle first fed to MediaPipe.
- `detectionRegion` is the normalized rectangle that produced `rawKeypoints`.
- Full-frame search is always `{ "x": 0, "y": 0, "w": 1, "h": 1 }`.
- `null` means unknown or not applicable. Never use `null` to mean full frame.

### Reacquire

- `reacquireAttempted` is true when the initial Adaptive Crop missed and the
  scanner attempted a full-frame fallback.
- `reacquired` is true only when that fallback found and accepted the Climber.
- When reacquire succeeds, preserve both the failed `initialSearchRegion` and
  the successful full-frame `detectionRegion`.

### Conditions

- `searchConditions` describes the pixels in `initialSearchRegion`.
- `reacquireConditions` describes the fallback full-frame region when fallback
  ran.
- Use the same luma mean, stdDev, sharpness, and condition flags vocabulary used
  by Scan Diagnostics / Reference Frame Metadata.

### Candidate metadata

Keep this compact:

- `candidateCount`
- `rejectedCandidateCount`
- `selectionMethod`: `tap | tracked | strongest`

Do not post every MediaPipe candidate pose in this iteration.

## Iteration 2 additions (2026-07-25)

The first attempt-backed corpus (2026-07-24, 68 runs) validated the v1 stream and
exposed five evidence gaps. All additions below are **additive and optional** —
readers fail open, and a v1 payload stays valid. Rationale and the behavior
changes they support:
[scanner-detection-improvements.md](scanner-detection-improvements.md).

- **`searchConditions.wall`** — currently always `null`; populate it with the
  same stats vocabulary as `climber`/`overall`. Needed to separate
  climber-region darkness from whole-scene darkness.
- **`bestUnselectedCandidateScore`** — the top confidence among MediaPipe
  candidates that were *not* selected/accepted on this attempt (`null` when
  there were none). Distinguishes a hard miss (nothing seen) from a near miss
  (candidate just under threshold) and lets the harness evaluate acceptance
  thresholds without shipping full candidate poses.
- **`reacquireSteps[]`** — when the expanding-ladder reacquire ships, export the
  ordered regions tried: `[{ "region": {x,y,w,h}, "found": bool }, ...]`. The
  single `reacquireAttempted`/`reacquired` bits stay for compatibility.
- **`synthesizedJoints[]`** — on accepted attempts whose source is
  `limbExpanded`, the joint names that were synthesized rather than detected,
  so backend PCK can score detected and expanded joints separately.
- **`inferenceMs`** — wall-clock per-attempt MediaPipe latency, so stride-1 dev
  Analyze cost is measurable before any always-on cadence change.

Do not add `selectionDistance` or full candidate poses; both remain explicitly
deferred.

## Backend compatibility

The harness will prefer `data.detectorAttempts[]` when present. Older runs that
only carry `data.frames[]` remain readable, but dense playback frames are treated
as legacy/proxy evidence. Missing `detectorAttempts[]` means the detector attempt
stream is unknown; it must not be inferred as raw detector success.

## Acceptance checklist

- Dev Analyze posts `detectorAttempts[]` for accepted, missing, flip-rejected,
  and quality-rejected outcomes.
- Raw rejected selected keypoints are preserved before scanner-side mutation.
- Crop regions are normalized and full frame is explicit.
- Reacquire attempts distinguish attempted fallback from successful rescue.
- Attempt-level pixel conditions are present for the initial search region.
- Dense playback frames do not drive current backend detector scoring.
- Normal user-facing scan behavior is unchanged.
