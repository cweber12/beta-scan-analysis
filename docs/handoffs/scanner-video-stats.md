# Handoff: Video Stats prefill — POST crops at calibration, verify suggested labels

**Audience:** an agent working in the **Beta Scanner** repo (the Next.js pose/ORB
app). You do not need the analysis harness repo open to do this work — this doc
gives you the new endpoint contract, the artifact it writes, and what the
scanner's calibration flow must change so condition labels are *verified* by the
user instead of authored from memory.

**Companion docs:**
[scanner-data-contract.md](scanner-data-contract.md) (bundle layout, the
`analysisInputs` block this feature prefills) and
[scanner-calibration-freshness.md](scanner-calibration-freshness.md) (the
`setupHash` staleness rules this feature reuses).

Harness issue: [#23](https://github.com/cweber12/beta-scan-analysis/issues/23).

---

## The problem this solves

Almost every video-condition Predictor is a hand label picked from memory at
calibration, and most bundles ship with `"unknown"` in most fields. The harness
now computes continuous image stats automatically in two phases:

- **Phase 1 (source stats)** — whole-frame stats at download/import, stored in
  the bundle's `metadata.json` (`video_stats` block). No scanner involvement.
- **Phase 2 (region stats)** — wall color/texture, climber↔wall contrast, and a
  shadow block, computed from the calibration crops **when the scanner POSTs
  them**. Stored in a new per-bundle `video-stats.json` stamped with the
  `setupHash` it was computed under.

From these stats the harness derives **suggested labels** for the fields the
user previously authored blind. Your job: call the endpoint mid-calibration,
prefill the `analysisInputs` form from the response, and record per-label
provenance so the human-verified layer stays auditable.

---

## The endpoint: `POST {HARNESS_API_BASE}/api/video-stats`

Synchronous (unlike `/api/vitpose`): decodes ~30 sampled frames and responds in
a few seconds. Call it **after saving `setup.json`** (crops drawn) and before
showing the `analysisInputs` form.

Request (camelCase and snake_case field names are both accepted; everything but
`routeFolder`/`videoKey` is optional and falls back to the bundle's just-saved
`setup.json`, so the minimal call is just the two identifiers):

```json
{
  "routeFolder": "planet-x",
  "videoKey": "jGa4kCQkXaQ_20260711-185530",
  "climberCrop": {"x": 0.48, "y": 0.73, "w": 0.27, "h": 0.27},
  "wallCrop":    {"x": 0.24, "y": 0.0,  "w": 0.53, "h": 0.95},
  "climberPoint": {"x": 0.59, "y": 0.98, "t": 0},
  "panning": false,
  "setupHash": "<hash of the setup.json this calibration just saved>"
}
```

Response `200`:

```json
{
  "routeFolder": "...", "videoKey": "...",
  "setupHash": "<echoed provenance anchor>",
  "artifactPath": ".../video-stats.json",
  "regionStats": { "wall": {...}, "climberWall": {...}, "shadow": {...},
                   "panningFlagged": false, "sampledFrames": 30 },
  "suggestions": {
    "shadows": "none|solid|patchy",
    "climber_contrast": "low|medium|high",
    "wall_contrast": "low|medium|high",
    "motion_blur": "low|medium|high",
    "camera_stability": "steady|moving"
  }
}
```

Errors: `400` no wall crop resolvable (request or setup.json) or bad path; `404`
unknown bundle or missing video binary; `500` decode/compute failure. On any
error, fall back to the current hand-authoring flow — the endpoint is an
enhancement, never a gate on calibration.

Notes:

- `suggestions` keys are the snake_case `analysisInputs` label keys; only labels
  whose driving stat resolved are present. Treat a missing key as "no
  suggestion" and leave that field at `"unknown"` for the user.
- The harness writes `video-stats.json` itself — the scanner never writes this
  artifact. Recalibration must simply re-POST; the harness overwrites and
  re-stamps (`setupHash`), exactly the Ground Truth staleness pattern.
- Suggestion thresholds are corpus-fit constants in the harness
  (`video_stats.SUGGESTION_THRESHOLDS`, fit 2026-07-19 over 39 bundles).
  `camera_stability` is a strong fit (balanced acc 0.97 vs existing labels);
  the contrast/blur bands are distribution-calibrated first passes — expect the
  user to override them, that's the point of verify-not-author.

---

## Work items

### 1. Calibration flow reorder: crops drawn → POST → prefill → verify → save

After the user finalizes crops and `setup.json` is saved (with its new
`setupHash`), POST `/api/video-stats`, then render the `analysisInputs` form
**prefilled** from `suggestions`. The user confirms or overrides each field,
then the labels are saved into `setup.json.analysisInputs` as today
(snake_case keys, per the data-contract doc).

- Show suggested values visually distinct from user-set ones (e.g. a subtle
  "suggested" affordance) until confirmed.
- Fields with no suggestion (`route_orientation`, `camera_angle`, `occlusion`,
  `notes`) stay manual, defaulting to `"unknown"`/empty as today.

### 2. Per-label provenance in `analysisInputs`

Record, per suggested label, whether the saved value was auto-accepted or
human-overridden, so downstream analysis can weigh them differently. Write a
sibling block in `setup.json` (additive — the harness pipeline keeps reading
`analysisInputs` unchanged):

```json
"analysisInputsProvenance": {
  "shadows": "auto-accepted",
  "climber_contrast": "human-overridden",
  "camera_stability": "auto-accepted",
  "route_orientation": "human-authored"
}
```

Vocabulary: `auto-accepted` (suggestion kept as-is), `human-overridden`
(suggestion shown, user changed it), `human-authored` (no suggestion existed).
Labels untouched by the user in an old bundle simply have no provenance entry.

### 3. Shadows vocabulary migration (+ climber-shadow option)

Suggestions use a **structural** shadows vocabulary: `none | solid | patchy`
(replacing the intensity grades `low|medium|high`). Update the form options to:

```text
shadows: none | solid | patchy | climber | unknown
```

`climber` (the climber casts the significant shadow) is human-only — the
automation cannot detect it yet, so it is never suggested; keep it selectable.
Old bundles keep their legacy values; no migration of stored labels.

### 4. Re-POST on recalibration

Every calibration save that produces a new `setupHash` must re-POST
`/api/video-stats` so the artifact tracks the current crops. (If you already
rebuilt the calibration-save pipeline for the freshness handoff, this is one
more step in that same sequence, before the ViTPose job kicks off or in
parallel with it — the two are independent.)

### 5. (Optional) surface the ViTPose camera-angle estimate

The harness ViTPose job now writes a `cameraAngle` block into
`video-stats.json` (`{"estimate": "level|high|low", "shoulderHipRatio": ...,
"source": "vitpose", "setupHash": "..."}`). The `camera_angle` hand label
remains authoritative and manual; if you want, display the estimate as a hint
next to the field (it arrives asynchronously with the ViTPose job, not in the
`/api/video-stats` response).

---

## What the harness does with this (context, no action needed)

- `video-stats.json` region stats become continuous Predictor columns in the
  correlation pipeline (`vs_*`), alongside phase-1 `src_*` columns; a
  `vs_stale` flag fires when the artifact's `setupHash` differs from a run's.
- The 39 existing bundles were backfilled (both phases) on 2026-07-19, stamped
  with their current `setupHash`es — so prefill works from the first
  recalibration onward.

---

## Verification

1. Calibrate a bundle in the scanner: after crops are drawn, confirm the
   `analysisInputs` form arrives prefilled and each suggested field is marked.
2. Accept one suggestion, override another, author a manual field; confirm
   `setup.json` carries `analysisInputs` + `analysisInputsProvenance` with
   `auto-accepted` / `human-overridden` / `human-authored` respectively.
3. Recalibrate with different crops: `video-stats.json` in the bundle must show
   the new `setupHash` and a fresh `computedAt`.
4. Kill the harness service and calibrate: the form must degrade to the manual
   flow (all fields `"unknown"`, no provenance for unsuggested labels), not
   block.
5. Harness side: `python -m analysis_pipeline analysis -o reports` — the run
   table gains `vs_*` columns for the recalibrated bundle with `vs_stale` false.

---

## Status

> **Instructions for the implementing agent:** update the table and log below as
> you land work. One row per work item; Status is
> `not-started | in-progress | blocked | done`, plus the scanner commit/PR.

| # | Work item                                      | Status      | Commit/PR | Notes |
|---|------------------------------------------------|-------------|-----------|-------|
| 1 | Flow reorder: POST → prefill → verify → save   | not-started |           |       |
| 2 | Per-label provenance block                     | not-started |           |       |
| 3 | Shadows vocab migration + climber option       | not-started |           |       |
| 4 | Re-POST on recalibration                       | not-started |           |       |
| 5 | (Optional) camera-angle hint                   | not-started |           |       |

### Log

- 2026-07-19 — handoff written (harness agent). Endpoint, artifact, backfill,
  and fitted thresholds are live on the harness side.
- 2026-07-19 — merged to harness `main` (PR #27). Everything described here is
  now what a running harness service serves; the 39-bundle corpus carries both
  phases, stamped with current `setupHash`es. Scanner work can start.

---

## Suggested skills

- `/webapp-testing` — drive the calibration flow end-to-end to verify prefill,
  provenance, and the degraded (harness-down) path.
- `/verify` before committing, and `/code-review` on the finished branch.
