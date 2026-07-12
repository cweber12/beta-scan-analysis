# ADR 0001 — `overlayQuality` is the primary pose outcome; self-metrics are symptoms

- Status: Accepted
- Date: 2026-07-11

## Context

The harness correlates video conditions against pose-detection quality. The
diagnostics the scanner ships include `detectionRate`, `flipRate`, `confidence`,
`avgKeypointCount`, and `gapsRefined`. It is tempting to treat these as the
"quality" being explained. They are not trustworthy as outcomes:

- They are **self-reported by the same pipeline whose quality we're judging**.
- Several are **circular**. `flipRate` is how aggressively the flip detector
  fired — a glitchy clip and an over-eager threshold produce the same number.
  `gapsRefined` / `limbExpandedFrames` measure how much *reconstruction* ran, not
  whether the result was right.
- The exported per-frame keypoints are post-processed (interpolated, gap-filled,
  smoothed), so per-frame `kp_count` / `mean_score` are a **proxy**, not raw
  detection.

The scanner already defines an `overlayQuality` field (an end-to-end 0..1
verdict) and a `badStretches` list — but both are unpopulated in every bundle in
the corpus (`overlayQuality: null` in 32/32 runs).

## Decision

Treat **`overlayQuality` + `badStretches` as the primary pose outcome**, and
demote `detectionRate` / `flipRate` / `confidence` / `gapsRefined` to
**predictors/symptoms**. Get the scanner to populate `overlayQuality` /
`badStretches` (see the Phase 2 data contract), and **validate** that
`overlayQuality` tracks perceived quality against a small hand-rated good/bad
sample of overlays before trusting it.

## Consequences

- Correlations answer "what conditions predict a *bad overlay*," not "what
  conditions make the detector *react*." Actionable for the stated goal.
- The pose-outcome sections of the report are **empty until the scanner ships the
  field and the corpus is re-scanned** — an accepted, sequenced cost.
- If human calibration shows `overlayQuality` does *not* track perceived quality,
  this ADR is reopened in favor of human ground-truth labeling.

## Alternatives considered

- **Human ground truth only** — most rigorous, but heavy manual effort and needs
  a rendering/labeling workflow up front. Kept as the fallback and as the
  calibration check, not the primary.
- **Keep self-metrics** — fastest, but leaves the whole analysis circular; it can
  never say whether detection was *correct*.
