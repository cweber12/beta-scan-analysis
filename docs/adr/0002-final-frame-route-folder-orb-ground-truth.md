# ADR 0002 — `final_frame.png` + `route_folder` are the ORB match ground truth

- Status: Accepted
- Date: 2026-07-11

## Context

To *improve* ORB detection we need an outcome that measures **matchability** — how
well wall features match a target — not just reference-feature richness
(`refKeypointCount`), which is all the corpus records today. The product's real
ORB task is matching a scan's wall to an uploaded **route photo**, but no route
photo is present in the bundles at scan time, and sourcing one per video is a
heavy, open-ended data-collection effort.

Two facts make a cheaper ground truth available from data already committed:

1. Every bundle already commits **`final_frame.png`** — a real photo of the same
   wall from the same session.
2. Multiple Videos exist per Route, grouped by **`route_folder`**.

## Decision

Define the ORB outcome as an **all-pairs cross-match**: train = a Video's
wall-crop ORB features on its reference frame; query = another Video's
`final_frame.png`. A pair is a **positive** iff the two Videos share
`route_folder`, else a **negative**. The headline metric is the **separation**
between same-route and cross-route inlier-ratio distributions (plus route-ID
precision/recall). The batch runs in the scanner repo (which owns the CV code) and
emits one `reports/orb_match_matrix.json`; the harness renders it.

## Consequences

- A real, condition-sensitive ORB outcome with **near-zero new data cost** —
  runnable immediately on the committed corpus.
- It measures *matchability of the same physical wall under real condition
  variation* (blur, contrast, angle), which is exactly what degrades the product
  path — even though it is a **surrogate** for the literal reference→route-photo
  match.
- Same-route pairs filmed from very different angles/zooms may legitimately fail
  to match; that is signal (ORB's viewpoint fragility), not noise — but it means
  "same route" is an *upper bound* on what ORB can be expected to match, not a
  guarantee.

## Alternatives considered

- **Route-photo corpus** — most faithful to the product, but requires sourcing and
  aligning a photo per Route. Left as a future upgrade; the cross-match can be
  re-pointed at real photos later without changing the harness renderer.
- **`final_frame` × `final_frame`** — simpler and symmetric, but includes climber
  / background clutter and skips the wall-crop path the product actually uses.
