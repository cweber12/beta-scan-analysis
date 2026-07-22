# ADR 0006 — the ViTPose seed contract is `seed_tap` + `seed_region`, decoupled from the Climber Crop

- Status: Accepted
- Date: 2026-07-22
- Amends: ADR 0003 (the `POST /api/vitpose` seed request shape)

## Context

ADR 0003 exposed `POST /api/vitpose` with a Climber selection expressed as
`climber_point` + `climber_crop` (plus `wall_crop`, `panning`). Those two fields
did double duty: `climber_point` was the tap that anchors **Climber Identity** (which
tracked person is the climber), and `climber_crop` was reused as the **seed gate** —
the region a candidate track's box had to fall inside to be accepted as the seed.

Overloading the Climber Crop as the seed gate coupled two independent concerns:

- The **Climber Crop** is a Video Stats input: it bounds the region whose luma /
  Laplacian region stats the harness computes (`video-stats.json`). It is drawn to
  frame the climber's body for *condition* measurement.
- The **seed gate** only needs to disambiguate *which* detected track is the climber
  at seed time. It wants to follow the tap, not the stats crop, and it wants to admit
  a slightly larger neighborhood (people drift at the edges of a tight body crop).

The beta-scanner harness (branch `feat/harness-vitpose-seed-region`) had already split
these on its side, forwarding a dedicated `seed_tap` + `seed_region` alongside the
legacy fields. The harness backend still centered the legacy names and used
`climber_crop` as the gate, so scanner and harness were drifting: a scanner that draws
a `seed_region` distinct from its Climber Crop would have that intent silently ignored,
and mixed-version deployments had no way to tell which contract a given harness spoke.

## Decision

**`seed_tap` + `seed_region` are the `POST /api/vitpose` seed contract of record.**
`climber_point` / `climber_crop` remain accepted as **backward-compatible aliases**
during the migration.

- **`seed_tap`** — the normalized tap `{x, y, t?}` that anchors Climber Identity.
  Its optional `t` anchors candidate selection to the **nearest tapped frame**, so a
  later-frame retap seeds the correct climber when several people appear near the clip
  start. When `seed_tap` is null (or its `t` is null), seeding falls back to the prior
  global/full-frame selection — robustly, not with an error.
- **`seed_region`** — the normalized `{x, y, w, h}` **seed gate**, decoupled from the
  Climber Crop. A candidate track's box passes the gate when its center falls inside
  `seed_region` expanded by a fixed pad on each side. When `seed_region` is null the
  gate is open (any track is eligible).
- **Precedence is deterministic:** when both a new field and its legacy alias are
  present, the **new field wins** (`_to_vitpose_request` in `app.py` resolves
  `seed_tap ?? climber_point`, `seed_region ?? climber_crop`). Legacy-only clients seed
  exactly as before.
- **Capability signalling:** `GET /api/contract` advertises
  `capabilities.decoupledSeed: true`. This is **additive** — `apiVersion` stays `1`.
  Scanners gate the new fields on this flag so a mixed-version deployment degrades
  visibly instead of drawing a `seed_region` that an old harness ignores.

Internally (`vitpose_job.py`) `VitPoseRequest` and the seeding helpers name their
fields `seed_tap` / `seed_region`, so the seed gate is decoupled from the Climber Crop
throughout. **Video Stats keeps its own genuine `climber_crop`** region stats — that
field is unrelated to seeding and is unchanged. The cross-program `seedDebug` output
keys (`tap` / `crop` / `mode` / `seedFound`) are **left unchanged** so the scanner's
existing debug reader keeps working; only the internal field names and the null-seed
warning text (now `seed_tap.t`) moved.

`setupHash` provenance is untouched: `vitpose.json` still stamps the `setupHash` it ran
under and the evaluate trusted-pairing rules (ADR 0004) are unchanged by this ADR.

## Scope boundary with issue #45

This ADR governs the **seed request** the scanner sends *to* the harness
(`POST /api/vitpose`) — how the climber is anchored and gated for ViTPose scaffolding.

Issue **#45** (`detectionAnnotations` ingest) is a **separate concern** and does not
overlap: it defines a `setupHash`-stamped block the scanner *writes into the bundle* to
refine per-frame detection quality (distractor / failure-class annotations that the
`analysis_pipeline` reads), layered on the Ground Truth review provenance of ADR
0004/0005. It touches neither the seed request nor `seed_tap`/`seed_region`. The two
share only the cross-repo handoff *style*, not any field. A change to the seed contract
must not be justified by, or entangled with, annotation ingest, and vice versa.

## Consequences

- The seed gate follows the tap and a purpose-drawn `seed_region`, independent of the
  Climber Crop, so a tight body crop no longer forces a tight seed gate and Video Stats
  can frame the body however condition measurement wants.
- Scanner and harness speak one seed contract of record; the `decoupledSeed` capability
  lets each side gate on the other without an `apiVersion` bump, keeping mixed-version
  rollout safe.
- Legacy `climber_point` / `climber_crop` clients keep working unchanged; the alias path
  can be dropped once downstream adoption completes (a future ADR, not this one).
- The bundle vocabulary and the evaluate pairing are unaffected; only the request seam
  and its capability advertisement changed.

## Alternatives considered

- **Keep overloading `climber_crop` as the seed gate** — rejected: it couples an
  unrelated Video Stats input to seed disambiguation and blocks the scanner's already-
  shipped `seed_region`; the drift would have been silent.
- **Rename with a breaking `apiVersion` bump (drop the legacy aliases now)** — rejected:
  it would strand every existing scanner build mid-migration. The additive alias path
  plus a `decoupledSeed` capability flag lets both repos roll forward independently.
- **A new endpoint (`POST /api/vitpose/v2`)** — rejected: the change is a field-level
  refinement of the same job, not a new operation; a capability flag on the existing
  contract carries it with far less surface.
