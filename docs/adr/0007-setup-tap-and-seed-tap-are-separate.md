# ADR 0007 — the setup tap and the ViTPose seed tap are separate values

- Status: Accepted
- Date: 2026-07-26
- Amends: ADR 0006 (the decoupled seed contract), which separated the seed *region*
  from the Climber Crop but left the seed *tap* overloaded
- Issue: [#101](https://github.com/cweber12/beta-scan-analysis/issues/101)

## Context

`setup.json.climberPoint` was doing two unrelated jobs at once:

- the **setup tap** — the calibration gesture that anchors MediaPipe's Climber
  selection for the scanner's own detection Run, made once when the Bundle is
  calibrated; and
- the **seed tap** — the point that tells the ViTPose scaffold which tracked person
  is the Climber, which a human legitimately re-taps whenever the scaffold seeds on
  the wrong subject.

They were byte-identical in 80 of 90 Bundles because they started life as the same
gesture. Then re-seeding wrote back over `climberPoint`, and the setup tap moved with
it: **27 Bundles now carry a setup tap sitting mid-climb**, and in 24 of those the
Climber's hips had already risen 5–47% of frame height before the tap. Nothing
detected this, because the calibration hash matched either way — the hash is computed
over the calibration, and the calibration is what changed.

Two consequences followed. Seven Bundles carry a scaffold seeded from a tap that is no
longer on disk (a **stale scaffold**, indistinguishable from a current one). And the
setup tap could not be adopted as the **climb start** — the obvious source, since a
single calibration gesture already says where the climb begins — because in 24 Bundles
that would have discarded real climbing, up to 28 seconds and nearly half a wall's
height in the worst case.

## Decision

**The setup tap and the seed tap are two distinct values, initially equal.**

- **Setup tap** (`setup.json.climberPoint`) — frozen at initial calibration. It seeds
  MediaPipe and it defines the **climb start**. Re-seeding must never write to it.
- **Seed tap** (`setup.json.seedTap`, sent as `seed_tap` on `POST /api/vitpose`) —
  identifies the Climber for the ViTPose scaffold *only*. It is free to move on every
  re-seed, and its correction propagates **backwards as well as forwards** over the
  whole trajectory (the stitcher already walks both directions from the seed, so
  re-tapping late in a clip fixes the early frames too).
- A Bundle that has never been re-seeded has the two equal and behaves exactly as it
  does today. Until the scanner adopts the split, the harness sees one tap and is
  unchanged — the fallback in `resolve_climb_window` reads `climberPoint.t` and,
  finding no split, yields today's behaviour.
- **Climb window.** `climb_start` comes from the frozen setup tap; `climb_end` from an
  explicit end marker (`setup.json.climbEnd`), because there is no gesture to infer a
  topout from. Both may be absent, and an absent window admits every frame — so the
  feature lands before every Bundle is marked. Within the job, the window bounds both
  the tracking and the posing legs: out-of-window frames are never tracked, never
  posed, and (issue #101's evaluate slice) never scored.
- **Seed hash.** A hash over the seed tap, seed region, climb window and video identity
  is stamped into `vitpose.json` as `seedHash`, so a scaffold **records which seed it
  was built from**. An unchanged hash skips the job and reports the skip; a changed one
  re-runs. This is one change wearing two hats — it is the correctness fix for silent
  staleness *and* the largest re-run saving — which is why they are not separable.
- **Capability signalling:** `GET /api/contract` advertises
  `capabilities.splitTaps: true`, additively, in the same style as `decoupledSeed`.
  `apiVersion` stays `1`.

**Cross-repo dependency.** beta-scanner must stop writing re-seed taps back into
`climberPoint` and must send the re-seed tap as `seed_tap`. Until it does, the harness
degrades to single-tap behaviour rather than erroring. See
`docs/handoffs/scanner-tap-split-adr0007.md`.

## Consequences

- Re-seeding the ViTPose scaffold is now free of side effects: a human can re-tap as
  many times as it takes to identify the right Climber without destroying the
  calibration MediaPipe depends on.
- The climb start becomes trustworthy, which is what makes the climb window (and,
  downstream, the `out-of-scope` absence reason) sound rather than a second source of
  contamination.
- Stale scaffolds are detectable for the first time: `seedHash` mismatch is the signal,
  where `setupHash` was structurally unable to provide one.
- **Existing Bundles are not migrated.** No migration can recover the intent of a tap
  that was already overwritten — the original setup tap is simply gone. This is the
  direct reason issue #101 ends in a corpus reset rather than a repair.
- Sequencing is load-bearing: the split must land *before* anything adopts the setup
  tap as the climb start.

## Alternatives considered

- **Keep one tap and re-derive the climb start some other way** (first tracked frame,
  first motion) — rejected: it invents a signal where the human already gave us one,
  and every candidate derivation is itself contaminated by the mis-tracking this work
  exists to fix.
- **Migrate the existing Bundles to the split contract** — rejected as impossible, not
  merely expensive: the overwritten value is unrecoverable, and a migration that
  guessed would launder a guess as calibration.
- **Freeze `climberPoint` and let the scaffold re-seed from an in-memory tap only** —
  rejected: the seed tap then has no on-disk home, so a scaffold could never record
  which seed it was built from, and the staleness defect would survive intact.
- **Bump `apiVersion` for the new request fields** — rejected for the ADR 0006 reason:
  the change is additive and a capability flag lets both repos roll forward
  independently.
