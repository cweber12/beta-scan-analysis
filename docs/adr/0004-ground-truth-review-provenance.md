# ADR 0004 - Ground Truth review provenance separates agreement from accuracy

- Status: Accepted (superseded in part by ADR 0005)
- Date: 2026-07-15

> **Note (2026-07-20):** ADR 0005 supersedes the weighting of
> `human-flagged-absent` below. The manual-absent button has been removed and
> ViTPose auto-absence is now reliable, so manual absent flags are deprecated: they
> are **excluded from scoring** (not accuracy-tier evidence) and presence comes from
> ViTPose's `state`, never the flag. The vocabulary, `setupHash` provenance, and the
> `auto` / `human-flagged-wrong` decisions in this ADR still stand.

## Context

ADR 0003 makes ViTPose++ an independent scaffold for beta-scanner's human-authored
Ground Truth. That only breaks detector circularity if downstream evaluation can
tell which Ground Truth frames were actually human challenged.

beta-scanner is moving to an auto-accept workflow: ViTPose-seeded frames are
accepted by default, and the human only flags frames as absent or wrong. That
inverts the old meaning of `verified: true`. It can no longer mean "a human
attested this frame"; it means "the review loop did not reject this frame." If
auto-accepted frames are written indistinguishably from human-corrected frames,
the accuracy tier silently becomes ViTPose grading scanner output.

The manual run at
`analysis/mandala-the/The_Mandala_V12_uncut_Bishop__CA__bouldering__cl_20260711-141635`
demonstrates the agreed shape: every frame carries `review: "auto"`, and the
artifact carries the originating `setupHash`.

## Decision

`ground-truth.json` carries top-level setup provenance and per-frame review
provenance:

```json
{
  "version": 1,
  "jointSet": ["nose", "left_shoulder"],
  "frames": [
    {
      "frameIndex": 0,
      "timestamp": 0,
      "state": "present",
      "joints": {},
      "review": "auto",
      "verified": true
    }
  ],
  "setupHash": "<setup.json setupHash>",
  "groundTruthHash": "<content hash>",
  "updatedAt": "2026-07-15T07:05:25.776Z"
}
```

`frames[].review` is required for new beta-scanner writes and has this closed
vocabulary:

- `auto` - the ViTPose scaffold was auto-accepted because the human did not flag
  the frame.
- `human-flagged-wrong` - the human reviewed the frame and marked the scaffold
  pose as wrong.
- `human-flagged-absent` - the human reviewed the frame and marked the climber
  absent.

`setupHash` is required on new `ground-truth.json` writes. It must equal the
`setupHash` from the `setup.json` used to choose crops, panning, quality tier, and
requested timestamps for the ViTPose scaffold.

`verified` remains tolerated as a legacy/UI compatibility flag, but it is not the
authority for tiering. Consumers must use `review` when separating evidence.

## Consequences

- `review: "auto"` frames are **agreement-tier evidence** only. They can show that
  beta-scanner output agrees with an unchallenged ViTPose scaffold, but they must
  not be counted as accuracy-tier human truth.
- `human-flagged-wrong` and `human-flagged-absent` frames are human-reviewed
  evidence and may contribute to accuracy-tier evaluation under the scoring rules
  that consume Ground Truth.
- `setupHash` makes a Ground Truth artifact auditable against the calibration that
  produced its scaffold. If `setup.json` changes later, stale Ground Truth can be
  detected instead of silently reused.
- Older `ground-truth.json` files without `review` or `setupHash` are legacy
  artifacts. Treat them conservatively as unknown provenance unless manually
  migrated.

## Alternatives considered

- **Keep overloading `verified`** - rejected because auto-accept changes its
  meaning and would make tiering depend on an ambiguous boolean.
- **Top-level review mode only** - rejected because a single artifact can contain
  mostly auto-accepted frames plus a few human flags.
- **Use free-form reviewer notes** - rejected because evaluation needs a small,
  machine-checkable vocabulary.
