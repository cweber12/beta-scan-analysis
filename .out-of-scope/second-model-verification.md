# Second-model auto-verification of Ground Truth

Running a second, architecturally independent pose model (RTMPose, Sapiens, or
similar) over the sampled timestamp grid, marking frames where the two models agree
as auto-verified, and using those frames to populate the **accuracy tier**.

Rejected 2026-07-30 after grilling the independence premise against the corpus.

## Why this is out of scope

### The problem it solves is 2.7% of truth frames, and already quarantined

The whole corpus has now been human-reviewed. Every stretch the human marked
`human-flagged-wrong` is a *correct ViTPose detection on the wrong person* — an
identity failure, not a joint-accuracy failure. Measured over 86 bundles:

| | |
|---|---:|
| Truth frames | 78,465 |
| `review: auto` | 76,349 (97.3%) |
| `human-flagged-wrong` | **2,113 (2.7%), in 7 bundles** |
| `human-flagged-absent` | 3 |

Six of those seven bundles are already excluded from trusted pooled metrics by the
#15 conformance gate. The seventh (`planet-x/R0Z6c1zlic0`) carries 10 bad frames out
of 742.

A second model plus its dependency stack plus a corpus-wide GPU re-run would exist
to auto-verify what complete human review has already attested.

### It shares the stage that actually fails

The proposal specified *"reusing the existing job's video decoding and Climber
tracking."* Both models would therefore receive the same box from the same
YOLO + ByteTrack track, selected by the same seed gate and the same stitch recovery.
Every error in the corpus is an error of *which person was tracked*, upstream of both
pose heads.

On such a frame both models pose the wrong person accurately, agree within any
threshold, and the frame is stamped auto-verified — promoting a known-bad frame into
the corpus's highest-trust evidence class. The failure is correlated at 1.0 by
construction, so no choice of second model reduces it.

Drawing independence properly — separate detector and tracker per stack, sharing only
decoded pixels — is defensible, but roughly doubles seeding cost to verify a 2.7%
defect rate that human review already covers.

### Agreement between two COCO-supervised models is not accuracy

The incumbent is `usyd-community/vitpose-plus-base` run through its COCO expert head.
The proposed alternatives are trained on largely the same public human-pose corpora.
The stated motivation — shared blind spots on heavily occluded limbs against the wall
— is precisely the distribution gap two COCO-supervised top-down models have in
common. Concordant error is highest exactly where the climbing poses are hardest,
which is where the auto-verified label would be least deserved.

Measuring the residual concordant-error rate would itself require human-labelled
frames, which is the labour the proposal exists to avoid.

## What happens to the accuracy tier instead

It stays, structurally present and permanently empty, and reports itself as
*not computable — no verified truth frames* rather than rendering an empty band. See
`docs/adr/0010`. Deleting it was considered and rejected: with only one pose number in
the report, `agreement` gets read as accuracy, which is the ADR 0003 circularity
re-entering through the report layer.

## What the flagged frames are for now

The 2,113 `human-flagged-wrong` frames are the only human-attested tracker-failure
labels in the corpus. They are excluded from scoring (ADR 0004/0005) but are the
validation set for truth-side identity signals — including the finding that 91% of
`suspected-mistrack` non-conformance lands on bundles whose truth is attested clean.

## Prior requests

- [#12](https://github.com/cweber12/beta-scan-analysis/issues/12) — "feat: second-model auto-verification for the accuracy tier (deferred)"
