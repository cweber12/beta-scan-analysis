# ADR 0005 - Manual absent flags are deprecated; presence comes from ViTPose auto

- Status: Accepted
- Date: 2026-07-20
- Supersedes (in part): ADR 0004 (the accuracy-tier weighting of
  `human-flagged-absent`)

## Context

ADR 0004 gave `human-flagged-absent` frames the strongest presence weight in the
evaluation: the manual flag was the presence *authority* (overriding the scaffold
`state`), and such frames were routed into the **accuracy tier** as human-attested
presence negatives. Issue #11 implemented exactly that.

The premise behind that weighting no longer holds:

- beta-scanner's dev harness has **removed the manual "absent" button**. No new
  `human-flagged-absent` frames will be written.
- ViTPose auto-detects absence reliably: it flags every frame with zero seeded
  landmarks as absent, and never marks a frame that has seeded landmarks as absent.
  Auto-absence (`state: "absent"`, `review: "auto"`) is now the trustworthy
  presence-negative signal.
- The existing manual absent flags were made under an older harness that did **not**
  identify absence consistently. They are a deprecated workflow's residue.
- They may be **stale or wrong**. As more frames are recorded and scaffolds are
  re-seeded, a re-seed can detect landmarks on a frame that still carries an old
  manual absent flag. ADR 0004's "flag overrides `state`" rule would then discard
  valid landmarks and score a correct scanner detection as a false positive.

A corpus scan (2026-07-20) found 37 `human-flagged-absent` frames, all currently
`state: "absent"` with no landmarks — so no *active* stale conflicts yet, but the
rule is primed to misfire on the next re-seed, and elevating these frames to
accuracy-tier evidence is unjustified given their provenance.

## Decision

- **Presence is always ViTPose's `state`.** The manual absent flag never overrides
  it. Auto-absence carries the presence-negative signal.
- **`human-flagged-absent` frames are excluded from every tier's scoring**, exactly
  like `human-flagged-wrong`. They remain counted in `truthFramesTotal` and are
  reported in `counts.review` and `counts.agreementSkipped.flaggedAbsent` so a
  record's frame math reconciles and the legacy flags stay auditable.
- **The accuracy tier has no trustworthy attestation source and stays empty.** No
  current `review` value is a positive human attestation; joints are never
  hand-attested. Second-model verification (issue #12) is the future accuracy
  source. The two-tier scaffold is retained so #12 can populate it without
  re-plumbing.

`human-flagged-wrong` handling is unchanged from ADR 0004 / issue #11: a known-bad
seed, excluded from scoring, surfaced in the skip accounting.

## Consequences

- Absence evidence now flows entirely from reliable ViTPose auto-detection; the
  scanner's hallucinations on truly-absent frames are still caught as presence false
  positives on `auto` frames.
- The stale-flag failure mode is closed: a re-seed that detects landmarks on a
  formerly hand-flagged frame is scored on its landmarks, not silently forced absent.
- Records regenerated under this ADR move `human-flagged-absent` counts out of the
  accuracy tier and into `counts.agreementSkipped`; `truthFramesVerified` returns to
  `0` across the corpus until issue #12.
- beta-scanner should eventually stop emitting `review: "human-flagged-absent"` (the
  button is already gone) and may migrate residual flags; until then, consumers in
  this repo treat them as deprecated per this ADR.

## Alternatives considered

- **Keep ADR 0004's accuracy-tier weighting** - rejected: the flags are a removed
  workflow's residue, are not trustworthy, and the presence-authority rule is a
  latent bug against re-seeds.
- **Exclude from accuracy but keep as agreement-tier absent evidence** - rejected:
  it retains a low-trust, deprecated signal for marginal benefit (all current such
  frames already agree with auto-absence, which is scored anyway) and keeps the
  stale-flag risk alive.
- **Drop the frames from the record entirely (no skip accounting)** - rejected: the
  record's frame totals would no longer reconcile and the legacy flags would become
  invisible to audit.
