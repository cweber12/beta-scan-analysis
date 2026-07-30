# ADR 0010 — the accuracy tier is permanently empty, and says so

- Status: Accepted
- Date: 2026-07-30
- Amends: ADR 0005 (which reserved the two-tier scaffold for issue #12)
- Issue: [#12](https://github.com/cweber12/beta-scan-analysis/issues/12) (closed
  `wontfix`), [#133](https://github.com/cweber12/beta-scan-analysis/issues/133)

## Context

ADR 0005 retired manual absent flags as accuracy-tier evidence and left the accuracy
tier empty, with an explicit promise: *"Second-model verification (issue #12) is the
future accuracy source. The two-tier scaffold is retained so #12 can populate it
without re-plumbing."* `TruthFrame.verified` has returned a hardcoded `False` ever
since, with a comment pointing at #12.

That promise no longer holds, and the corpus is what retired it.

The whole corpus has now been human-reviewed. Every stretch flagged
`human-flagged-wrong` is a **correct ViTPose detection on the wrong person** — an
identity failure upstream of the pose head, not a joint-accuracy failure. That is
2,113 frames of 78,465 (2.7%), confined to 7 of 86 bundles, six of which the #15
conformance gate already quarantines.

Second-model auto-verification was designed to reuse the existing job's decoding and
**Climber tracking**. Both models would then receive the same box from the same
tracker, so on exactly the frames that fail, both pose the wrong person accurately,
agree, and the frame would be stamped auto-verified — promoting a known-bad frame into
the corpus's highest-trust class. Full reasoning in
`.out-of-scope/second-model-verification.md`.

So there is no forthcoming attestation source, and the tier's emptiness is permanent
rather than pending. The remaining question was whether to keep it.

## Decision

**The accuracy tier stays, permanently empty, and reports itself as not computable.**

- No `review` value is a positive human attestation, and joints are never
  hand-attested. `TruthFrame.verified` stays `False` — now by decision rather than by
  deferral, and its comment must stop pointing at #12.
- Where the accuracy tier would render, the record and report state **"not computable
  — no verified truth frames"** rather than emitting an empty band. Issue #133 carries
  this.
- **Agreement is never accuracy.** The agreement tier scores a Run against the
  unchallenged ViTPose scaffold; it measures distance from a seed, not from reality.

Deleting the tier was the serious alternative and is rejected below.

## Consequences

- A permanently-empty band is the standing structural reminder that pose accuracy is
  unmeasured in this harness. Its emptiness is now a documented statement rather than
  a symptom awaiting a fix.
- `.out-of-scope/second-model-verification.md` exists so the proposal is not
  re-litigated; it records the corpus evidence, not just the verdict.
- The 2,113 `human-flagged-wrong` frames remain excluded from scoring (ADR 0004/0005)
  but gain a second role: they are the only human-attested tracker-failure labels in
  the corpus, and therefore the validation set for any truth-side identity signal.
- Nothing about ADR 0003 changes. ViTPose remains a seed, the human remains the truth
  authority, and this ADR records that the authority has exercised review as
  *negative* labelling only.

## Alternatives considered

- **Build second-model verification (issue #12).** Rejected: it would spend a second
  dependency stack and a corpus-wide GPU re-run to auto-verify a 2.7% defect rate that
  complete human review already covers, while sharing the tracking stage that produces
  every one of those defects. Drawing independence properly (separate detector and
  tracker, sharing only decoded pixels) is defensible but roughly doubles seeding cost
  for the same 2.7%.

  **One caveat, stated so the rejection is not read wider than it is.** The review
  criterion was *"is this the right climber?"*, so it attests **identity**, not joint
  correctness. A truth defect leaving the skeleton on the correct person — a
  left/right laterality swap (#148 H2) — is invisible to it, and is the one failure
  class two pose heads on a *shared* box would genuinely disagree about. The rejection
  holds anyway: test-time flip augmentation on the existing model addresses laterality
  at a fraction of a second model's cost, and #148 sequences the measurement.
  Read this ADR as retiring second-model verification, not as a claim that truth is
  defect-free.
- **Delete the accuracy tier and report only agreement.** Rejected, though it was
  close. With one pose number in the report, `agreement` is read as accuracy — the
  ADR 0003 circularity re-entering through the report layer instead of the data layer.
  Issue #133 records that the empty tier has *already* been misread as a detector
  problem across several baselines; that argues for naming the emptiness, not for
  removing the thing being named. Renaming the survivor to `scaffold agreement` would
  move the caveat into the term, but relies on the name doing that work forever.
- **Keep emitting an empty band silently.** Rejected: it is the status quo, and it is
  the specific failure #133 was opened against — a reader cannot distinguish
  "not measured" from "measured badly".
