# ADR 0008 — absence provenance is derived in the harness, not authored into Ground Truth

- Status: Accepted
- Date: 2026-07-26
- Upholds: ADR 0004 (review provenance) and the issue #23 camera-angle decision —
  per-frame metadata the harness can derive does not belong in the truth artifact
- Issue: [#101](https://github.com/cweber12/beta-scan-analysis/issues/101)

## Context

A truth frame was either `present` or `absent`. That single label was flattening four
different situations, and the difference between them is the difference between four
different fixes:

- the Climber had not started climbing yet, or had already topped out;
- the ViTPose scaffold never sampled the frame, because it ran at 1 Hz while truth was
  exported onto a 0.1 s grid;
- the scaffold's tracker lost the Climber, or never acquired them at all;
- the Climber really is not in the frame.

Only the last one is evidence about the scanner. The corpus audit behind issue #101
found **44% of every pooled truth-absent frame coming from just 5 videos**, where
"absent" meant one of the first three — 2,665 frames from the sampling-rate mismatch
alone. That population is the entire evidence base under the headline defect
`hallucination on truth-absent frames 46.5%`, whose stated fix is presence gating. If
half those absences are a scaffold artifact or a tracking loss, the recommendation does
not follow from the evidence.

So the reason has to be recorded. The question is *where*.

## Decision

**The harness derives the absence reason from evidence already on disk. Ground Truth
stays pure keypoints.**

Every absent truth frame gets one of five reasons, resolved most-decisive first:

| Reason | Derived from |
|---|---|
| `out-of-scope` | the climb window on the Bundle's calibration (ADR 0007) |
| `not-sampled` | the scaffold's timestamp step compared against the truth step |
| `untracked` | the scaffold's seed-found flag and the structure of its tracking gaps |
| `confirmed-absent` | the residual once the others are excluded |
| `unknown` | nothing on disk to derive from |

The ordering is the argument. Out-of-scope is first because a post-topout frame is not
evidence at all, whatever the tracker did. `not-sampled` is next because a frame the
scaffold never looked at cannot tell us anything about tracking. Only then can a
tracking loss be claimed, and only what survives all three is an absence the harness is
willing to call **confirmed**.

**Only `confirmed-absent` enters the presence 2×2 and the hallucination split.**
Everything else is counted and reported (`unconfirmedAbsent`), never dropped and never
silently promoted. The schema bump is v14; a frame written before it reads `unknown`,
in the established fail-open tradition — the same shape as pre-v12 frames reading as
presence-*unknown* rather than absent.

**Why not author it into Ground Truth.** The truth artifact is the human's, and it is
the one thing in the Bundle nobody can regenerate. Three reasons decided this:

1. **Derivable facts go stale in a way authored ones cannot be corrected.** Every input
   to the reason — the climb window, the scaffold's grid, its tracking gaps — lives in
   artifacts the reset regenerates. Deriving means the reason improves automatically
   when the inputs do; authoring means a stale reason outlives the evidence that
   produced it, with no way to tell.
2. **Precedent.** ADR 0004 put review provenance *on the frame* because a human
   asserted it; the issue #23 camera-angle estimate went to `video-stats.json` rather
   than the scaffold precisely because the harness computed it. Absence reason is the
   second kind, not the first.
3. **It would make Ground Truth a harness output.** The scanner writes truth; the
   harness reads it. Writing derived metadata back into it would invert that and make
   the truth hash — the anchor the whole pairing model rests on — change whenever the
   harness's derivation changed.

## Consequences

- The hallucination split now means what it claims. Its denominator is the frames the
  harness can actually claim are Climber-free, and the held-out remainder is reported
  beside it with its reasons.
- **The published baseline is superseded, not regressed.** The truth-absent numbers move
  materially, and that is the point of the work: they were measuring a contaminated
  population. Anything quoting `hallucination-on-absent 46.5% → presence gating` is
  quoting a number computed over frames that did not mean what the column header said.
- A Bundle with no scaffold on disk contributes no confirmed absences at all. That is
  the honest direction — it under-claims rather than over-claims — and the corpus reset
  regenerates every scaffold, so it is a transitional state, not a permanent hole.
- The reason is recomputable at any time by re-running `evaluate`; nothing about it is
  baked into an artifact a human owns.
- `not-sampled` also surfaces as its own **non-conformance cause** (`rate-mismatch`),
  because a Bundle whose scaffold under-sampled the truth grid routes to *regenerating
  the scaffold* — not to the truth-repair worklist and not to the detector worklist.

## Alternatives considered

- **Author the reason into `ground-truth.json` per frame** — rejected for the three
  reasons above; the decisive one is that it makes the truth hash a function of harness
  logic.
- **Keep one `absent` label and filter the known-bad Bundles out of pooling** — rejected:
  it treats a per-frame property as a per-Bundle one, discards good frames from the
  affected Bundles, and leaves the same conflation in place for every Bundle that has
  not yet been audited.
- **Infer "the Climber left" from trajectory geometry** (Climber near a frame edge before
  the gap) — rejected as a heuristic dressed as evidence. The four reasons above are each
  read off a recorded fact; a geometric guess would be the fifth thing a reader could not
  check.
- **Treat every unexplained absence as confirmed** (fail-*closed*) — rejected: that is
  exactly the assumption that produced the contaminated baseline.
