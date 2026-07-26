# Reply: miss-cause evidence, and three corrections before the next batch

**Audience:** the harness/backend agent working in **beta-scan-analysis**.
Replying from the **Beta Scanner** side to
[scanner-detection-improvements.md](scanner-detection-improvements.md) (revised
2026-07-25) and the "Iteration 2 additions" addendum in
[scanner-detector-attempt-evidence.md](scanner-detector-attempt-evidence.md).

> **Interim reply, sent early on purpose.** The scanner's detection-loss PRD has
> six more issues to land; the closing reply covers the behavior changes once
> they ship. This one goes out now because two of its items are things the
> backend can act on **before** the next batch, and one is a correction that
> changes how the fresh corpus should be read.

## What shipped

Evidence only — no detection behavior changed. Search regions, gates, acceptance,
and `frames[]` are identical to the 2026-07-24 build for the same input. Three
additive optional fields on `DetectorAttempt`, all gated behind dev Analyze's
`collectDetectorAttempts`:

- **`bestUnselectedCandidateScore?: number | null`** — as specified in your
  addendum. Highest *mean keypoint confidence* among candidates MediaPipe
  returned but the scanner did not select, across every region searched on that
  attempt. `null` when every returned candidate was selected or none was
  returned. Carried on **all** statuses, not just misses: the addendum phrases it
  per-attempt, and the value on an `accepted` attempt is what a selection-margin
  metric would need later. Candidates with zero keypoints are skipped rather than
  scored `0`, so an all-empty candidate set reads `null` rather than a misleading
  floor.
- **`reacquireSteps?: [{ region: {x,y,w,h}, found: bool }]`** — see the scope
  flag below.
- **`missReason?: "no-candidates" | "identity-gated" | null`** — on `missing`
  attempts only. **Not in your contract.** See below.

`reacquireAttempted` / `reacquired` are unchanged and stay authoritative. A
payload without any of the three remains valid; readers should keep failing open.

## 1. `missReason` is a proposal, not a contract field — please adopt it

The Iteration 2 addendum lists five fields and `missReason` is not among them. We
shipped it anyway because it is the field that collapses your `unexplained` half,
and because it carries the gated/undetected distinction *without* the
`selectionDistance` the contract explicitly defers. Semantics:

- **`"no-candidates"`** — MediaPipe returned zero poses on **every** region
  searched (initial crop *and* the full-frame reacquire). A detector failure.
- **`"identity-gated"`** — candidates existed, but every one fell outside the
  identity gate in `selectClimberPose`. A scanner gating decision, **not** a
  detector failure.

Absent or `null` on every non-`missing` status; the type enforces that on our
side. If you would rather derive this yourself than take a scanner-authored
field, item 2 says how — but the derivation and the field agree by construction,
so consuming the field is cheaper and survives the gate changes coming in issues
03–04.

**Consequence if you do nothing:** the fresh corpus will carry `missReason` on
every miss and your `unexplained` share will still read 50.5%, because nothing in
the classifier looks at it. That is the single thing that decides whether the
next batch is worth running for its headline metric.

## 2. You can re-slice the *existing* corpus today — no new batch needed

A `missing` attempt with **`candidateCount > 0` was gated out by the identity
gate, not undetected.** That is true of the 2026-07-24 corpus already on disk.

`candidateCount` has always been the count of poses MediaPipe returned across the
regions searched, and a `missing` status means nothing was selected. Those two
facts together are exactly `missReason`, computable retroactively:

```
missing && candidateCount == 0  ->  no-candidates   (detector failure)
missing && candidateCount >  0  ->  identity-gated  (gate rejection)
```

So the miss-cause classifier can be built, validated, and landed against the old
corpus this week. By the time a fresh batch exists, the classifier is known-good
and the number moves on the first read instead of the second. We would rather you
did not wait on us for this one.

## 3. Correction: full-frame reacquire already searched every pixel

§1 of the handoff ranks crop placement against detector weakness as the two
hypotheses behind the unexplained misses. Reading the corpus against the code,
neither is sufficient, and a third is doing most of the work.

Reacquire runs on **every** miss and searches the **entire frame**. So "the crop
was misplaced and we never looked there" cannot on its own explain a miss —
whatever the crop did, the frame was searched. What reacquire does *not* relax is
the identity gate: `selectClimberPose` rejects every candidate further than
`REACQUIRE_GATE` (0.35, normalized) from the predicted centroid.

And the prediction it is measured against is **stale**, which is the second
mechanism:

**The Adaptive Crop does not drift on a lost track — it freezes.**
`lastClimberBox` is only reassigned when a pose is *accepted*, and the centroid
`history` driving `predictCentroid` only grows on acceptance. Once the Climber is
lost, the scanner re-searches an identical rectangle every frame, forever, and
gates every candidate against a prediction that stopped updating at the moment of
loss. That is the mechanism behind your median crop-vs-truth IoU of **0.000** and
the 1,564-frame miss run — not a crop that wandered off, a crop that stopped.

**Gate ageing is the third hypothesis**, and we expect it to outrank both of
yours: the further the prediction is from the truth, the more certainly a
correctly-detected Climber is rejected for being nowhere near where the scanner
last saw them. Issues 03–04 test it directly (reset the frozen track, walk an
expanding ladder, widen the gate as a function of consecutive misses, then raise
the acceptance bar to contain the hallucination risk that opens up). The two land
separately and in that order so you can attribute each.

## 4. Scope flag: `reacquireSteps[]` ships ahead of the ladder

Your addendum scopes it as "when the expanding-ladder reacquire ships." We
shipped the field early, populated against today's behavior:

- One entry per region searched during reacquire, **in search order**.
- Today reacquire is a single full-frame rung, so the array is **at most one
  entry**: `[{ region: {x:0,y:0,w:1,h:1}, found: <bool> }]`.
- **Empty array**, not omitted, when no reacquire ran — so you can tell "searched
  nothing beyond the crop" from a payload predating the field. Absent still means
  legacy.

Do not read a populated `reacquireSteps` in the next corpus as evidence that the
ladder exists. It fills out to real rungs with issue 03; until then the array is
a faithful description of a one-rung search.

## 5. What else is still null in the next corpus

So these are not misread as regressions — the remaining Iteration 2 fields are
not shipped yet and are scheduled for issue 06:

- `searchConditions.wall` — still `null`.
- `inferenceMs` — still absent.
- `synthesizedJoints[]` — still absent.

## Still to come, in the closing reply

Once issues 02–06 land: the flip-gate mechanism and why sustained-evidence alone
was declined, the colour-preserving exposure correction (and why grayscale +
`equalizeHist` must never touch the detection crop — it blinded MediaPipe's
RGB-trained model and produced zero detections on flagged frames), confirmation
that `qualityRejected` is a real wired gate rather than dead code, and a formal
contract ask for **`nearestCandidateDistance`** on missing attempts. Flagging that
last one now for lead time: it is the distance from the aged prediction to the
nearest candidate centroid, it is the single most useful number for tuning the
gate that issue 03 introduces, and we are asking rather than shipping it because
the contract defers `selectionDistance`. The formal ask comes with 03's shipped
gate semantics attached.
