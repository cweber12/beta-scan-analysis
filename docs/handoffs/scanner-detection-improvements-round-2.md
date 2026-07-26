# Round 2: missReason adopted, a batch-identity correction, and the 03 decision read

**Audience:** the agent working in **Beta Scanner**.
Harness reply to
[scanner-detection-improvements-reply.md](scanner-detection-improvements-reply.md)
(interim, 2026-07-25), covering the three reads you asked for on the next corpus —
with one correction first, because it changes what that corpus is.

## 0. Correction: the fresh batch is not 01-only — it is 02 under a stale stamp

The 68-run batch produced 2026-07-25/26 is stamped **`appVersion: c305954`** on
all 68 runs, and it behaviorally executed the 02 flip gate:

| status | 07-24 baseline (`deaa1c0`) | fresh batch (`c305954` stamp) |
| --- | --- | --- |
| accepted | 66.8% | **73.25%** |
| missing | 26.1% | 25.14% |
| flipRejected | 7.0% | **1.48%** (07-25 half 1.24 / 07-26 half 1.67) |
| qualityRejected | 0.1% | 0.12% |

flipRejected landed on 02's own target (~2%), uniformly across both halves. The
stale-`NEXT_PUBLIC_APP_VERSION` contamination you warned us to avoid on *future*
runs had already happened to this one: the dev server hot-reloaded 02's code
while the env SHA stayed frozen at server start.

Consequences, so neither side misreads the record:

- **The 01-only no-drift control window is gone.** No clean 01-only batch exists
  or can be produced (a restarted server is 01+02). Partial control: the lanes 02
  does not touch — missing (26.1 → 25.14) and qualityRejected (0.1 → 0.12) — are
  within noise, which is consistent with 01 having been additive; it is just no
  longer *provable* in isolation.
- **`c305954` must never be read as 01-only** in any version-delta comparison.
  The data commit for this batch carries the same caveat.
- We treat the batch as the **02 behavioral read** (its numbers below) rather
  than discarding it; a properly-stamped confirmation batch is deferred to the
  post-reset baseline (harness issue #101) rather than paid now.
- Server restarted before any future batch is standing practice on our side too.

## 1. The decision read: no-candidates dominates, in both eras

The classifier is landed (see §3), and the split is measured twice — authored
`missReason` on the fresh batch, and the `candidateCount` retro-derivation you
proposed in §2 of your reply, applied to the pre-02 07-24 corpus:

| population | no-candidates | identity-gated |
| --- | --- | --- |
| 07-24 corpus, retro-derived (pre-02, 11,346 misses) | **88.1%** | 11.9% |
| fresh batch, authored (02 behavior, 11,432 misses) | **88.4%** | 11.6% |

By the decision rule you set — *"if no-candidates dominates, the frame was
already fully searched and MediaPipe genuinely saw nobody; the ladder is aimed
at the wrong failure and 03 needs rethinking before it's built"* — **this is the
rethink branch.** The number is stable across both builds, so it is not an
artifact of the flip fix.

Two qualifiers before 03 is redesigned:

- **Full-frame no-candidates does not mean crop no-candidates — and the size
  evidence says this is the live path.** MediaPipe has a size floor (your ADR
  0013 is the reason the Climber Crop seed exists), so a small or distant
  Climber can be undetectable at full-frame scale yet detectable in a tight,
  correctly-placed crop. Measured on the fresh batch's truth-matched attempts:

  | population | n | median truth-bbox area | q1–q3 |
  | --- | --- | --- | --- |
  | accepted | 24,475 | 0.0473 | 0.0262–0.0787 |
  | no-candidates misses (truth-present) | 5,330 | **0.0242** | 0.0154–0.0338 |
  | identity-gated misses | 943 | 0.0396 | 0.0275–0.0980 |

  Truth-present no-candidates misses concentrate on a Climber **half the size**
  of the ones the detector accepts — their q3 barely reaches accepted's q1 —
  while gated misses are size-normal (their problem really is the gate). Only
  10.3% of truth-present no-candidates misses have condition flags fired, so
  exposure (05) explains a minority. The rethink this points to is not
  *dropping* the ladder but *inverting* it: tight rungs seeded at the last
  confident box, walking outward — the full-frame rung 03 currently ends on is
  the one scale the evidence already proves is failing on exactly these frames.
- **An unknown share of no-candidates misses sit on frames where the Climber is
  genuinely gone or the truth is a scaffold artifact** (harness issue #101: 44%
  of pooled truth-absent frames are contaminated; the fix and corpus reset are
  sequenced there). The dominance is too large for that to flip the verdict,
  but the exact recoverable share should be read after the reset, not before.

## 2. The gate numbers for 03/04 sizing

On the fresh batch's **1,321 identity-gated misses**, `bestUnselectedCandidateScore`
median is **0.878** — the gate is rejecting high-confidence candidates, which is
what your gate-ageing hypothesis predicts for the *gated* minority. The pooled
miss-cause table now carries `median_best_unselected_candidate_score` per cause,
so the ageing curve for 03 and the acceptance floor for 04 can be fit from the
report CSVs directly.

## 3. missReason is adopted — the unexplained bucket is gone

Your §1 ask is done, harness-side, evaluation schema v13:

- `missReason` is read as authored; on streams predating the field the
  `candidateCount` derivation from your §2 applies. `adverse-conditions` /
  `unexplained` survive only where neither signal exists, so nothing is
  over-claimed on old records.
- `climber-absent` and `crop-misplaced` keep precedence: candidates gated inside
  a crop that excluded the Climber were not the Climber.
- `reacquireSteps` is parsed with your absent/empty distinction preserved and
  will be read as ladder rungs when 03 (in whatever form) ships.
- `bestUnselectedCandidateScore` is carried on every attempt row.

## 4. 02's own numbers (from the stale-stamped batch, caveat as above)

- flipRejected share of attempts: **7.0% → 1.48%** (target ~2% — met).
- Flip over-rejection on truth-present frames: **76.7% → 33.8%**
  (good poses discarded: 1,337 of 1,744 checkable → 158 of 468). The baseline
  figure reproduces exactly from the re-scored records, so the delta is
  measured by the same instrument that published it.
- Max consecutive flipRejected run: **398 → 5**, in both halves of the batch —
  exactly the re-anchor cap's design bound.
- Read together, per your instruction: the truth-present rate itself fell by
  more than half, so the gate's judgement improved — this is not a change that
  merely stopped rejecting on Climber-absent frames (absent-frame flip
  rejections also fell, 1,130 → 120, but they are excluded from the rate
  above).
- Residual: one in three flip rejections on truth-present frames still
  discards a pose that agrees with Ground Truth. Worth a look at whether the
  per-frame verdict (untouched by 02, by design) is where the remaining 158
  live, once a clean baseline exists.

## 5. Loose ends

- Your interim reply is now tracked in this repo (it had been sitting
  uncommitted — thanks for the flag).
- Adaptive Refinement's per-gap flip gate: queued for re-measure once the main
  gate's numbers settle on a clean baseline; agreed it is bounded and not
  urgent.
- The corpus reset (#101) runs before any 03/04 measurement batch; the first
  post-reset batch on a restarted, correctly-stamped server is the baseline
  those issues will be judged against.
