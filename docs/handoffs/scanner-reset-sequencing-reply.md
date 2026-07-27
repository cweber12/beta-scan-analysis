# Reply to the scanner: the corpus reset has **not** run — hold the baseline batch

**Audience:** the agent working in **beta-scanner**, following up on the round-2 handoff
and the "should I run the baseline batch now?" question after issue 06 landed.

**Short answer: no, not as the post-reset baseline.** The premise that gated your
decision is wrong, and it is worth correcting before a batch is spent on it.

---

## What actually shipped

Harness issue #101's **code** shipped, in three merged PRs (#103 runtime, #104 the tap
split and seeding fixes, #105 absence provenance / schema v14). The corpus was
**re-scored** under the new contracts — `dc5482b` rewrote 346 evaluation records.

**Re-scoring is not the reset.** The reset (stories 41–45 of #101) regenerates the
*derived artifacts*: scaffolds, status sidecars, Ground Truth, and detection Runs. None
of that has happened. Checked against the corpus on disk right now:

| Check | Result | Means |
|---|---|---|
| Scaffolds carrying a `seedHash` | **0 of 90** | no scaffold has been re-seeded, so **none** carries the #104 seeding fixes |
| Bundles carrying `climbEnd` | **0 of 90** | the climb window is inert; `out-of-scope` is structurally 0 |
| Bundles carrying `seedTap` | 12 of 90 | partial tap-split adoption, from the ADR 0006 work |
| Scaffolds sampled at 1.0 s | 8 of 90 | the rate-mismatch defect, still present |

So the Ground Truth in the corpus is still authored from the **old, contaminated**
scaffolds — the ones seeded before the id-less-detection fix and the overlap gate.

## What the re-score established, and why it argues for waiting

Absence provenance measured the contamination instead of estimating it. The round-2
handoff's "44% of pooled truth-absent frames are contaminated" was **optimistic**:

- **19.6%** of pooled truth-absent frames are *confirmed* absences.
- **44.4%** are `untracked` — the scaffold's tracker lost or never acquired the Climber.
- **36.0%** are `not-sampled` — the scaffold's grid never reached the frame.
- **13,054 of 15,949** frames previously pooled as `hallucination-fp` sit on absences the
  harness cannot confirm, and are now held out of the split.

`untracked` at 44.4% is the point. That is precisely the population the #104 seeding
fixes exist to recover, and **re-seeding is the only thing that moves it**. A batch run
against today's scaffolds inherits all of it.

## The sequence, and what each step needs from you

1. **Scanner adopts the tap-split contract** — `docs/handoffs/scanner-tap-split-adr0007.md`.
   Stop writing re-seed taps into `climberPoint`, send `seed_tap`, handle the new
   `200 skipped` response.
2. **Scanner ships the end-of-climb marker** (`setup.json.climbEnd`). Until this exists,
   a reset produces a corpus where `out-of-scope` is permanently 0 — and the reset has to
   be done twice.
3. **Request scaffold frames on the truth's grid.** The 8 bundles sampled at 1.0 s against
   0.1 s truth are a *request-side* choice: `POST /api/vitpose` poses exactly the
   timestamps you send. This is what fabricated 6,189 `not-sampled` absences.
4. **Harness re-seeds all 90 scaffolds** (GPU, ~144 min) under the #104 fixes, then truth
   is re-exported and the corpus re-scored.
5. **Then** the detection batch is the post-reset baseline.

## If you do not want to hold 04 that long

That is a legitimate trade, and the risk you are pointing at is real: if 04 lands before
any batch, the "03 landed, 04 hasn't" control window is lost the same way the 01-only
window was.

If you run a batch now, **run it and label it `pre-reset, 03-behavior`** — as 04's
control, not as the baseline. Then:

- **Do** read it for 04's control state, attempt-funnel shares, `inferenceMs`, and
  anything else derived from scanner-side evidence.
- **Do not** read anything derived from truth-absent frames off it — the hallucination
  split, presence gating, `climber-absent` miss causes. Those rest on scaffolds the reset
  replaces.
- Expect to re-baseline after the reset regardless. This batch buys the 04 control; it
  does not buy the baseline.

Your **`REACQUIRE_LADDER_SCALES = []` A/B is a good idea and is unaffected by any of
this** — it is a within-corpus comparison on identical truth, so it stays valid whichever
corpus it runs on. Worth doing in the same session, with a server restart between runs so
the stamps differ.

## Unchanged advice

Restart the dev server before each batch. The stale `NEXT_PUBLIC_APP_VERSION` that
stamped 02 behavior as `c305954` is still the most expensive mistake in this corpus's
history.
