# Handoff: calibration freshness + annotation/calibration state in the scanner

**Audience:** an agent working in the **Beta Scanner** repo (the Next.js pose/ORB
app). You do not need the analysis harness repo open to do this work — this doc
tells you what the harness observed, the contract it enforces, and what the
scanner must change so its UI stops reporting stale calibrations/annotations as
healthy.

**Companion doc:** [scanner-data-contract.md](scanner-data-contract.md) describes
the bundle layout and what the scanner reads/writes. Read its "bundle layout"
section first if you're new to this.

---

## The problem, as observed on 2026-07-18

The user recalibrated **every** bundle and scanned at least one detection run per
bundle. The scanner UI then showed a calibration warning for exactly **one** run
(`planet-x/tf0hELD_M88_20260711-185634` — which is a special case: ViTPose has
never found the climber there, so it has no ground truth at all). Every other
bundle looked healthy: current calibration, accepted ground truth.

The harness disagrees. `python -m analysis_pipeline evaluate` pairs each
detection run with the bundle's ground truth **only when their setupHashes
match**, and it currently skips **22 run/truth pairs across 16 bundles** with
`setupHash mismatch` — the run was scanned under the current calibration, but the
ground truth still stamps a hash from an older calibration. The UI's freshness
signal and the actual hash chain have diverged: the scanner is showing
"accepted" annotations that are stale evidence.

Full list of skipped pairs: see harness issue
[#21](https://github.com/cweber12/beta-scan-analysis/issues/21). Affected route
folders: get-carter (4 bundles), lizard-therapy (1), mandala-the (3),
maze-of-death (2), midnight-lightning (2), planet-x (4).

### Two root causes (both scanner-side)

1. **setupHash churn.** Every calibration session mints a new `setupHash`, even
   when the calibration parameters barely (or don't) change. Each new hash
   silently invalidates all existing ground truth for that bundle. This is the
   treadmill: recalibrate → every truth in the bundle is stale → nothing warns.

2. **Ground-truth export races the ViTPose job.** The export seeds from whatever
   ViTPose scaffold is on disk. If the human exports before the
   current-calibration ViTPose job finishes, the truth stamps the *previous*
   calibration's hash. Third documented occurrence, timeline from
   `planet-x/Planet_X__V6____Joshua_Tree__CA_20260711-142049` on 2026-07-18:

   | time  | event                                             |
   |-------|---------------------------------------------------|
   | 07:53 | recalibration writes `setup.json` (new hash `25051d75…`) |
   | 07:55 | ground truth exported — stamps **stale** hash `11f537e9…` |
   | 07:59 | fresh ViTPose job finishes (`vitpose.json`, current hash) |
   | 08:01 | detection run scanned (current hash) — **cannot pair** with the 07:55 truth |

   Two earlier instances are documented in harness issue #7's comments.

### Why the UI misses it

Only the *truthless* bundle warned. Whatever the current warning checks
(missing artifacts, ViTPose failure), it is **not** comparing the hash chain:
`ground-truth.json`'s `setupHash` vs the current `setup.json` `setupHash` vs
each run's stamped hash. Stale-but-present truth passes silently.

---

## The contract the harness enforces (do not change it — match it)

Per ADR 0004 in the harness repo
(`docs/adr/0004-ground-truth-review-provenance.md`):

- `ground-truth.json` must carry the `setupHash` of the `setup.json` that
  produced its ViTPose scaffold, plus per-frame `review` provenance
  (`auto` / `human-flagged-wrong` / `human-flagged-absent`).
- Evaluate pairs a run with truth **iff** the run's stamped `setupHash` equals
  the truth's self-reported `setupHash` (legacy truths without one fall back to
  the bundle's current `setup.json`).
- The harness's ViTPose sidecar `vitpose.status.json` reports
  `running | done | error`, and `vitpose.json` stamps the `setupHash` it was
  generated under.

So: an "accepted" annotation is only *valid evidence* while its stamped hash
equals the calibration the runs are scanned under. The scanner UI must reflect
exactly that predicate.

---

## Work items

### 1. Content-derived setupHash (kill the churn)

Derive `setupHash` from the calibration *content* (crops, climber point,
panning, quality tier, condition labels — whatever participates in scaffold
generation), not from session identity or timestamps. Re-saving an unchanged
calibration must produce the **same** hash, leaving existing truth and runs
valid. Normalize before hashing (stable key order, rounded floats) so
serialization noise can't mint new hashes.

### 2. Gate ground-truth export on the ViTPose job

Block (or queue) the ground-truth export until:

- `vitpose.status.json` is `done`, **and**
- `vitpose.json`'s stamped `setupHash` equals the current `setup.json`'s.

If either fails, the export UI should say why ("ViTPose still running for this
calibration" / "scaffold is from an older calibration — re-run ViTPose") instead
of silently exporting a stale scaffold. Stamp the exported truth's `setupHash`
from the scaffold actually used, not from `setup.json` at export time — that
makes the race structurally impossible rather than merely unlikely.

### 3. Surface staleness in the UI (the missing warning)

For each bundle, compute and display three states from the hash chain:

- **Truth stale:** `ground-truth.json.setupHash ≠ setup.json.setupHash` →
  "annotations were accepted under an older calibration — re-run ViTPose and
  re-export". An accepted-annotation badge must never render as healthy in this
  state.
- **Run unpaired:** a detection run's stamped hash ≠ the truth's hash → the run
  produces no evaluation evidence; show it as such in run history.
- **Truthless:** no `ground-truth.json` (today's only warning) — keep it.

The first two are what's missing today across the 16 affected bundles.

### 4. Invalidate annotation state on recalibration

When a calibration save produces a **new** hash (real content change), the
bundle's existing ground truth must transition to a visible "stale" state — not
remain "accepted". The re-review flow then re-runs ViTPose and re-exports.

Open question for the implementer: whether `human-flagged-wrong` /
`human-flagged-absent` frames can be carried forward into the new truth (the
video frames are unchanged; only the scaffold is recalibrated) or whether the
human must re-review from scratch. Carrying them forward preserves expensive
human work; if you do, keep the per-frame `review` values intact and re-seed
only the `auto` frames.

---

## Verification (harness side, after implementation)

1. In the scanner, for each affected bundle: re-run ViTPose under the current
   calibration, wait for `done`, re-export ground truth.
2. In the harness repo: `python -m analysis_pipeline evaluate` — expect the
   `setupHash mismatch` skip count to drop from **22 to 0** (some pairs may
   legitimately skip for other reasons, e.g. truthless tf0hELD_M88).
3. Regression check for item 1: re-save an unchanged calibration, confirm
   `setup.json`'s hash is byte-identical and `evaluate` output is unchanged.
4. Regression check for item 2: trigger a recalibration and immediately attempt
   export — it must refuse until the fresh ViTPose job completes.

---

## Status

> **Instructions for the implementing agent:** update the table and log below as
> you land work. Keep one row per work item; set Status to
> `not-started | in-progress | blocked | done`, and note the scanner commit/PR.

| # | Work item                                   | Status      | Commit/PR | Notes |
|---|---------------------------------------------|-------------|-----------|-------|
| 1 | Content-derived setupHash                   | not-started |           |       |
| 2 | Export gated on ViTPose job state + hash    | not-started |           |       |
| 3 | UI staleness surfacing (truth/run/truthless)| not-started |           |       |
| 4 | Annotation invalidation on recalibration    | not-started |           |       |

### Log

- 2026-07-18 — handoff written (harness agent). 22/51 run-truth pairs skip on
  setupHash mismatch; UI warns only on the truthless bundle.

---

## Suggested skills

For the agent implementing this in the scanner repo:

- `/diagnose` — if the export race is hard to reproduce, use the disciplined
  reproduce → instrument loop on the export path.
- `/webapp-testing` — drive the Next.js UI to verify the new staleness badges
  and the export gate end-to-end.
- `/verify` before committing, and `/code-review` on the finished branch.

## References

- Harness issue [#21](https://github.com/cweber12/beta-scan-analysis/issues/21)
  — the 22 stale pairs and per-bundle fix list.
- Harness issue [#22](https://github.com/cweber12/beta-scan-analysis/issues/22)
  — adjacent: wrong `route_folder` in some mandala `metadata.json` (separate
  fix, don't bundle it here).
- Harness ADR 0004 — ground-truth review provenance + setupHash contract.
- [scanner-data-contract.md](scanner-data-contract.md) — bundle layout and
  read/write ownership.
