# Handoff: manual-absent flag deprecated — presence is ViTPose auto (ADR 0005)

**Audience:** an agent working in the **Beta Scanner** repo (the Next.js pose/ORB
app). You do not need the analysis harness repo open — this doc is a *delta* on the
Ground Truth review contract you implemented for harness issue #5.

**Companion docs:**
[scanner-data-contract.md](scanner-data-contract.md) (Phase 3 is the full,
now-amended review contract) and
[scanner-calibration-freshness.md](scanner-calibration-freshness.md) (the
`setupHash` staleness rules referenced below).

**Harness refs:** ADR 0005 (`docs/adr/0005-manual-absent-flags-deprecated.md`) and
issue [#11](https://github.com/cweber12/beta-scan-analysis/issues/11). This
supersedes the `human-flagged-absent` semantics in ADR 0004 / the original #5
handoff.

---

## What changed and why

The original review contract (ADR 0004 / issue #5) gave `human-flagged-absent` the
*strongest* presence weight in evaluation: the manual flag overrode the frame
`state` and fed the harness's **accuracy tier** as human-attested presence truth.

Three things made that wrong:

1. The scanner harness **removed the manual "absent" button** — no new
   `human-flagged-absent` frames are produced.
2. **ViTPose auto-absence is reliable**: it flags every zero-landmark frame absent
   and never marks a seeded frame absent. Auto-detection, not a human, is the
   trustworthy presence-negative signal.
3. The existing manual absent flags came from an **older, inconsistent** workflow
   and **may be stale** — a re-seed can now detect landmarks on a frame that was
   hand-flagged absent, leaving the flag contradicting the seed.

So the harness (ADR 0005) now:

- takes **presence from `state`**, never from a review flag;
- **excludes `human-flagged-absent` frames from all scoring**, exactly like
  `human-flagged-wrong` (they surface only in a record's `counts.agreementSkipped`);
- keeps the **accuracy tier empty** until a second pose model lands (#12).

`auto` absences (`review: "auto"`, `state: "absent"`) are still scored — a scanner
hallucination on a truly-absent frame is still caught as a presence false positive.

---

## What the scanner must do

### 1. Stop emitting `human-flagged-absent` (required)
The absent button is gone; make sure no code path still writes
`review: "human-flagged-absent"` on new saves. Express absence as an **`auto`**
frame with `state: "absent"`:

```json
{ "frameIndex": 12, "timestamp": 12, "state": "absent",
  "joints": {}, "verified": true, "review": "auto" }
```

### 2. Set `state` from the seed (required)
`state` is now the presence authority. When the `vitpose.json` frame you seeded
from has `keypoints: []`, write `state: "absent"`; otherwise `state: "present"`.
Do not derive presence from any human action.

### 3. Keep `human-flagged-wrong` as-is (unchanged)
The "wrong" flag is untouched by ADR 0005: keep letting the human mark a seed
skeleton wrong, keep the frame in the file, keep writing
`review: "human-flagged-wrong"`. The harness still excludes it from scoring and
counts it in skip accounting.

### 4. Migrate legacy absent flags (optional, low priority)
Existing `ground-truth.json` files still carry `human-flagged-absent` frames (37
across the current corpus). The harness now **ignores the flag and excludes those
frames from scoring**, so they do no harm — but they also contribute nothing. If
you want those genuinely-absent frames to count again as `auto` presence-negative
evidence, rewrite them **in place** on the next save of a bundle:

- if `state: "absent"` (all current ones qualify): set `review: "auto"`, keep
  `state`, keep `joints: {}`;
- if a re-seed has since put landmarks on the frame: set `review: "auto"` and let
  `state` reflect the seed (`present`).

Recompute `groundTruthHash` after any such rewrite (it produces a fresh evaluation
record, which is correct). If you skip this, nothing breaks — it is purely
reclaiming absent evidence.

---

## Don't break (unchanged from the #5 contract)

- **`setupHash`** copied into `ground-truth.json` from the `vitpose.json` you
  seeded from, and **`setupHash` inside each pose run's `data`** — the harness
  refuses to pair a run against truth from a different setup.
- **`appVersion`** in pose `diagnostics` (real git sha) — drives per-version
  regression trends.
- **Per-keypoint `score` thinning** — a missing joint is measured as a coverage
  failure, not hidden.
- **`groundTruthHash`** recomputed on every save.

---

## Acceptance

- A fresh Ground Truth save emits **no** `human-flagged-absent` frames; absences
  are `review: "auto"` + `state: "absent"`.
- `state` is set from the seed's landmark presence on every frame.
- `human-flagged-wrong` still round-trips unchanged.
- (Optional) legacy `human-flagged-absent` frames rewritten to `auto` on next save.
