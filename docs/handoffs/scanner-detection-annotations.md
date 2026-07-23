# Handoff: detectionAnnotations for scanner-coupled quality labels

**Audience:** an agent working in the **Beta Scanner** repo (the Next.js pose/ORB
app). The analysis harness reads this contract and uses it to refine per-frame
quality summaries; write the block into `ground-truth.json` alongside `review`.

**Companion docs:**
[scanner-data-contract.md](scanner-data-contract.md) for the bundle layout and
[scanner-review-provenance-adr0005.md](scanner-review-provenance-adr0005.md) for the
`review` semantics this block sits beside.

Harness issue: [#45](https://github.com/cweber12/beta-scan-analysis/issues/45).

---

## What this block is

`detectionAnnotations` is a list of frame-index ranges in `ground-truth.json`.
Each range refines the harness's automatic per-frame quality class for the frames it
covers. It does **not** replace `review`; it sits alongside it.

Schema:

```json
{
  "setupHash": "<hash of the setup.json this truth was authored against>",
  "detectionAnnotations": [
    {
      "startFrame": 120,
      "endFrame": 132,
      "failureClass": "wrong-subject",
      "distractor": "tree_bush"
    }
  ]
}
```

Rules:

- Frame indices are inclusive and match the `frameIndex` values in `frames`.
- `setupHash` is required on each annotation range. The harness ignores stale ranges
  whose `setupHash` does not match the bundle's active setup.
- `failureClass` uses the harness's frame-quality taxonomy:
  `ok | wrong-subject | hallucination-fp | flipped-rotated | distorted`.
- `distractor` uses:
  `tree_bush | rock_wall_shape | crash_pad_bag | animal | shadow | spectator |
  hallucination_none | gear | other`.
- If no annotation matches a frame, the harness falls back to the automatic class
  from issue #44.
- If multiple valid ranges overlap, keep the later/manual refinement authoritative.

---

## How the harness uses it

The harness resolves the active annotation per truth frame and writes both the
automatic and effective classes into the pooled frame-quality rows. That means:

- the failure-class frequency table prefers the human label when one is present;
- the distractor-frequency table reflects confirmed human distractors;
- condition correlations still use the effective class, so confirmed labels beat
  inference without changing the rest of the issue #44 pipeline.

Legacy bundles without `detectionAnnotations` keep working: the harness treats them as
auto-only truth.

---

## Verification

1. Save a bundle with one annotated frame range and one stale `setupHash` range.
2. Re-run the harness and confirm the active range overrides the auto class while the
   stale range is ignored.
3. Confirm the report shows the failure-class and distractor-frequency sections with
   the confirmed labels surfaced.

---

## Status

| # | Work item | Status | Commit/PR | Notes |
|---|-----------|--------|-----------|-------|
| 0 | Add `detectionAnnotations` to Ground Truth | done | | harness issue #45 |
| 1 | Verify active setupHash only | done | | stale ranges ignored |
| 2 | Surface human labels in frame-quality aggregation | done | | failure class + distractor tables |
