# Handoff: Ground Truth should stamp the scaffold `seedHash` it was authored from

**Audience:** an agent working in **beta-scanner**. You do not need the analysis harness
repo open — this is a *delta* on what you write into `ground-truth.json` and on how you
judge truth staleness.

**Companion docs:**
[scanner-tap-split-adr0007.md](scanner-tap-split-adr0007.md) (where `seedHash` comes
from) and [scanner-calibration-freshness.md](scanner-calibration-freshness.md) (the
`setupHash` staleness rules this extends).

**Harness refs:** issue
[#119](https://github.com/cweber12/beta-scan-analysis/issues/119), the detector that
motivated it (PR #118), and ADR 0007.

---

## The defect, concretely

Ground Truth is authored *from* the ViTPose scaffold, but records nothing about **which**
scaffold. Regenerate a scaffold — a re-seed, a detector-resolution change — and the truth
on disk keeps describing the superseded one. Nothing on either side can tell.

Every frame the new scaffold poses that the old truth calls absent becomes a **phantom
absence**: it lands in the truth-absent population, is classified as a scanner
hallucination, and inflates the very metric harness issue #101 exists to make
trustworthy.

Measured after the #101 corpus reset — **11 bundles adrift**:

| Bundle | Truth present | Scaffold poses |
|---|---|---|
| `fKjfXtqLA1I` | 190 | 1811 |
| `w420jGWP2W0` | **0** | 1235 |
| `The_Mandala` | 68 | 600 |
| `VxhW7T4vg7E` | **0** | 463 |

`w420jGWP2W0` and `VxhW7T4vg7E` were accepted the same day their scaffolds were
regenerated, and record **zero** present frames against fully-posed scaffolds. Both show
as accepted and healthy in your dev corpus UI.

## Why neither side catches it

Your two signals, from `app/api/dev/shared.ts`:

- `hasGroundTruth` — file existence. The comment is explicit: *"`ground-truth.json` is
  only ever written by Accept & save, so existence is acceptance."*
- `truthStale` — true when the truth stamps an **older `setupHash`** than `setup.json`.

`setupHash` tracks *calibration*. **Re-seeding does not change the calibration**, so a
truth authored from a two-week-old scaffold still matches, and both signals read healthy.

This is structurally the same blind spot ADR 0007 closed for scaffolds. There, `setupHash`
matched whether or not a re-seed had moved the tap, so a stale scaffold was undetectable;
`seedHash` fixed it. The identical hole sits one layer up, between scaffold and truth.

---

## What to change

### 1. Stamp the scaffold's `seedHash` into Ground Truth

On **Accept & save**, read `seedHash` from the bundle's `vitpose.json` and write it into
`ground-truth.json`:

```jsonc
{
  "version": 1,
  "setupHash": "...",            // unchanged — the calibration anchor
  "scaffoldSeedHash": "3c6b5831a1b2c3d4",   // NEW — from vitpose.json.seedHash
  "groundTruthHash": "...",
  "frames": [ ... ]
}
```

Field name is the harness's suggestion, not a requirement — if you prefer `seedHash`,
say so and the harness will read that instead. What matters is that it is the scaffold's
hash, copied verbatim, and that it is written at accept time rather than derived later.

A scaffold with no `seedHash` (one written before ADR 0007) means there is nothing to
stamp — omit the field rather than inventing one.

### 2. Extend `truthStale`

It currently answers "was this truth authored under an older *calibration*". It should
also answer "was it authored from an older *scaffold*":

```
truthStale = truthSetupHash !== setupHash
          || (truthScaffoldSeedHash != null
              && scaffoldSeedHash != null
              && truthScaffoldSeedHash !== scaffoldSeedHash)
```

Note the null guards. Truth written before this change carries no stamp and must degrade
to today's behaviour — **not** to "stale". Fail-open is the established tradition on both
sides of this contract; a missing stamp is *unknown* provenance, never a failure.

An accepted badge must not read as healthy when the scaffold has moved, exactly as it
must not when the calibration has.

### 3. Nothing else changes

Nothing is auto-accepted, and this does not change that — `ReseedSweeper` states the
design and it stands. This makes staleness **visible**; it does not remove the human from
acceptance.

---

## What the harness does with it

`scaffold_truth_drift` (PR #118) already ships a heuristic: it compares the truth's
present-frame count against the scaffold's posed-frame count and flags a bundle when the
shortfall is ≥20 frames *and* truth holds under half the posed count. Loose on purpose,
so ordinary human editing never trips it — but it is an inference, and it will both miss
cases and occasionally annoy.

Once truth carries the stamp, the harness prefers an exact hash comparison and keeps the
heuristic only as the fallback for unstamped truth. The stale-truth worklist and its
report section already exist and need no change beyond the sharper signal.

The harness already records the scaffold's `seedHash` in that block, so a re-export can
be verified against it the moment you start writing it.

---

## Not in scope

- **Re-accepting the 11 bundles currently adrift.** That is corpus work under #101.
- **Auto-accepting truth when a scaffold changes.** Deliberately not proposed.
