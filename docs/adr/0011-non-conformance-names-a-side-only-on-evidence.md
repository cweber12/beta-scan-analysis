# ADR 0011 — non-conformance states a side only where there is evidence, and v15 opens a new baseline cycle

- Status: Accepted
- Date: 2026-07-30
- Advances: ADR 0009 (`BASELINE_CYCLE_SCHEMA` v14 → v15, at a cycle boundary)
- Issue: [#147](https://github.com/cweber12/beta-scan-analysis/issues/147)

## Context

The #15 gate fits `scanner = a·truth + b` per axis. A poor fit proves the scanner's poses
and the truth's disagree. **It cannot say which of them is wrong** — the fit is symmetric
in its two inputs.

The cause vocabulary asserted a side anyway. `suspected-mistrack` named the truth, and the
report told the reader so explicitly: *"This is the truth-repair worklist (#21/#34):
re-seed these bundles' Ground Truth."* The truth-repair worklist was filtered on that
cause alone.

Once the whole corpus had been human-reviewed, the claim became measurable. Every
`human-flagged-wrong` stretch is a human attestation that the truth put a correct pose on
the wrong person — the only truth-side evidence the harness holds:

| non-conformance cause | truth-side (human-flagged) | truth attested clean |
| --- | ---: | ---: |
| `suspected-mistrack` | 11 records / 5 bundles | **112 records / 38 bundles** |
| `sparse-match` | 16 / 5 | 42 / 15 |

**91% of its firings landed on bundles whose truth is attested free of identity error.**
Reproduce with `python -m scripts.measure_conformance_attribution`.

That was not a theoretical cost. #34 was built from this worklist, named 12 bundles as
mis-tracked, and exactly one carries a human wrong-flag. Six of the seven bundles that
*do* have wrong-person truth were never on it — including the worst in the corpus, at
1,448 wrong frames of 2,222. A re-seed of the original list was attempted and mostly
"failed", which was itself misread: eleven of the twelve targets had no truth defect to
repair.

## Decision

**1. The cause is renamed to `trajectory-divergence` and stops naming a side.** Its report
blurb now says the fit proves disagreement and directs the reader to the attribution.

**2. A new `conformance.attribution` names the side only on positive evidence**, with two
values, deliberately not three:

- `truth-identity` — the run's truth population contains human-attested wrong-person
  frames. Positive evidence.
- `unattributed` — everything else.

**There is no `scanner-side` value.** Absence of a truth flag is not evidence the scanner
failed; emitting one would rebuild this exact defect pointed the other way. The honest
majority of non-conforming records cannot be attributed, and the record says so.

A **laterality** defect — right person, left/right joints exchanged — is invisible to the
review that produces the flags, because a consistent swap renders as the same skeleton. It
therefore lands in `unattributed` rather than being silently counted against the scanner.
If [#148](https://github.com/cweber12/beta-scan-analysis/issues/148)'s H2 measures it, it
earns its own value then, on evidence.

**3. The truth-repair worklist keys off the attribution, never the cause.** A divergent
record nobody can attribute is an open question, not a re-seed candidate. It stays visible
in the quarantine table, where it reads as one.

**4. `BASELINE_CYCLE_SCHEMA` advances v14 → v15 at a cycle boundary.** The v14 cycle's
analyse phase completed and produced #147, #148, #149 and #150; the maintainer confirmed
nothing was mid-analysis. The whole population is re-scored under `--mode all`, which is
what makes this a boundary rather than the mid-cycle bump ADR 0009 flags.

## Consequences

- The truth-repair worklist shrinks to bundles with attested truth defects. #34's list is
  superseded; it has been re-scoped separately.
- `unattributed` is the majority verdict on non-conforming records, and that is the
  intended outcome: it records the absence of evidence instead of manufacturing a side.
  A reader who wants more must supply evidence — #150's per-frame identity confidence is
  the next source, and #148 H2 the one after.
- A pre-v15 record reads its old cause spelling through a compatibility mapping and
  reports `unattributed`, so an unre-scored record is never silently upgraded to a claim
  it does not support.
- Attribution is emitted on conforming records too. A conforming bundle whose truth
  carries flagged stretches is a real thing — the flags are excluded from the fit — and
  hiding it would put the reader back to inferring.

## Alternatives considered

- **Add `scanner-side` as a third attribution.** Rejected, and this was the tempting one:
  the issue that opened this work proposed it. It treats absence of truth-side evidence as
  presence of scanner-side evidence, which is the same inversion that produced #34.
- **Keep the cause name and add attribution only.** Rejected: the name is what is read,
  quoted, and acted on. #148's PRD restated "#34 holds 12 bundles whose truth is
  mis-tracked" as established fact the same week the measurement refuted it. Leaving the
  name would leave the misreading's source in place while adding a field nobody consults.
- **Report-only, deriving attribution at render time with no schema change.** A genuine
  option, and the right one had the cycle still been open — it delivers the finding with
  zero basis disruption. Rejected because the cycle *had* closed, making the bump cheap
  (`--mode all` reads JSON; it does not decode video), and because a render-time
  derivation leaves every consumer of the record — worklists, CSVs, future agents — still
  reading a cause that asserts a side.
- **Defer to the next collection.** Rejected: the next collection would be scored under a
  vocabulary already known to be wrong, and comparability would then argue for keeping it.
