# Corpus exclusions

Videos deliberately removed from `analysis/`, and why. **Read this before re-adding a
video** — every entry here was downloaded, seeded and reviewed at least once before it
was judged unusable, so re-adding one costs the same work again and reintroduces the
same contamination.

Video binaries are gitignored, so a removed bundle leaves no trace on disk. The YouTube
id is the part of the bundle name before the timestamp; that is all it takes to
re-download one by accident.

## Why exclusion is not the same as quarantine

The pipeline's quarantine gate (`trends._quarantined_rows`, issue #15) operates on
**evaluation records** — it withholds a scored pairing from the pooled derivations. It
has nothing to say about a Bundle that produces no records at all. There is deliberately
no bundle-level quarantine flag: a Bundle that cannot be scored is not a Bundle whose
scores need discounting, it is one that should not be in the corpus.

The consequence worth stating plainly: **leaving an unusable bundle in place, truthless,
is the worst of the three options.** A Bundle with no `ground-truth.json` scores against
its ViTPose scaffold directly, so a bad scaffold does not sit inert — it enters the
corpus *as* truth.

## Excluded

| Video key | Route | Removed | Why |
|---|---|---|---|
| `tf0hELD_M88` | planet-x | 2026-07-28 (#101) | ViTPose has never found the Climber. Best coverage ever recorded was 35%; the grid repair left it at **25/833 (3%)**. Nothing to accept truth against. |
| `w420jGWP2W0` | atman | 2026-07-28 (#101) | Poses confidently at **100%** coverage — on the wrong person. |
| `VxhW7T4vg7E` | rug-rat | 2026-07-28 (#101) | Poses confidently at **90%** coverage — on the wrong person. |

### The two failure modes are not the same

`tf0hELD_M88` fails *visibly*: coverage near zero, and every signal in the harness says
so. The other two fail *invisibly*, which is what makes them dangerous. High coverage,
a healthy-looking scaffold, a `seedHash`, no drift-detector flag — and the wrong subject
throughout. Every automated check the harness has reads them as good bundles.

That is the recurring shape tracked in issue #119 and the reason absence provenance
exists (ADR 0008): **when something looks healthy but the data disagrees, suspect the
signal before the data.** A wrong-subject scaffold is a case where the harness currently
has no signal at all, and the only detector is a human watching the overlay.

`w420jGWP2W0` and `VxhW7T4vg7E` both held truth recording every frame absent — 1815 and
653 frames — because the scanner could not match a single pose against their off-grid
scaffolds. That absence was an artifact of the grid defect, not of the wrong-subject
problem; the two coincided, and the second was only visible once the first was repaired.

## If one is re-added

Confirm on the overlay that the scaffold tracks the intended Climber before accepting
any truth. Detector resolution is the lever most likely to change a subject association
(`--imgsz`, see `docs/adr/` and issue #116) — it moves *which* person is detected, not
just how many.
