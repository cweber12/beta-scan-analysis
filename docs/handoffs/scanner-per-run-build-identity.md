# Handoff: per-run build identity that cannot decouple from the executing code

**Audience:** an agent working in **beta-scanner**. You do not need the analysis harness
repo open — this is a *delta* on one field you already write into pose `diagnostics`.

**Companion docs:**
[scanner-data-contract.md](scanner-data-contract.md) (the `appVersion` contract this
amends — see "Things you already emit that the evaluation depends on") and
[scanner-detection-improvements-round-2.md](scanner-detection-improvements-round-2.md)
(where the `c305954` contamination was first reported to you).

**Harness refs:** issue
[#130](https://github.com/cweber12/beta-scan-analysis/issues/130), under PRD
[#129](https://github.com/cweber12/beta-scan-analysis/issues/129). **#130 is the first
slice and it gates the rest** — the next full-corpus sweep is waiting on it, because
every other slice assumes a run can be attributed to a build.

---

## The defect, concretely

`appVersion` in pose `diagnostics` is read from `NEXT_PUBLIC_APP_VERSION`, which Next
resolves **once at dev-server start**. A hot reload changes the running code without
changing the stamp. So the field answers "which build was this server started from",
while every consumer reads it as "which code produced this run".

This is not hypothetical. Measured on the corpus right now — **495 pose runs, all 495
stamped, 13 distinct builds**:

| Build | Runs | Days | Note |
|---|---|---|---|
| `deaa1c0` | 141 | 07-24 | the within-build repeat set — *assumed* one build |
| `e45e58f` | 94 | 07-28 | post-reset sweep |
| `c305954` | **67** | 07-25, 07-26 | **stamped 01, behaviourally ran 02's flip fix** |
| `495795e` | 55 | 07-21 | |
| …9 more | 138 | | 07-28 alone spans 4 distinct builds |

Two rows carry the whole argument:

- **`c305954`, 67 runs.** The stamp is wrong, and the 01-only no-drift control window
  is permanently lost. Nothing in the record can distinguish the 33 runs on 07-25 from
  the 34 on 07-26, and no evidence on disk says which — if either — ran 01.
- **`deaa1c0`, 141 runs in one day.** Harness issue
  [#134](https://github.com/cweber12/beta-scan-analysis/issues/134) wants to establish
  the run-to-run **variance floor** from this set. That number is only meaningful if
  "within-build" is actually true. If a hot reload landed mid-session, the measured
  floor silently absorbs a real behavioural delta and every later "this change is
  within noise" verdict inherits it.

That 07-28 spans four distinct builds is the point, not a counterexample: restarts *do*
happen, so the stamp is not useless — it is **unverifiable**. We cannot tell a clean
batch from a contaminated one, which means we have to treat all of them as suspect.

## Why the current mitigation is not enough

The mitigation of record is a line in the harness's `CLAUDE.md` telling humans to
restart the dev server before every batch, repeated to you in
[scanner-reset-sequencing-reply.md](scanner-reset-sequencing-reply.md). That is a
process workaround for a data-integrity defect, and **it has already failed once** —
which is exactly how `c305954` got into the corpus.

It also fails in a second way that a stricter process cannot fix: it is
**undetectable after the fact**. A batch is either trusted or discarded on the strength
of someone's memory of whether they restarted. There is no artifact to check.

The harness already got the symmetric case right. `_version_regression`
([trends.py:667-680](../../analysis_pipeline/trends.py#L667-L680)) restricts every
version comparison to `(video, truthHash)` pairs present on both sides, so **a truth
revision can never masquerade as a scanner change**. The opposite axis has no such
guard: it cannot stop **two different builds masquerading as one version**. That
asymmetry is what this closes.

---

## What to change

### 1. Emit a second identifier alongside `appVersion`

In pose `diagnostics` (`data.diagnostics`, where `appVersion` already lives — the
harness reads it at [trends.py:360](../../analysis_pipeline/trends.py#L360)):

```jsonc
"diagnostics": {
  "schemaVersion": 1,
  "appVersion": "e45e58f",              // unchanged — keep emitting it
  "detectorCodeHash": "9f2c1a7b4d80",   // NEW — derived from the executing code
  ...
}
```

**Do not replace `appVersion`.** Conflict detection needs both: the pair
*(same stamp, different hash)* is the `c305954` signature. One field alone detects
nothing.

Field name is the harness's suggestion, not a requirement — if a different name fits
your codebase, say so and the harness will read that instead. What matters are the
properties below.

### 2. What the identifier must satisfy

Mechanism is your call — hashing the detector module sources at request time is the
cheapest route, a compile-time manifest read *per request* (rather than per server
lifetime) also works. Either way:

- **Derived per request**, or per module-graph instantiation — never once per server
  lifetime. In Next dev a hot reload re-instantiates the module graph, which is the
  event the current stamp misses.
- **Covers the code that determines detection behaviour** — detector entry, the search
  ladder and its scales, crop geometry, the identity gate, recovery logic, and the
  constants they read. Not the UI, not styling. A change that cannot alter a keypoint
  need not move the hash, but **err toward over-covering**: a hash that moves
  spuriously costs some pooling, while one that fails to move reintroduces this exact
  defect.
- **Identical code ⇒ identical hash**, across restarts, checkouts, and machines. See
  the pitfalls below — this is the property most easily lost by accident.
- **Cheap and memoized** per module instantiation, not recomputed per frame. It must
  not show up in `inferenceMs`.
- **Same shape as the other hashes in this contract** — lowercase hex, truncated to
  12–16 chars is fine (`setupHash` and `seedHash` are the precedent).

### 3. Make it checkable before a batch is spent

Expose the current `detectorCodeHash` somewhere a human can read it without running a
scan — the dev corpus UI, or alongside whatever `hasGroundTruth` / `truthStale` already
surfaces. The failure this fixes costs a whole batch, and the cheapest possible check
is "does the hash on screen match the last run's".

### Pitfalls that quietly break the guarantee

- **Line endings.** This corpus is produced on Windows. A hash over raw source bytes
  differs between a CRLF and an LF checkout of *identical* code. Normalize newlines
  before hashing.
- **Absolute paths.** Hashing module paths, or `import.meta.url`, keys the hash to a
  working directory. Use repo-relative paths, or hash contents only.
- **Timestamps / build ids / `Date.now()`.** Anything that varies per build makes every
  run unique, which destroys pooling as thoroughly as a frozen stamp destroys
  attribution — just less visibly.
- **Non-deterministic module ordering.** If you hash a set of modules, sort them first.
- **Production builds.** The stamp cannot drift in a prod build, so dev is the path that
  matters. Emit the field in **both** anyway, so the harness never has to special-case
  which one it is reading.

---

## What the harness does with it

Records without the field degrade to today's behaviour — **unknown provenance, never a
conflict**. Fail-open is the established tradition on both sides of this contract; the
same null-guard discipline as
[scanner-truth-scaffold-provenance.md](scanner-truth-scaffold-provenance.md).

Once the field exists (harness-side work, tracked on #130 — not yours):

- `evaluate` **flags same-stamp/different-hash** rather than silently pooling across it.
  That is the `c305954` signature, and it becomes a visible flag on the batch instead of
  a postmortem.
- `_version_regression` groups by the pair, so a build that hot-reloaded mid-batch
  splits into its real behavioural groups instead of averaging them.
- **`deaa1c0`'s 141 runs get verified as a genuine within-build set** before #134 fits a
  variance floor to them — or get split, which is the more useful outcome if it happens.

One benefit runs the other way, and is worth having: *different* `appVersion` with the
**same** `detectorCodeHash` means a commit that did not touch detection. The harness can
legitimately pool those runs, which **increases** usable n instead of fragmenting it.
Today every commit looks like a potential behavioural change.

## Not in scope

- **Retroactively identifying what `c305954` actually ran.** Not recoverable; the
  evidence does not exist. #130 makes future occurrences detectable, not past ones.
- **Removing or renaming `appVersion`.** It stays, unchanged, as the human-readable
  anchor.
- **Any change to how detection itself works.** This slice is pure provenance. The
  detection-behaviour asks live in
  [scanner-detection-improvements-round-2.md](scanner-detection-improvements-round-2.md).
- **Dropping the restart-before-batch habit.** Keep it until the field ships and the
  harness is reading it.
