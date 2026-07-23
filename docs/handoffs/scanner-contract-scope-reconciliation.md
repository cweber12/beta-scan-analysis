# Handoff: reconcile the scanner contract PRD to its true scope

**Audience:** an agent working in the **Beta Scanner** repo (the Next.js pose/ORB app).
You do not need the analysis harness repo open — this doc tells you exactly how to trim
the scanner-side PRD so it is in sync with the harness, and what to split off into a
separate backlog item.

**Harness issue:** [#63](https://github.com/cweber12/beta-scan-analysis/issues/63)
(rewritten to docs-only scope). **Companion docs:**
[scanner-data-contract.md](scanner-data-contract.md) (bundle layout + the contract
probe) and [scanner-video-stats.md](scanner-video-stats.md) (probe shape + work item 0).

---

## Why

A grill session in the scanner repo produced a PRD — *"Pose Pipeline Contract Authority
for Analyzer Suggestions"* — that over-scoped the cross-program contract for a
local-only, one-maintainer, two-repo project. On review it turned out **not** to be a
mirror of harness #63 at all:

- **Harness #63** points *harness → scanner*: the harness self-describes its API and
  artifacts via `GET /api/contract`, and the scanner gates on it. **This already exists**
  and is structurally drift-proof (endpoints derived from the live route table, per-artifact
  schema versions, an additive `capabilities` map, one breaking-change `apiVersion`).
- **The scanner PRD** points *scanner → analyzer* and bolts an entirely separate, new
  product (a tuning-suggestion / calibration safety loop) onto a contract-governance shell.

So "sync the two docs" means **shrink both to the thin real overlap** — which is already
documented — not align two halves. The harness side is done (#63 rewritten, handoff docs
noted). This doc is the scanner side.

**The contract mechanism of record is `GET /api/contract`.** There is deliberately no
generated schema artifact, no second runtime schema endpoint, and no contract-governance
layer (CI drift gate / PR contract metadata) planned — considered and deferred until a
concrete drift incident justifies the cost.

---

## What to do: three buckets

Sort every element of the scanner PRD into one of these and act accordingly.

### Bucket A — DELETE (governance ceremony, cut on both sides)

Remove these from the PRD entirely; they are deferred until a concrete drift incident:

| PRD element | Stories | Why cut |
| --- | --- | --- |
| Source-derived deterministic contract generation | 4, 5 | `/api/contract` is already runtime-derived and can't drift |
| CI drift / compat / lifecycle / changelog / min-bump gates | 1, 12, 34, 35 | Governance for a one-maintainer local repo |
| Index-first discovery, pinned IDs, checksums, compat lanes/aliases | 2, 3 | No distribution boundary to justify it |
| Semver major/minor governance | 6, 7 | `apiVersion` + the `artifacts` version map already cover this |
| Two-module strict-schema + tunables-manifest split | 8 | No consumer needs deep-shape validation today |
| Deprecation lifecycle / replaced-by / exemption governance | 10, 11, 12 | Ceremony ahead of need |
| Committed-artifact endpoint serving | (impl decision) | This is the "second endpoint" — cut |
| Machine-readable changelog deltas | 35 | Deferred with the rest of the governance |
| Artifact-signing triggers | 36 | Explicitly premature |

### Bucket B — SPLIT OUT into a separate deferred PRD (see below)

The entire analyzer suggestion / calibration loop — **stories 13–31**. This has **no
harness counterpart** and describes a capability that does not exist: the harness emits
*label* suggestions (video-stats prefill), **not** tuning-knob suggestions. It is a
distinct future product, not part of the data contract. File it as its own backlog item
(stub below) and remove it from this PRD.

Covers: JSON-Patch suggestions + inverse patches (18, 19); preflight dry-run + whole-set
dependency validation (20, 21); one-per-tunable + confidence-threshold / top-N controls
(22, 23); review-only default + guarded auto-apply (24, 25, 26); heuristic→calibrated
confidence + sample floor (27, 28, 29); append-only redacted outcome logging (30, 31);
evidence-signal IDs + contribution weights (15, 16); typed dependency edges + primary/
secondary impact (13, 14); version/checksum provenance stamping (17).

### Bucket C — KEEP, reduced to a pointer (the real overlap)

Everything genuinely required is **already documented** — reduce the PRD to referencing it:

- **Probe + gate + degrade** (stories 32, 33 survive only as "degrade visibly when the
  probe fails/is stale") → already [scanner-video-stats.md](scanner-video-stats.md)
  **work item 0**: probe `GET {HARNESS_API_BASE}/api/contract` once at startup, cache it,
  gate features on `endpoints` / `artifacts` / `capabilities` / `suggestions`, degrade
  visibly (never a silent 404).
- **Write the bundle artifacts the harness reads** → already
  [scanner-data-contract.md](scanner-data-contract.md): `setup.json.analysisInputs`,
  `ground-truth.json`, and the detection diagnostics.

---

## The reconciled PRD body (replace the 38 stories with this)

> **Scanner ↔ harness contract (reconciled).** The scanner participates in one
> cross-program contract with the analysis harness. Its obligations are:
> 1. **Probe + gate.** Fetch `GET {HARNESS_API_BASE}/api/contract` once at startup, cache
>    it, gate harness-facing features on `endpoints` / `artifacts` / `capabilities` /
>    `suggestions`, and degrade visibly on mismatch or unreachable (see
>    `scanner-video-stats.md` work item 0).
> 2. **Write bundle artifacts.** Emit `setup.json.analysisInputs`, `ground-truth.json`,
>    and detection diagnostics per `scanner-data-contract.md`.
>
> **Out of scope.** Any scanner-generated contract system, machine-schema artifacts,
> second schema endpoint, or CI/PR contract governance — the harness `/api/contract` probe
> is the contract mechanism of record (harness #63). The analyzer tuning-suggestion loop is
> a separate, deferred concern (its own backlog item).

---

## The parked-PRD stub (file as its own backlog item)

> **Title:** Analyzer tuning-suggestion loop (deferred)
> **Status:** backlog / deferred — do not start.
>
> **Idea:** an analyzer that consumes pose-detection tunable semantics and emits *safe,
> explainable, reversible* tuning suggestions — JSON-Patch changes with inverse patches for
> one-step rollback, preflight + whole-set dependency validation, at most one suggestion per
> tunable per cycle, confidence that starts heuristic and calibrates from logged outcomes
> once a sample floor is met, review-only by default with optional guarded auto-apply, and
> append-only redacted outcome logging.
>
> **Why parked:** a distinct product from the cross-program *data* contract, not a schema
> concern. The harness emits only *label* suggestions (video-stats prefill) today; a
> tuning-suggestion engine is net-new invention. Deferred until there is a concrete need and
> a proper design pass; it must not ride along with the contract reconciliation (harness #63).
>
> **Source:** stories 13–31 of the original *"Pose Pipeline Contract Authority for Analyzer
> Suggestions"* PRD.

---

## Acceptance

- The scanner PRD's contract obligations reduce to the two-point reconciled body above; its
  Out-of-Scope mirrors harness #63.
- Bucket B lives only in the separate parked backlog item, not in the contract PRD.
- No new contract-generation, schema-artifact, or governance work is scheduled.
