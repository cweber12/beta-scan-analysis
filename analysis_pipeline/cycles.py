"""The Cycle, read from the analysis side — what the arm comparison is allowed to pool.

#168 built the **Cycle**: a comparison group opened before the first batch and closed after
the last, with a determinism canary and a truth-hash snapshot spanning it, writing
``comparableBundles`` at close — the Bundles whose inputs demonstrably never moved between
the first arm and the last. #164 built the arm-versus-arm reporting. Nothing connected
them, so the report pooled every harness run on disk regardless of whether anything had
certified that the ground under it held still. This module is that connection (issue #176).

**Read, never imported.** ``cycle_integrity`` writes the artifact, and importing it here
would drag ``mediapipe_job`` → ``youtube_core`` → ``yt_dlp`` and ``vitpose_job`` →
``video_stats`` into the ``analysis_pipeline`` import graph, which ADR 0003 and ADR 0012
exist to keep out. So the manifest is read as JSON, exactly as every other artifact in this
pipeline is, and ``test_cycle_integrity.py`` — which may import both sides — asserts the
reader and the writer agree field by field. That is the same trade
``cycle_integrity.truth_identity`` already makes in the opposite direction.

**Four postures, not one rule.** #132 settled gate-versus-covariate on a single test: *is
the gate criterion correlated with the thing being measured?* For the #15 conformance gate
on failure-mode metrics the answer is yes — the runs that fail it are disproportionately
the runs where detection went badly — so gating there selects on the outcome and the
treatment is a covariate. The Cycle's criterion is different in kind: a Bundle leaves
``comparableBundles`` because a human re-seeded its truth or recalibrated its setup, which
are **operational events, not detector outcomes**. So the hazard #132 guards against is
much weaker here, and the rule splits by what a number *is* rather than by section:

1. ``certified`` — **gate** the pooled arm summary and the deltas. Those are truth-fit
   numbers in #132's sense: a Bundle whose truth moved mid-Cycle makes a delta that
   silently contains a truth change, which is the exact confound PRD #156 exists to escape.
   The per-Bundle table keeps every row, with comparability as a column — nothing vanishes.
2. ``failed`` / ``refused`` — **refuse.** ``close_cycle`` writes ``comparableBundles`` even
   when it fails and logs *"The arms in this cycle are NOT comparable to each other. Do not
   publish a comparison over them."* Keying on the presence of that list rather than on
   ``certified`` would publish precisely what the artifact forbids, so no pooled comparison
   is rendered at all — the runs are still named, as evidence, but not laid out as one.
3. ``open`` — **in flight.** ``comparableBundles`` is written only at close, so there is
   nothing to gate on yet. The comparison is provisional and says so; it is never rendered
   as certified.
4. no Cycle at all — **label, don't gate.** This is the whole pre-#168 corpus and any probe
   run outside a Cycle. There is nothing to gate against, so the honest output is the
   comparison it renders today plus an explicit "not drift-checked" marker. Never silence,
   and never something a reader could take for a certified result.

This keeps the layering coherent with ``app.py``, which deliberately reports the enclosing
cycle in the batch 202 and never gates on it: a batch outside a Cycle is legitimate — the
batch is allowed to run anything, and the *report* is what refuses to pool it.

**The window is the join.** Nothing durable stamps a run with its Cycle: the association
lives in the batch 202 response and nowhere else. What a run does carry is its base
timestamp, in the ``exp-<ts>-<arm8>-p<n>`` id #160 introduced, so the Cycle's
``(openedRunTs, closedRunTs)`` window is what places a run inside or outside it — the same
join ``cycle_integrity.collect_cycle_runs`` uses. A timestamp window is a weaker join than
a stamp, so every run it drops is named rather than merely excluded.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CYCLES_DIR_NAME = "cycles"

# Mirrors ``cycle_integrity``'s vocabulary. Duplicated rather than imported for the reason
# in the module docstring, and asserted equal in ``test_cycle_integrity.py``.
STATUS_OPEN = "open"
STATUS_CERTIFIED = "certified"
STATUS_FAILED = "failed"
STATUS_REFUSED = "refused"

# What the report does with the arm comparison, given the Cycle it found.
POSTURE_NONE = "no-cycle"
POSTURE_IN_FLIGHT = "in-flight"
POSTURE_CERTIFIED = "certified"
POSTURE_UNCERTIFIED = "uncertified"

# Named so the report can state which rule it applied, per the #132 precedent that no
# section may be read as if it were the other.
RULE = {
    POSTURE_CERTIFIED: "gate",
    POSTURE_UNCERTIFIED: "refuse",
    POSTURE_IN_FLIGHT: "label (provisional)",
    POSTURE_NONE: "label (not drift-checked)",
}

# Where a Bundle stands relative to the Cycle. ``newly-eligible`` is deliberately distinct
# from ``excluded``: those Bundles were never snapshotted, so they did not fail anything.
BUNDLE_COMPARABLE = "comparable"
BUNDLE_EXCLUDED = "excluded"
BUNDLE_NEWLY_ELIGIBLE = "newly-eligible"
BUNDLE_NOT_ELIGIBLE = "not-eligible"
BUNDLE_SNAPSHOTTED = "snapshotted"        # open cycle: in the manifest, verdict pending
BUNDLE_UNKNOWN = "outside-cycle"          # in no list the manifest holds
BUNDLE_NO_CYCLE = "no-cycle"

# Where a run sits relative to the window. ``unplaceable`` is not "outside": a run written
# before #160's ``exp-`` id convention carries no id this join can read — and neither can
# ``collect_cycle_runs``, so the Cycle's own census cannot see it either. Saying that beats
# calling it out-of-window, which would imply the window was consulted.
RUN_INSIDE = "inside"
RUN_BEFORE = "before-cycle"
RUN_AFTER = "after-cycle"
RUN_UNPLACEABLE = "unplaceable"

# ``exp-<base_ts>-<arm8>-p<pass>``, optionally with the writer's collision suffix — the
# same id ``cycle_integrity.RUN_ID_PATTERN`` matches.
RUN_ID_PATTERN = re.compile(r"^exp-(\d{8}-\d{6})-([0-9a-f]{8})-p(\d+)(?:-\d+)?$")


def run_base_ts(run_ts: str) -> str | None:
    """The batch timestamp inside an experimental run id, or ``None`` if it has none.

    Matches ``cycle_integrity.RUN_ID_PATTERN`` exactly, deliberately: a laxer pattern here
    would place runs inside a window that the Cycle's own ``runs`` census — which reads the
    strict form — never counted, and the report would then pool runs the artifact says the
    Cycle does not contain.

    ``None`` is a third answer and must not collapse into "outside". A run whose id predates
    the ``exp-`` convention (#160) cannot be *placed* by either side, which is a different
    statement from being outside the window, and the report says which.
    """

    match = RUN_ID_PATTERN.match(str(run_ts or ""))
    return match.group(1) if match else None


def _bundle_key(entry: Any) -> tuple[str, str] | None:
    if not isinstance(entry, dict):
        return None
    route, key = entry.get("route"), entry.get("videoKey")
    if not route or not key:
        return None
    return (str(route), str(key))


@dataclass(frozen=True)
class CycleScope:
    """One Cycle as the arm comparison needs it — or the absence of one.

    Always constructed, never ``None``: a corpus with no Cycle is a *posture*, not a missing
    object, and making callers branch on ``None`` is how the no-Cycle case ends up untested.
    """

    posture: str = POSTURE_NONE
    cycle_id: str = ""
    status: str = ""
    certified: bool = False
    opened_run_ts: str = ""
    closed_run_ts: str = ""
    opened_at: str = ""
    closed_at: str = ""
    module_version: str = ""
    sample_coefficient: int | None = None
    model_locks: dict[str, str] = field(default_factory=dict)
    comparable: frozenset[tuple[str, str]] = frozenset()
    excluded: tuple[dict[str, Any], ...] = ()
    newly_eligible: tuple[tuple[str, str], ...] = ()
    not_eligible: dict[tuple[str, str], str] = field(default_factory=dict)
    snapshotted: frozenset[tuple[str, str]] = frozenset()
    failures: tuple[str, ...] = ()
    canary: dict[str, Any] = field(default_factory=dict)
    runs: dict[str, Any] = field(default_factory=dict)
    bundle_count: int = 0
    other_cycles: tuple[dict[str, Any], ...] = ()

    # -- what the report is allowed to do ---------------------------------- #

    @property
    def rule(self) -> str:
        return RULE.get(self.posture, RULE[POSTURE_NONE])

    @property
    def gates(self) -> bool:
        """Whether the pooled arm numbers are restricted to ``comparableBundles``."""

        return self.posture == POSTURE_CERTIFIED

    @property
    def refuses(self) -> bool:
        """Whether a pooled comparison may be published at all."""

        return self.posture == POSTURE_UNCERTIFIED

    @property
    def scopes_runs(self) -> bool:
        """Whether the ``(openedRunTs, closedRunTs)`` window applies to the run population.

        A Cycle in any state scopes its own runs — including a failed one, whose window is
        exactly the span the failure applies to. With no Cycle there is no window and the
        population is every harness run on disk, as it is today.
        """

        return self.posture != POSTURE_NONE

    # -- placing runs and Bundles ------------------------------------------ #

    def place_run(self, run_ts: str) -> str:
        """Where one run sits relative to this Cycle's window."""

        if not self.scopes_runs:
            return RUN_INSIDE
        base = run_base_ts(run_ts)
        if base is None:
            return RUN_UNPLACEABLE
        if self.opened_run_ts and base < self.opened_run_ts:
            return RUN_BEFORE
        if self.closed_run_ts and base > self.closed_run_ts:
            return RUN_AFTER
        return RUN_INSIDE

    def bundle_state(self, route: str, video_key: str) -> tuple[str, str]:
        """``(state, detail)`` for one Bundle — the comparability column's contents.

        Every state that is not ``comparable`` carries *why* in ``detail``, because a Bundle
        dropped from a comparison is never silently dropped (the #15/#88 precedent).
        """

        key = (str(route), str(video_key))
        if self.posture == POSTURE_NONE:
            return (BUNDLE_NO_CYCLE, "no Cycle on this corpus — not drift-checked")
        if key in self.comparable:
            return (BUNDLE_COMPARABLE, "")
        for entry in self.excluded:
            if _bundle_key(entry) == key:
                return (BUNDLE_EXCLUDED, ", ".join(entry.get("reasons") or ()) or "moved")
        if key in self.newly_eligible:
            return (BUNDLE_NEWLY_ELIGIBLE,
                    "became eligible after the Cycle opened — never snapshotted, so it "
                    "was never in the Cycle rather than excluded from it")
        if key in self.not_eligible:
            return (BUNDLE_NOT_ELIGIBLE, self.not_eligible[key])
        if key in self.snapshotted:
            return (BUNDLE_SNAPSHOTTED, "snapshotted; the Cycle has not closed, so no "
                                        "verdict exists yet")
        return (BUNDLE_UNKNOWN, "not named anywhere in this Cycle's manifest")

    def pools(self, route: str, video_key: str) -> bool:
        """Whether this Bundle may contribute to a **pooled** arm number."""

        if self.refuses:
            return False
        if not self.gates:
            return True
        return (str(route), str(video_key)) in self.comparable

    def as_dict(self) -> dict[str, Any]:
        """Flat summary for the report and the CSV export."""

        return {
            "posture": self.posture,
            "rule": self.rule,
            "cycle_id": self.cycle_id,
            "status": self.status,
            "certified": self.certified,
            "opened_run_ts": self.opened_run_ts,
            "closed_run_ts": self.closed_run_ts,
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "module_version": self.module_version,
            "sample_coefficient": self.sample_coefficient,
            "model_locks": dict(self.model_locks),
            "comparable_count": len(self.comparable),
            "excluded_count": len(self.excluded),
            "newly_eligible_count": len(self.newly_eligible),
            "bundle_count": self.bundle_count,
            "failures": list(self.failures),
            "canary": dict(self.canary),
            "runs": dict(self.runs),
            "other_cycles": [dict(c) for c in self.other_cycles],
        }


# --------------------------------------------------------------------------- #
# Reading the artifacts
# --------------------------------------------------------------------------- #

def cycles_dir(analysis_root: Path) -> Path:
    return Path(analysis_root) / CYCLES_DIR_NAME


def list_cycle_docs(analysis_root: Path) -> list[dict[str, Any]]:
    """Every readable Cycle artifact, oldest id first. Unreadable ones are skipped.

    Skipped rather than raised on: a half-written manifest must not take the whole report
    down, and the count of Cycles on disk is surfaced beside the resolved one so a
    disappearance is visible.
    """

    out: list[dict[str, Any]] = []
    for path in sorted(cycles_dir(analysis_root).glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and doc.get("cycleId"):
            out.append(doc)
    return out


def _canary_summary(doc: dict[str, Any]) -> dict[str, Any]:
    canary = doc.get("canary") or {}
    comparison = canary.get("comparison") or {}
    opened = canary.get("opened") or {}
    closed = canary.get("closed") or {}
    return {
        "route": canary.get("route") or "",
        "video_key": canary.get("videoKey") or "",
        "identical": comparison.get("identical"),
        "frames_compared": comparison.get("framesCompared"),
        "frames_differing": comparison.get("framesDiffering"),
        "first_divergence": comparison.get("firstDivergence"),
        "fields": list(comparison.get("fields") or ()),
        "opened_detection_rate": opened.get("detectionRate"),
        "closed_detection_rate": closed.get("detectionRate"),
        "opened_witnesses": opened.get("witnesses"),
        "closed_witnesses": closed.get("witnesses"),
        "min_detection_rate": canary.get("minDetectionRate"),
        "config_hash": opened.get("configHash") or "",
    }


def scope_from_doc(doc: dict[str, Any], others: list[dict[str, Any]] | None = None) -> CycleScope:
    """One Cycle artifact as a ``CycleScope``.

    ``comparableBundles`` is read only for a **certified** Cycle. A failed one carries the
    list too — ``close_cycle`` writes it either way — and reading it there is the one
    mistake that would publish a comparison the artifact explicitly forbids.
    """

    status = str(doc.get("status") or "")
    certified = bool(doc.get("certified"))
    if status == STATUS_OPEN:
        posture = POSTURE_IN_FLIGHT
    elif status == STATUS_CERTIFIED and certified:
        posture = POSTURE_CERTIFIED
    else:
        posture = POSTURE_UNCERTIFIED

    manifest = doc.get("manifest") or {}
    selection = manifest.get("selection") or {}
    verification = doc.get("verification") or {}

    comparable: frozenset[tuple[str, str]] = frozenset()
    if posture == POSTURE_CERTIFIED:
        comparable = frozenset(
            k for k in (_bundle_key(e) for e in (doc.get("comparableBundles") or ()))
            if k is not None)

    not_eligible: dict[tuple[str, str], str] = {}
    for entry in selection.get("excluded") or ():
        key = _bundle_key(entry)
        if key is not None:
            not_eligible[key] = str(entry.get("reason") or "not eligible")

    snapshotted = frozenset(
        k for k in (_bundle_key(b) for b in (manifest.get("bundles") or ()))
        if k is not None)

    newly_eligible = tuple(
        k for k in (_bundle_key(e) for e in (verification.get("added") or ()))
        if k is not None)

    return CycleScope(
        posture=posture,
        cycle_id=str(doc.get("cycleId") or ""),
        status=status,
        certified=certified,
        opened_run_ts=str(doc.get("openedRunTs") or ""),
        closed_run_ts=str(doc.get("closedRunTs") or ""),
        opened_at=str(doc.get("openedAt") or ""),
        closed_at=str(doc.get("closedAt") or ""),
        module_version=str(doc.get("moduleVersion") or ""),
        sample_coefficient=(int(doc["sampleCoefficient"])
                            if isinstance(doc.get("sampleCoefficient"), (int, float))
                            else None),
        model_locks={str(k): str(v) for k, v in (doc.get("modelLocks") or {}).items()},
        comparable=comparable,
        excluded=tuple(e for e in (verification.get("excluded") or ()) if isinstance(e, dict)),
        newly_eligible=newly_eligible,
        not_eligible=not_eligible,
        snapshotted=snapshotted,
        failures=tuple(str(f) for f in (doc.get("failures") or ())),
        canary=_canary_summary(doc),
        runs=dict(doc.get("runs") or {}),
        bundle_count=int(manifest.get("bundleCount") or len(snapshotted)),
        other_cycles=tuple(
            {"cycle_id": str(d.get("cycleId") or ""), "status": str(d.get("status") or ""),
             "certified": bool(d.get("certified"))}
            for d in (others or ())),
    )


def resolve_cycle(analysis_root: Path) -> CycleScope:
    """The Cycle the arm comparison is read under.

    An **open** Cycle wins over every closed one: a sweep in flight is the comparison a
    reader is looking at, and rendering the last certified Cycle over the top of it would
    show a verdict for runs that are not the ones being produced. Otherwise the
    most-recently-closed Cycle applies, and the ones before it are named but not merged —
    two Cycles are two comparison groups, and pooling them would rebuild exactly the
    unattributable population the Cycle exists to prevent.
    """

    docs = list_cycle_docs(analysis_root)
    if not docs:
        return CycleScope()
    open_docs = [d for d in docs if d.get("status") == STATUS_OPEN]
    if open_docs:
        chosen = open_docs[-1]
    else:
        chosen = max(docs, key=lambda d: (str(d.get("closedRunTs") or ""),
                                          str(d.get("cycleId") or "")))
    others = [d for d in docs if d is not chosen]
    return scope_from_doc(chosen, others)


# --------------------------------------------------------------------------- #
# Applying it to the arm frames
# --------------------------------------------------------------------------- #

def tag_runs(arms: Any, cycle: CycleScope) -> Any:
    """Add ``cycle_placement`` / ``in_cycle_window`` to the per-run arm frame.

    Tagged rather than filtered here, so the full harness population stays in the origin
    accounting and the CSV export; the comparison takes the in-window subset from it.
    """

    if arms is None:
        return arms
    out = arms.copy()
    placements = [cycle.place_run(ts) for ts in out["run_ts"]] if not out.empty else []
    out["cycle_placement"] = placements
    out["in_cycle_window"] = [p == RUN_INSIDE for p in placements]
    return out


def runs_outside_window(arms: Any, cycle: CycleScope) -> list[dict[str, Any]]:
    """Every harness run the window drops, named — never a silent difference in a count."""

    if arms is None or getattr(arms, "empty", True) or "cycle_placement" not in arms.columns:
        return []
    rows: list[dict[str, Any]] = []
    for r in arms[~arms["in_cycle_window"]].itertuples():
        rows.append({
            "route_folder": r.route_folder,
            "video_key": r.video_key,
            "run_ts": r.run_ts,
            "base_ts": run_base_ts(r.run_ts) or "",
            "config_hash": r.config_hash,
            "arm": r.arm,
            "placement": r.cycle_placement,
        })
    return sorted(rows, key=lambda d: (d["placement"], d["base_ts"], d["route_folder"],
                                       d["video_key"]))


def tag_bundles(per_bundle: Any, cycle: CycleScope) -> Any:
    """Add the comparability covariate to the per-(arm, Bundle) frame.

    Every row survives. This is the covariate half of the rule: the per-Bundle table shows
    every Bundle the arms ran on, marked, and the pooled lines above it are what the gate
    removes rows from.
    """

    if per_bundle is None:
        return per_bundle
    out = per_bundle.copy()
    if out.empty:
        for col in ("cycle_state", "cycle_detail", "cycle_comparable"):
            if col not in out.columns:
                out[col] = []
        return out
    states = [cycle.bundle_state(r.route_folder, r.video_key) for r in out.itertuples()]
    out["cycle_state"] = [s for s, _ in states]
    out["cycle_detail"] = [d for _, d in states]
    out["cycle_comparable"] = [cycle.pools(r.route_folder, r.video_key)
                               for r in out.itertuples()]
    return out


def pooled_subset(per_bundle: Any, cycle: CycleScope) -> Any:
    """The rows a pooled arm number may be computed from, under the settled rule.

    ``certified`` gates to ``comparableBundles``; ``failed``/``refused`` yields nothing at
    all; ``open`` and no-Cycle pass everything through, labelled by the caller.
    """

    if per_bundle is None or getattr(per_bundle, "empty", True):
        return per_bundle
    if cycle.refuses:
        return per_bundle.iloc[0:0]
    if not cycle.gates:
        return per_bundle
    return per_bundle[per_bundle["cycle_comparable"]].reset_index(drop=True)


def cycle_bundle_rows(cycle: CycleScope) -> list[dict[str, Any]]:
    """The Cycle's own verdict on every Bundle it names, for the report and the CSV.

    Held Bundles are not listed individually — there are 84 of them on this corpus and the
    interesting rows are the ones that moved — but every excluded and newly-eligible Bundle
    is named with its verdict, which is the whole point of the artifact recording them.
    """

    rows: list[dict[str, Any]] = []
    for entry in cycle.excluded:
        key = _bundle_key(entry)
        if key is None:
            continue
        rows.append({
            "route_folder": key[0],
            "video_key": key[1],
            "state": BUNDLE_EXCLUDED,
            "reasons": ", ".join(entry.get("reasons") or ()),
            "detail": "; ".join(
                f"{m.get('field')}: {m.get('opened')!r} → {m.get('closed')!r}"
                for m in (entry.get("moved") or ()) if isinstance(m, dict)),
        })
    for route, key in cycle.newly_eligible:
        rows.append({
            "route_folder": route, "video_key": key,
            "state": BUNDLE_NEWLY_ELIGIBLE, "reasons": "",
            "detail": "became eligible after the Cycle opened; never snapshotted",
        })
    return sorted(rows, key=lambda d: (d["state"], d["route_folder"], d["video_key"]))
