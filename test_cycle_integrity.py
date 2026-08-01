"""Stub-backed tests for the cycle integrity guard (issue #168).

No MediaPipe, no video, no batch. The whole guard is arithmetic over hashes and a detector
behind a Protocol, which is deliberate: a drift guard that costs an eight-hour cycle to
exercise is one nobody exercises, and one nobody exercises is indistinguishable from one
that cannot fire.

So most of these tests are **firings**. Asserting that a clean cycle certifies proves very
little — a guard that always certifies passes that test. Each check here is paired with the
mutation it is supposed to catch: truth revised, truth edited without re-stamping its hash,
a trajectory rebuilt under an unchanged config, a model swapped, an empty canary.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import crop_track
import cycle_integrity as ci
import mediapipe_job as mj

JOINTS = ("nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip")


# --------------------------------------------------------------------------- #
# Stub seam
# --------------------------------------------------------------------------- #

class StubDetector:
    """A detector whose output is a pure function of its inputs and an explicit ``offset``.

    Deterministic by default — that is the property the canary rests on, so the stub has to
    have it too. ``offset`` is how a test says "the detector moved between the two passes",
    and it is the only way this stub can produce different bytes.
    """

    def __init__(self, config, *, offset: float = 0.0, detects: bool = True,
                 model_sha: str | None = None) -> None:
        self.config = config
        self.offset = offset
        self.detects = detects
        self.model_sha = model_sha
        self.calls: list[tuple[float, ...]] = []

    def detect(self, video_path, timestamps, config, crop_track=None):
        self.calls.append(tuple(timestamps))
        out = {}
        for t in timestamps:
            if not self.detects:
                out[t] = []
                continue
            out[t] = [
                mj.Keypoint(name=name, x=round(0.1 * i + t / 100 + self.offset, 6),
                            y=round(0.2 * i + t / 200, 6), score=0.9)
                for i, name in enumerate(JOINTS)
            ]
        return out


def _factory(**kwargs):
    built: list[StubDetector] = []

    def factory(config):
        detector = StubDetector(config, **kwargs)
        built.append(detector)
        return detector

    factory.built = built  # type: ignore[attr-defined]
    return factory


# --------------------------------------------------------------------------- #
# Corpus fixture
# --------------------------------------------------------------------------- #

def _truth_doc(count: int, *, shift: float = 0.0, gt_hash: str | None = None) -> dict:
    frames = [
        {
            "frameIndex": i,
            "timestamp": round(i * 0.5, 3),
            "state": "present",
            "joints": {name: {"x": round(0.1 * j + shift, 4), "y": 0.2 * j}
                       for j, name in enumerate(JOINTS)},
        }
        for i in range(count)
    ]
    doc = {"version": 1, "jointSet": "coco13", "frames": frames, "setupHash": "sh"}
    if gt_hash is not None:
        doc["groundTruthHash"] = gt_hash
    return doc


def _bundle(root: Path, route: str, key: str, *, frames: int = 40, track: bool = True,
            gt_hash: str | None = None) -> Path:
    bundle = root / "analysis" / route / key
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / f"{key}.mp4").write_bytes(b"v")
    (bundle / "setup.json").write_text(
        json.dumps({"setupHash": f"sh-{key}", "climberPoint": {"x": 0.5, "y": 0.5, "t": 0}}),
        encoding="utf-8")
    (bundle / "ground-truth.json").write_text(
        json.dumps(_truth_doc(frames, gt_hash=gt_hash or f"gt-{key}")), encoding="utf-8")
    if track:
        boxes = [crop_track.CropBox(round(i * 0.5, 3), 0.5, 0.4, 0.15,
                                    crop_track.SRC_DETECTED)
                 for i in range(frames)]
        crop_track.write_crop_track(
            bundle, crop_track.CropTrack(config=crop_track.CropTrackConfig(), boxes=boxes,
                                        selected_half=0.15, setup_hash=f"sh-{key}"))
    return bundle


def _corpus(tmp: Path, keys=("canary", "b", "c")) -> Path:
    for key in keys:
        _bundle(tmp, "r", key)
    return tmp / "analysis"


def _open(root: Path, factory=None, **kwargs):
    return ci.open_cycle(
        root, factory or _factory(), canary_route="r", canary_video_key="canary", **kwargs)


def _close(root: Path, factory=None, **kwargs):
    return ci.close_cycle(root, factory or _factory(), **kwargs)


# --------------------------------------------------------------------------- #
# The manifest
# --------------------------------------------------------------------------- #

def test_the_manifest_snapshots_every_eligible_bundle_and_the_harness_identity():
    """A cycle records what its runs will be measured against, so a comparison published
    months later can be re-checked instead of trusted."""

    tmp = Path(tempfile.mkdtemp())
    try:
        root = _corpus(tmp)
        doc = _open(root)

        assert doc["status"] == ci.STATUS_OPEN
        assert doc["manifest"]["bundleCount"] == 3
        snapshots = {b["videoKey"]: b for b in doc["manifest"]["bundles"]}
        assert set(snapshots) == {"canary", "b", "c"}
        for entry in snapshots.values():
            assert entry["truthHash"] and entry["truthContentHash"]
            assert entry["truthHash"] != entry["truthContentHash"], (
                "a declared groundTruthHash must be snapshotted as declared, and the "
                "content hash taken independently — one cannot stand in for the other")
            assert entry["setupHash"] and entry["cropTrackContentHash"]
            assert entry["truthFrames"] == 40
        # The identity of the harness itself, which no per-run stamp carries.
        assert doc["moduleVersion"] == mj.MODULE_VERSION
        assert doc["sampleCoefficient"] == mj.SAMPLE_COEFFICIENT
        assert doc["modelLocks"], "all three model pins, not just the canary's"
        assert ci.cycle_path(root, doc["cycleId"]).is_file(), "written as a tracked artifact"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_truth_identity_matches_the_hash_evaluate_names_records_by():
    """The guard has to snapshot the *same* quantity that decides whether two records pair.

    ``evaluate`` names records ``<run_ts>_vs_<truthHash8>.json`` off ``groundTruthHash``
    when declared and a content hash otherwise. If this module computed anything else it
    would be guarding a hash nobody pools on.
    """

    from analysis_pipeline import evaluate

    tmp = Path(tempfile.mkdtemp())
    try:
        declared = _bundle(tmp, "r", "declared")
        undeclared = tmp / "analysis" / "r" / "undeclared"
        undeclared.mkdir(parents=True)
        (undeclared / "ground-truth.json").write_text(json.dumps(_truth_doc(5)),
                                                      encoding="utf-8")
        for bundle in (declared, undeclared):
            truth = evaluate.load_truth(bundle)
            source, truth_hash, content_hash, frames = ci.truth_identity(bundle)
            assert truth_hash == truth.truth_hash
            assert source == truth.source
            assert frames == len(truth.frames)
            assert len(content_hash) == 64
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# The canary
# --------------------------------------------------------------------------- #

def test_an_unchanged_detector_certifies_the_cycle():
    tmp = Path(tempfile.mkdtemp())
    try:
        root = _corpus(tmp)
        _open(root)
        doc = _close(root)

        assert doc["status"] == ci.STATUS_CERTIFIED and doc["certified"]
        assert doc["failures"] == []
        comparison = doc["canary"]["comparison"]
        assert comparison["identical"] and comparison["framesDiffering"] == 0
        assert comparison["framesCompared"] == 40
        assert doc["canary"]["opened"]["framesSha"] == doc["canary"]["closed"]["framesSha"]
        assert doc["verification"]["heldCount"] == 3
        assert len(doc["comparableBundles"]) == 3
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_detector_that_moved_by_one_ulp_fails_the_cycle_and_says_where():
    """The whole reason the canary is a byte comparison rather than a metric comparison: a
    change far below any PCK difference still fails, and fails *loudly*."""

    tmp = Path(tempfile.mkdtemp())
    try:
        root = _corpus(tmp)
        _open(root)
        doc = _close(root, _factory(offset=1e-6))

        assert doc["status"] == ci.STATUS_FAILED and not doc["certified"]
        assert doc["failures"] == [ci.FAILURE_CANARY_DRIFT]
        comparison = doc["canary"]["comparison"]
        assert not comparison["identical"]
        assert comparison["framesDiffering"] == 40
        assert comparison["firstDivergence"] == 0.0
        # Named fields, not just "the hashes differ".
        assert "framesSha" in {f["field"] for f in comparison["fields"]}
        assert comparison["differingFrames"], "a failure has to say which frames moved"
        assert comparison["differingFramesTruncated"] == 40 - ci.MAX_REPORTED_FRAMES
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_model_swap_is_named_as_the_field_that_moved():
    """The failure a canary exists for. The report has to name the *cause* — a reader who
    only learns that bytes differ still has to go and find out why."""

    tmp = Path(tempfile.mkdtemp())
    try:
        root = _corpus(tmp)
        _open(root, _factory(model_sha="aaaa"))
        doc = _close(root, _factory(model_sha="bbbb"))

        assert doc["failures"] == [ci.FAILURE_CANARY_DRIFT]
        moved = {f["field"]: (f["opened"], f["closed"])
                 for f in doc["canary"]["comparison"]["fields"]}
        assert moved["config.modelSha"] == ("aaaa", "bbbb")
        assert "configHash" in moved, "a model swap must move the arm stamp too (#165)"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_an_empty_canary_refuses_to_open_the_cycle():
    """An all-empty canary is byte-identical under any weights, so it would certify a model
    swap it never saw. Refusing at open costs two minutes; discovering it at close costs the
    whole cycle."""

    tmp = Path(tempfile.mkdtemp())
    try:
        root = _corpus(tmp)
        try:
            _open(root, _factory(detects=False))
        except ci.CycleIntegrityError as exc:
            message = str(exc)
        else:
            raise AssertionError("an empty canary must refuse, not warn")

        assert "REFUSED" in message
        # ...and the refusal is recorded, because a refusal is evidence too.
        refused = [d for d in ci.list_cycles(root) if d["status"] == ci.STATUS_REFUSED]
        assert len(refused) == 1
        assert refused[0]["failures"] == [ci.FAILURE_CANARY_UNWITNESSED]
        assert refused[0]["canary"]["opened"]["witnesses"] is False
        # A refused cycle is not an open one — it must not block the next attempt.
        assert ci.open_cycle_doc(root) is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_canary_arm_that_crops_refuses_a_bundle_with_no_trajectory():
    """Silently falling back to full frame is how a canary ends up empty: on the real canary
    Bundle full-frame detects 0% and the tracked crop reaches ~92%."""

    tmp = Path(tempfile.mkdtemp())
    try:
        root = _corpus(tmp)
        (root / "r" / "canary" / crop_track.ARTIFACT_NAME).unlink()
        try:
            _open(root)
        except ci.CycleIntegrityError as exc:
            assert crop_track.ARTIFACT_NAME in str(exc)
        else:
            raise AssertionError("a cropped canary arm with no trajectory must refuse")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_canary_is_an_instrument_and_writes_no_run():
    """Written as a run it would put a second copy of one Bundle under one arm into every
    pooled number — extra weight for bytes that are identical to data already there."""

    tmp = Path(tempfile.mkdtemp())
    try:
        root = _corpus(tmp)
        _open(root)
        _close(root)
        detections = root / "r" / "canary" / "detections"
        assert not detections.exists() or not list(detections.glob("*.json"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_closing_canary_reuses_the_snapshotted_frame_set():
    """Re-deriving the grid at close would let a truth revision on the canary Bundle change
    the frame set, and the byte comparison would then fail for a reason that is not detector
    drift. Truth movement is the snapshot's job, and it still reports it."""

    tmp = Path(tempfile.mkdtemp())
    try:
        root = _corpus(tmp)
        doc = _open(root)
        opened_grid = tuple(doc["canary"]["opened"]["timestamps"])

        # The canary Bundle's truth doubles in length — a different 12·√n grid entirely.
        (root / "r" / "canary" / "ground-truth.json").write_text(
            json.dumps(_truth_doc(80, gt_hash="gt-canary-v2")), encoding="utf-8")

        factory = _factory()
        closed = _close(root, factory)
        assert factory.built[-1].calls[-1] == opened_grid, (
            "the close pass must sample exactly the frames the open pass did")
        assert closed["canary"]["comparison"]["identical"], (
            "a truth revision is not detector drift and must not masquerade as it")
        excluded = {e["videoKey"]: e for e in closed["verification"]["excluded"]}
        assert ci.REASON_TRUTH_HASH in excluded["canary"]["reasons"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# The truth-hash snapshot
# --------------------------------------------------------------------------- #

def test_a_bundle_whose_truth_moved_is_excluded_and_named():
    """Ground Truth is human-edited between sessions. Arms that paired against different
    truth are silently incomparable, and nothing before this refused to pool them."""

    tmp = Path(tempfile.mkdtemp())
    try:
        root = _corpus(tmp)
        _open(root)
        (root / "r" / "b" / "ground-truth.json").write_text(
            json.dumps(_truth_doc(40, shift=0.05, gt_hash="gt-b-revised")), encoding="utf-8")
        doc = _close(root)

        verification = doc["verification"]
        assert verification["heldCount"] == 2 and verification["excludedCount"] == 1
        excluded = verification["excluded"][0]
        assert (excluded["route"], excluded["videoKey"]) == ("r", "b")
        assert ci.REASON_TRUTH_HASH in excluded["reasons"]
        assert excluded["moved"][0]["opened"] == "gt-b"
        assert excluded["moved"][0]["closed"] == "gt-b-revised"
        assert {"route": "r", "videoKey": "b"} not in doc["comparableBundles"]
        # An excluded Bundle is a finding about the corpus, not a failure of the cycle: the
        # other arms are still comparable to each other.
        assert doc["status"] == ci.STATUS_CERTIFIED
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_truth_edited_without_restamping_its_hash_is_still_caught():
    """The case the declared hash cannot see, and the reason the snapshot carries two.

    An edit that leaves ``groundTruthHash`` alone is invisible to record naming — both arms'
    records pair under the same name against different joints — which makes it exactly the
    kind of drift that pools silently.
    """

    tmp = Path(tempfile.mkdtemp())
    try:
        root = _corpus(tmp)
        _open(root)
        (root / "r" / "c" / "ground-truth.json").write_text(
            json.dumps(_truth_doc(40, shift=0.02, gt_hash="gt-c")), encoding="utf-8")
        doc = _close(root)

        excluded = {e["videoKey"]: e for e in doc["verification"]["excluded"]}
        assert set(excluded) == {"c"}
        assert excluded["c"]["reasons"] == [ci.REASON_TRUTH_CONTENT]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_trajectory_rebuilt_under_an_unchanged_config_is_caught():
    """``track_hash`` names the tracker *settings*, so a re-run that landed differently keeps
    the same arm stamp while the detector sees different pixels."""

    tmp = Path(tempfile.mkdtemp())
    try:
        root = _corpus(tmp)
        _open(root)
        bundle = root / "r" / "b"
        rebuilt = crop_track.CropTrack(
            config=crop_track.CropTrackConfig(), selected_half=0.15,
            boxes=[crop_track.CropBox(round(i * 0.5, 3), 0.62, 0.4, 0.15,
                                      crop_track.SRC_DETECTED) for i in range(40)])
        crop_track.write_crop_track(bundle, rebuilt)
        doc = _close(root)

        excluded = {e["videoKey"]: e for e in doc["verification"]["excluded"]}
        assert excluded["b"]["reasons"] == [ci.REASON_CROP_CONTENT]
        moved = {m["field"] for m in excluded["b"]["moved"]}
        assert moved == {"cropTrackContentHash"}, (
            "the tracker config is unchanged — only the trajectory moved")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_recalibration_and_disappearance_are_named_distinctly():
    """"The truth was revised" and "the truth is gone" call for different responses from
    whoever reads the artifact."""

    tmp = Path(tempfile.mkdtemp())
    try:
        root = _corpus(tmp)
        _open(root)
        (root / "r" / "b" / "setup.json").write_text(
            json.dumps({"setupHash": "sh-b-recalibrated"}), encoding="utf-8")
        (root / "r" / "c" / "ground-truth.json").unlink()
        doc = _close(root)

        excluded = {e["videoKey"]: e for e in doc["verification"]["excluded"]}
        assert ci.REASON_SETUP_HASH in excluded["b"]["reasons"]
        assert ci.REASON_TRUTH_GONE in excluded["c"]["reasons"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_bundle_that_appeared_mid_cycle_is_named_rather_than_counted():
    """It was never in the comparison, but a reader comparing corpus counts would otherwise
    find the cycle short and have no way to learn why."""

    tmp = Path(tempfile.mkdtemp())
    try:
        root = _corpus(tmp)
        _open(root)
        _bundle(tmp, "r", "late")
        doc = _close(root)

        assert doc["verification"]["added"] == [{"route": "r", "videoKey": "late"}]
        assert doc["verification"]["heldCount"] == 3
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Lifecycle and audit trail
# --------------------------------------------------------------------------- #

def test_a_second_cycle_cannot_open_while_one_is_still_open():
    """Two overlapping cycles would both claim the same batches, and a run cannot belong to
    two comparisons with different manifests."""

    tmp = Path(tempfile.mkdtemp())
    try:
        root = _corpus(tmp)
        first = _open(root)
        try:
            _open(root)
        except ci.CycleIntegrityError as exc:
            assert first["cycleId"] in str(exc)
        else:
            raise AssertionError("a second open must be refused")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_closed_cycle_cannot_be_closed_again():
    """A second close would overwrite the diff the first one recorded."""

    tmp = Path(tempfile.mkdtemp())
    try:
        root = _corpus(tmp)
        doc = _open(root)
        _close(root)
        try:
            _close(root, cycle_id=doc["cycleId"])
        except ci.CycleIntegrityError as exc:
            assert "not open" in str(exc)
        else:
            raise AssertionError("closing a closed cycle must be refused")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_closing_with_no_open_cycle_is_an_error_not_a_silent_pass():
    tmp = Path(tempfile.mkdtemp())
    try:
        root = _corpus(tmp)
        try:
            _close(root)
        except ci.CycleIntegrityError as exc:
            assert "No open cycle" in str(exc)
        else:
            raise AssertionError("there is nothing to certify")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_cycle_names_the_runs_it_covers_by_arm():
    """The audit trail: a published comparison naming a cycle can be checked against the
    runs the cycle actually contained, rather than whatever is on disk when someone re-reads
    it. Read off ``exp-`` filenames, which is what that prefix was for (#160)."""

    tmp = Path(tempfile.mkdtemp())
    try:
        root = _corpus(tmp)
        doc = _open(root)
        opened_ts = doc["openedRunTs"]

        detections = root / "r" / "b" / "detections"
        detections.mkdir(parents=True)
        for run_ts in (f"exp-{opened_ts}-1111aaaa-p0", f"exp-{opened_ts}-1111aaaa-p1",
                       f"exp-{opened_ts}-2222bbbb-p0",
                       "exp-19990101-000000-3333cccc-p0",   # before the cycle opened
                       "20260101-000000"):                  # a scanner run, not experimental
            (detections / f"{run_ts}_pose.json").write_text("{}", encoding="utf-8")

        runs = _close(root)["runs"]
        assert runs["runCount"] == 3, "only experimental runs inside the window"
        assert runs["bundlesWithRuns"] == 1
        assert set(runs["arms"]) == {"1111aaaa", "2222bbbb"}
        assert runs["arms"]["1111aaaa"] == {"runs": 2, "bundles": 1}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# The analysis-side reader (issue #176)
#
# ``analysis_pipeline.cycles`` reads this artifact rather than importing this module,
# because importing it would drag ``mediapipe_job`` → ``youtube_core`` → ``yt_dlp`` into
# the pipeline's import graph, which ADR 0003 and ADR 0012 exist to keep out. That trade
# only holds while the two halves agree, and this file is the one place that may import
# both — the same arrangement ``truth_identity`` has with ``evaluate.load_truth``.
# --------------------------------------------------------------------------- #

def test_the_pipeline_reader_agrees_with_what_this_module_writes():
    """Field for field, over an artifact this module actually produced.

    Not a shape assertion over a hand-built fixture: the failure this guards against is a
    field being *renamed here* and the reader silently falling back to a default, which a
    fixture written to the reader's expectations could never catch."""

    from analysis_pipeline import cycles as cy

    tmp = Path(tempfile.mkdtemp())
    try:
        root = _corpus(tmp)
        _open(root)
        # One Bundle's truth moves, one appears late: an artifact with something in every
        # list the reader has to split apart.
        (root / "r" / "c" / "ground-truth.json").write_text(
            json.dumps(_truth_doc(40, shift=0.05, gt_hash="gt-moved")), encoding="utf-8")
        _bundle(tmp, "r", "late")
        doc = _close(root)

        scope = cy.resolve_cycle(root)
        assert scope.posture == cy.POSTURE_CERTIFIED
        assert scope.cycle_id == doc["cycleId"]
        assert scope.status == doc["status"] == ci.STATUS_CERTIFIED
        assert scope.certified is doc["certified"] is True
        assert scope.opened_run_ts == doc["openedRunTs"]
        assert scope.closed_run_ts == doc["closedRunTs"]
        assert scope.module_version == doc["moduleVersion"] == mj.MODULE_VERSION
        assert scope.sample_coefficient == doc["sampleCoefficient"] == mj.SAMPLE_COEFFICIENT
        assert scope.model_locks == doc["modelLocks"]
        assert scope.canary["identical"] is doc["canary"]["comparison"]["identical"] is True
        assert scope.canary["frames_compared"] == doc["canary"]["comparison"]["framesCompared"]

        # The three populations the report must never conflate.
        assert scope.comparable == {(b["route"], b["videoKey"])
                                    for b in doc["comparableBundles"]}
        assert scope.bundle_state("r", "canary") == (cy.BUNDLE_COMPARABLE, "")
        state, detail = scope.bundle_state("r", "c")
        assert state == cy.BUNDLE_EXCLUDED and ci.REASON_TRUTH_HASH in detail
        assert scope.bundle_state("r", "late")[0] == cy.BUNDLE_NEWLY_ELIGIBLE
        assert scope.newly_eligible == (("r", "late"),)

        # ...and the run-id join, which must accept exactly what this module's census
        # accepts. Compared on behaviour rather than on the pattern string, which differs
        # only by ``re.escape``: a laxer pattern in the pipeline would pool runs the
        # cycle's own census never counted, and a stricter one would drop runs it did.
        for run_id, base in (
            ("exp-20260731-225432-fbd1fcab-p0", "20260731-225432"),
            ("exp-20260731-225432-fbd1fcab-p2-1", "20260731-225432"),
            ("20260731-124044-5672bf66-p0", None),      # pre-#160 id: neither side reads it
            ("exp-20260731-225432-XXXXXXXX-p0", None),  # not a hex arm digest
            ("20260101-000000", None),                  # a scanner run
        ):
            mine = cy.run_base_ts(run_id)
            theirs = ci.RUN_ID_PATTERN.match(run_id)
            assert mine == base
            assert (theirs.group(1) if theirs else None) == base
        for status in (ci.STATUS_OPEN, ci.STATUS_CERTIFIED, ci.STATUS_FAILED,
                       ci.STATUS_REFUSED):
            assert status in (cy.STATUS_OPEN, cy.STATUS_CERTIFIED, cy.STATUS_FAILED,
                              cy.STATUS_REFUSED)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_pipeline_reader_refuses_a_failed_cycles_comparable_bundles():
    """``close_cycle`` writes ``comparableBundles`` on a failed cycle too, and the artifact
    says in as many words not to publish a comparison over them. The reader must key on
    ``certified`` — reading the list would publish exactly what the guard forbids."""

    from analysis_pipeline import cycles as cy

    tmp = Path(tempfile.mkdtemp())
    try:
        root = _corpus(tmp)
        _open(root)
        doc = _close(root, _factory(offset=1e-6))     # the detector moved under the cycle

        assert doc["status"] == ci.STATUS_FAILED and doc["certified"] is False
        assert doc["comparableBundles"], "the failed artifact still lists them"

        scope = cy.resolve_cycle(root)
        assert scope.posture == cy.POSTURE_UNCERTIFIED
        assert scope.refuses is True and scope.gates is False
        assert scope.comparable == frozenset(), "a failed cycle certifies nothing"
        assert scope.pools("r", "canary") is False
        assert ci.FAILURE_CANARY_DRIFT in scope.failures
        assert scope.canary["identical"] is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_pipeline_reader_sees_an_open_cycle_as_in_flight():
    """Mid-sweep there is no verdict, and the one thing the report must not do is render a
    provisional comparison as a certified one."""

    from analysis_pipeline import cycles as cy

    tmp = Path(tempfile.mkdtemp())
    try:
        root = _corpus(tmp)
        doc = _open(root)

        scope = cy.resolve_cycle(root)
        assert scope.posture == cy.POSTURE_IN_FLIGHT
        assert scope.certified is False
        assert scope.gates is False and scope.refuses is False
        assert scope.comparable == frozenset()
        assert scope.closed_run_ts == ""
        assert scope.bundle_state("r", "b")[0] == cy.BUNDLE_SNAPSHOTTED
        # The window is open-ended, so a run landing now is inside it.
        assert scope.place_run(f"exp-{doc['openedRunTs']}-1111aaaa-p0") == cy.RUN_INSIDE
        assert scope.place_run("exp-19990101-000000-1111aaaa-p0") == cy.RUN_BEFORE
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_guard_runs_without_mediapipe_installed():
    """Same property ``test_mediapipe_job.py`` has: everything that decides whether a cycle
    is comparable is pure, and the detector is behind the same Protocol."""

    assert "mediapipe" not in sys.modules, (
        "importing cycle_integrity must not import mediapipe")
    assert ci.CANARY_CROP == mj.CROP_TRACKED, (
        "the canary arm must crop — an uncropped canary on this Bundle detects 0% and "
        "would pass through a model swap")


def _run_all():
    fns = [fn for name, fn in sorted(globals().items())
           if name.startswith("test_") and callable(fn)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")


if __name__ == "__main__":
    sys.exit(_run_all())
