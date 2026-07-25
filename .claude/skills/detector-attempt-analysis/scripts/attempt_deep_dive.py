"""Detector-attempt corpus deep-dive (companion to the detector-attempt-analysis skill).

Evaluation records used to be a thin wrapper around the raw ``detectorAttempts[]``
streams, so this script re-derived most of its tables by walking the streams itself.
That is no longer true: records now carry rejection verdicts (#85), crop IoU /
containment / miss causes (#86) and non-conformance causes (#88) as **record fields**,
scored under thresholds the record stamps on itself. Re-deriving them here would mean
maintaining a second, silently-diverging copy of that scoring, so the script reads the
fields and walks the streams only for what no record carries.

  1. corpus health     — records: conformance + cause, held/frozen, PCK, presence
  2. attempt funnel    — records: status mix, pooled and distributed over runs
  3. crop quality      — records: ``cropQuality`` IoU / containment / miss causes / flags
  4. rejection gates   — records: ``frameQuality.rejectionCorrectness`` verdicts
  5. raw-stream extras — streams only: selectionMethod, reacquire success, condition
                          luma / sharpness by status, status run-lengths, full-frame crops
  6. per-run table     — records

Record sections score the **truth-matched** attempts of each run (the population every
other record metric uses); the raw-stream section scores every attempt in the run. The
two therefore differ by a percentage point or so on the status mix — that is a
population difference, not a disagreement. For the full-stream funnel with proper
run-unit median / p90 / tail columns, prefer the standing report's
``eval_attempt_funnel_{status,runs,run_stats,flags}.csv`` (#87).

Usage:  python .claude/skills/detector-attempt-analysis/scripts/attempt_deep_dive.py \
            [analysis_root] [--runs-prefix 20260724-16]

``--runs-prefix`` filters *every* section to detection runs whose run_ts starts with it.
Numpy/pandas-free on purpose — stdlib only.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from pathlib import Path

STATUS_ORDER = ["accepted", "missing", "flipRejected", "qualityRejected", "unknown"]
REJECTION_STATUSES = ["flipRejected", "qualityRejected"]
VERDICTS = ["goodPoseRejected", "badPoseRejected", "truthUnknown"]


def load(p: str) -> dict:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def med(xs, nd=1):
    return quant(xs, 0.5, nd)


def quant(xs, q, nd=1):
    """Linear-interpolated quantile — numpy's default, so a median/p90 printed here and
    the same statistic in the standing report's funnel CSVs (#87) are the same number."""
    xs = sorted(x for x in xs if isinstance(x, (int, float)))
    if not xs:
        return None
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo), nd)


def pct(a, b):
    return f"{a / b:.1%}" if b else "-"


def rate(v):
    return "-" if v is None else f"{v:.1%}"


def attempt_records(root: str, runs_prefix: str = ""):
    """Current attempt-backed evaluation records (schema v8+): (path, record).

    A floor, not an equality: the record schema keeps gaining additive blocks (v9
    rejection correctness, v10 crop quality, v11 non-conformance cause), and pinning to
    one version would silently empty this script the first time the corpus regenerates."""
    out = []
    for r in glob.glob(os.path.join(root, "*", "*", "evaluations", "*.json")):
        d = load(r)
        if (d.get("schemaVersion") or 0) < 8:
            continue
        fq = d.get("frameQuality") or {}
        if fq.get("detectorEvidence") != "attempts":
            continue
        if runs_prefix and not str(d.get("runTs") or "").startswith(runs_prefix):
            continue
        out.append((r, d))
    return out


def corpus_health(recs) -> None:
    print("== corpus health ==")
    print(f"attempt-backed schema-v8 records: {len(recs)}")
    conform = sum(1 for _, d in recs if (d.get("conformance") or {}).get("conforms"))
    print(f"conforming (#15 gate): {conform}/{len(recs)}")
    # Read the cause off the record (issue #88) — never re-derive it from miss rates here.
    causes = collections.Counter(
        (d.get("conformance") or {}).get("cause") or "unannotated (pre-v11)"
        for _, d in recs if not (d.get("conformance") or {}).get("conforms"))
    for cause, n in causes.most_common():
        print(f"  non-conforming cause {cause}: {n}")

    tot = held = froz = pck_c = pck_t = 0
    pres = collections.Counter()
    cls = collections.Counter()
    for _, d in recs:
        fq = d["frameQuality"]
        tot += fq.get("detectedFrames") or 0
        held += fq.get("heldPoseCount") or 0
        froz += fq.get("frozenStaleCount") or 0
        for k, v in (fq.get("classCounts") or {}).items():
            cls[k] += v
        agr = d.get("agreement") or {}
        for k, v in (agr.get("presence") or {}).items():
            pres[k] += v
        pck = ((agr.get("aggregate") or {}).get("pck") or {})
        pck_c += pck.get("correct") or 0
        pck_t += pck.get("total") or 0
    print(f"detected frames: {tot} | heldPose {pct(held, tot)} | frozenStale {pct(froz, tot)}")
    pd_, pu = pres["presentDetected"], pres["presentUndetected"]
    ad, au = pres["absentDetected"], pres["absentUndetected"]
    print(f"detect rate on present: {pct(pd_, pd_ + pu)} | hallucination on absent: {pct(ad, ad + au)}")
    print(f"pooled PCK@0.5-torso: {pct(pck_c, pck_t)}")
    n_cls = sum(cls.values())
    print("frameQuality classes:",
          {k: f"{v} ({v / n_cls:.1%})" for k, v in cls.most_common()})


def attempt_funnel(recs) -> None:
    """Status mix off ``detectorAttemptStatusCounts`` — the record's own tally of the
    truth-matched attempts, pooled and distributed over runs.

    The per-run distribution is here because the pooled share alone is a trap: a corpus
    where one long run misses everything and one where every run misses a quarter of its
    frames pool identically."""

    stat = collections.Counter()
    shares = collections.defaultdict(list)
    for _, d in recs:
        st = d["frameQuality"].get("detectorAttemptStatusCounts") or {}
        n = sum(st.values())
        for s, c in st.items():
            stat[s] += c
            if n:
                shares[s].append(c / n)
    n_att = sum(stat.values())
    print(f"\n== attempt funnel ({len(recs)} runs, {n_att} truth-matched attempts) ==")
    print("statuses:", {k: f"{v} ({v / n_att:.1%})" for k, v in stat.most_common() if v})
    print(f"{'status':16s} {'pooled':>8s} {'run med':>8s} {'run p90':>8s} {'runs>50%':>9s}")
    for s in STATUS_ORDER:
        sh = shares.get(s)
        if not sh:
            continue
        print(f"{s:16s} {pct(stat[s], n_att):>8s} {rate(quant(sh, 0.5, 4)):>8s} "
              f"{rate(quant(sh, 0.9, 4)):>8s} {sum(1 for x in sh if x > 0.5):>9d}")


def crop_quality(recs) -> None:
    """Crop placement and miss causes straight off the record's ``cropQuality`` block (#86).

    IoU answers "did the crop frame the Climber", containment answers "did it cover them
    at all" — reported separately because a large but correctly-placed crop scores badly
    on the first and perfectly on the second. Search-condition *flags* live here rather
    than with the stream-derived luma/sharpness because they are the exact evidence the
    ``adverse-conditions`` miss cause was decided from."""

    causes = collections.Counter()
    iou_i = collections.defaultdict(list)
    iou_d = collections.defaultdict(list)
    contained = scored = matched = 0
    flags = collections.defaultdict(collections.Counter)
    flagn = collections.Counter()
    thresholds = {}
    for _, d in recs:
        cq = d.get("cropQuality") or {}
        thresholds = cq.get("thresholds") or thresholds
        matched += cq.get("matchedAttempts") or 0
        for c, n in (cq.get("missCauseCounts") or {}).items():
            causes[c] += n
        for e in cq.get("frames") or []:
            s = e.get("status") or "unknown"
            for bucket, key in ((iou_i, "initialSearchRegionIou"),
                                (iou_d, "detectionRegionIou")):
                if isinstance(e.get(key), (int, float)):
                    bucket[s].append(e[key])
            if e.get("cropContainedTruth") is not None:
                scored += 1
                contained += bool(e["cropContainedTruth"])
            flagn[s] += 1
            for f in e.get("firedSearchFlags") or []:
                flags[s][f] += 1

    print(f"\n== crop quality ({matched} matched attempts) ==")
    print(f"thresholds: {thresholds}")
    all_i = [v for xs in iou_i.values() for v in xs]
    all_d = [v for xs in iou_d.values() for v in xs]
    print(f"initialSearchRegion IoU vs truth bbox: median {med(all_i, 3)} p90 {quant(all_i, 0.9, 3)} (n={len(all_i)})")
    print(f"detectionRegion     IoU vs truth bbox: median {med(all_d, 3)} p90 {quant(all_d, 0.9, 3)} (n={len(all_d)})")
    print(f"crop contained truth bbox: {contained}/{scored} ({pct(contained, scored)})")
    print("initial IoU (median) by status:",
          {s: med(iou_i[s], 3) for s in STATUS_ORDER if iou_i.get(s)})
    n_miss = sum(causes.values())
    print("miss causes:", {k: f"{v} ({v / n_miss:.1%})" for k, v in causes.most_common()}
          if n_miss else "(no missing attempts)")
    for s in STATUS_ORDER:
        if flagn[s] and flags[s]:
            print(f"  {s:16s} search flags (n={flagn[s]}): "
                  f"{ {k: f'{v / flagn[s]:.1%}' for k, v in flags[s].most_common()} }")


def rejection_gates(recs) -> None:
    """Rejection correctness off ``frameQuality.rejectionCorrectness`` (#85).

    The record scores each rejected attempt's raw pose against truth with the same
    geometry gate ``_classify_detection`` uses, plus a joint-agreement floor. The old
    hand-rolled ``centroidDist <= 0.10`` proxy here disagreed with it — centroid distance
    alone passes a pose whose joints scatter — so it is gone.

    Two denominators, both from the record: over every truth-checkable rejection, and
    over the truth-*present* subset. Rejections on Climber-absent frames are correct by
    construction, so the second is the number that judges the gate's geometry."""

    totals = collections.Counter()
    by_status = {s: collections.Counter() for s in REJECTION_STATUSES}
    for _, d in recs:
        rc = d["frameQuality"].get("rejectionCorrectness") or {}
        for src, sink in [(rc, totals)] + [((rc.get("byStatus") or {}).get(s) or {},
                                            by_status[s]) for s in REJECTION_STATUSES]:
            for v in VERDICTS:
                sink[v] += (src.get("verdictCounts") or {}).get(v) or 0
            sink["truthAbsent"] += src.get("truthAbsent") or 0

    print("\n== rejection gates vs truth ==")
    print(f"{'gate':16s} {'rejected':>9s} {'good':>7s} {'bad':>7s} {'unknown':>8s} "
          f"{'over-rej':>9s} {'over-rej(present)':>18s}")
    for label, c in [("all", totals)] + [(s, by_status[s]) for s in REJECTION_STATUSES]:
        good, bad = c["goodPoseRejected"], c["badPoseRejected"]
        checkable = good + bad
        present = checkable - c["truthAbsent"]
        print(f"{label:16s} {good + bad + c['truthUnknown']:>9d} {good:>7d} {bad:>7d} "
              f"{c['truthUnknown']:>8d} {pct(good, checkable):>9s} {pct(good, present):>18s}")


def raw_stream_extras(root: str, runs_prefix: str) -> None:
    """The measures no record carries: which selector picked the pose, whether a
    reacquire worked, the absolute luma/sharpness of the searched region, how long a
    failure state persists, and how often the scanner fell back to the whole frame.

    Everything here needs either the *un*matched attempts or the attempt *ordering*, both
    of which the record drops."""

    selmeth = collections.Counter()
    reacq = collections.Counter()
    cond = collections.defaultdict(lambda: collections.defaultdict(list))
    full_frame = collections.Counter()
    runs_of = collections.defaultdict(list)
    n_runs = n_att = 0
    for p in glob.glob(os.path.join(root, "*", "*", "detections", "*_pose.json")):
        stem = os.path.basename(p).replace("_pose.json", "")
        if runs_prefix and not stem.startswith(runs_prefix):
            continue
        da = (load(p).get("data") or {}).get("detectorAttempts")
        if not da:
            continue
        n_runs += 1
        n_att += len(da)
        cur_s, cur_len = None, 0
        for a in da:
            s = a.get("status") or "unknown"
            selmeth[a.get("selectionMethod")] += 1
            if a.get("reacquireAttempted"):
                reacq["attempted"] += 1
                reacq["succeeded" if a.get("reacquired") else "failed"] += 1
            sc = a.get("searchConditions") or {}
            for region in ("overall", "climber"):
                r = sc.get(region) or {}
                for k in ("mean", "stdDev", "sharpness"):
                    if isinstance(r.get(k), (int, float)):
                        cond[s][f"{region}.{k}"].append(r[k])
            reg = a.get("initialSearchRegion")
            if isinstance(reg, dict) and (reg.get("w") or 0) * (reg.get("h") or 0) >= 0.999:
                full_frame[s] += 1
            if s == cur_s:
                cur_len += 1
            else:
                if cur_s is not None:
                    runs_of[cur_s].append(cur_len)
                cur_s, cur_len = s, 1
        if cur_s is not None:
            runs_of[cur_s].append(cur_len)

    print(f"\n== raw-stream extras ({n_runs} runs, {n_att} attempts, all / not just truth-matched) ==")
    print("selectionMethod:", dict(selmeth.most_common()))
    if reacq["attempted"]:
        print(f"reacquire: {dict(reacq)} | success {pct(reacq['succeeded'], reacq['attempted'])}")
    for s in STATUS_ORDER:
        if s in ("missing", "flipRejected") and runs_of.get(s):
            print(f"{s} run-lengths: med {med(runs_of[s])} max {max(runs_of[s])} "
                  f"(n={len(runs_of[s])})")
    print("full-frame initial regions by status:", dict(full_frame))
    print("search conditions by status (absolute luma / sharpness; the record keeps only "
          "the flags):")
    for s in STATUS_ORDER:
        c = cond.get(s)
        if not c:
            continue
        print(f"  {s:16s} n={len(c['overall.mean'])} "
              f"overall(luma={med(c['overall.mean'])},sharp={med(c['overall.sharpness'])}) "
              f"climber(luma={med(c['climber.mean'])},sharp={med(c['climber.sharpness'])})")


def per_run_table(recs) -> None:
    print("\n== per-run table (sorted by miss%) ==")
    rows = []
    for r, d in recs:
        parts = Path(r).parts
        agr = d.get("agreement") or {}
        pres = agr.get("presence") or {}
        fq = d["frameQuality"]
        st = fq.get("detectorAttemptStatusCounts") or {}
        n = sum(st.values()) or 1
        pck = ((agr.get("aggregate") or {}).get("pck") or {})
        rc = fq.get("rejectionCorrectness") or {}
        cq = d.get("cropQuality") or {}
        pd_, pu = pres.get("presentDetected", 0), pres.get("presentUndetected", 0)
        ad, au = pres.get("absentDetected", 0), pres.get("absentUndetected", 0)
        rows.append((parts[-4], parts[-3][:28],
                     "Y" if (d.get("conformance") or {}).get("conforms") else "N",
                     pct(pck.get("correct") or 0, pck.get("total") or 0),
                     pct(pd_, pd_ + pu), pct(ad, ad + au),
                     st.get("missing", 0) / n, st.get("flipRejected", 0) / n,
                     rate(rc.get("overRejectionRateTruthPresent")),
                     (cq.get("initialSearchRegionIou") or {}).get("median")))
    rows.sort(key=lambda x: -x[6])
    print(f"{'route':22s} {'video':28s} conf {'pck':>6s} {'det':>6s} {'hall':>6s} "
          f"{'miss':>6s} {'flip':>6s} {'orej':>6s} {'iou':>6s}")
    for rt, vd, cf, pk, dt, hl, ms, fl, oj, iou in rows:
        print(f"{rt:22s} {vd:28s} {cf:>4s} {pk:>6s} {dt:>6s} {hl:>6s} {ms:>6.1%} {fl:>6.1%} "
              f"{oj:>6s} {'-' if iou is None else f'{iou:.3f}':>6s}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis_root", nargs="?", default="analysis")
    ap.add_argument("--runs-prefix", default="",
                    help="only read detection runs whose run_ts starts with this")
    args = ap.parse_args()

    recs = attempt_records(args.analysis_root, args.runs_prefix)
    corpus_health(recs)
    attempt_funnel(recs)
    crop_quality(recs)
    rejection_gates(recs)
    raw_stream_extras(args.analysis_root, args.runs_prefix)
    per_run_table(recs)


if __name__ == "__main__":
    main()
