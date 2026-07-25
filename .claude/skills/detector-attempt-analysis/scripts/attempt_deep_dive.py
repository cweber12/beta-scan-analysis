"""Detector-attempt corpus deep-dive (companion to the detector-attempt-analysis skill).

Reads schema-v8 evaluation records plus the raw ``detectorAttempts[]`` streams and
prints the standard diagnostic tables:

  1. corpus health   — record counts by evidence type, conformance, held/frozen rates
  2. attempt funnel  — status mix, selection methods, reacquire effectiveness
  3. conditions      — search-condition medians + flag rates by status
  4. crops           — initial-search-region area by status, full-frame counts
  5. rejections      — flipRejected raw-pose-vs-truth centroid split (over-rejection)
  6. per-run table   — conformance, PCK, detect/hallucination/miss/flip rates

Usage:  python .claude/skills/detector-attempt-analysis/scripts/attempt_deep_dive.py \
            [analysis_root] [--runs-prefix 20260724-16]

``--runs-prefix`` filters which detection runs feed sections 2-4 (default: every run
that carries detectorAttempts). Records in section 1/5/6 are always the current
schema-v8 attempt-backed set. Numpy/pandas-free on purpose — stdlib only.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from pathlib import Path


def load(p: str) -> dict:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def med(xs, nd=1):
    xs = sorted(x for x in xs if isinstance(x, (int, float)))
    n = len(xs)
    if not n:
        return None
    v = xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
    return round(v, nd)


def pct(a, b):
    return f"{a / b:.1%}" if b else "-"


def attempt_records(root: str):
    """Current attempt-backed schema-v8 evaluation records: (path, record)."""
    out = []
    for r in glob.glob(os.path.join(root, "*", "*", "evaluations", "*.json")):
        d = load(r)
        if d.get("schemaVersion") != 8:
            continue
        fq = d.get("frameQuality") or {}
        if fq.get("detectorEvidence") == "attempts":
            out.append((r, d))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis_root", nargs="?", default="analysis")
    ap.add_argument("--runs-prefix", default="",
                    help="only mine detection runs whose run_ts starts with this")
    args = ap.parse_args()
    root = args.analysis_root

    # ---- 1. corpus health -----------------------------------------------------
    recs = attempt_records(root)
    print(f"== corpus health ==")
    print(f"attempt-backed schema-v8 records: {len(recs)}")
    conform = sum(1 for _, d in recs if (d.get("conformance") or {}).get("conforms"))
    print(f"conforming (#15 gate): {conform}/{len(recs)}")
    tot = held = froz = 0
    pres = collections.Counter()
    cls = collections.Counter()
    pck_c = pck_t = 0
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
        for jd in (agr.get("perJoint") or {}).values():
            pck_c += jd["pck"]["correct"]
            pck_t += jd["pck"]["total"]
    print(f"detected frames: {tot} | heldPose {pct(held, tot)} | frozenStale {pct(froz, tot)}")
    pd_, pu = pres["presentDetected"], pres["presentUndetected"]
    ad, au = pres["absentDetected"], pres["absentUndetected"]
    print(f"detect rate on present: {pct(pd_, pd_ + pu)} | hallucination on absent: {pct(ad, ad + au)}")
    print(f"pooled PCK@0.5-torso: {pct(pck_c, pck_t)}")
    n_cls = sum(cls.values())
    print("frameQuality classes:",
          {k: f"{v} ({v / n_cls:.1%})" for k, v in cls.most_common()})

    # ---- 2-4. raw attempt streams --------------------------------------------
    stat = collections.Counter()
    selmeth = collections.Counter()
    reacq = collections.Counter()
    cond = collections.defaultdict(lambda: collections.defaultdict(list))
    flags = collections.defaultdict(collections.Counter)
    flagn = collections.Counter()
    area_by = collections.defaultdict(list)
    full_frame = collections.Counter()
    run_missing = []
    miss_runs, flip_runs = [], []
    n_runs = 0
    for p in glob.glob(os.path.join(root, "*", "*", "detections", "*_pose.json")):
        stem = os.path.basename(p).replace("_pose.json", "")
        if args.runs_prefix and not stem.startswith(args.runs_prefix):
            continue
        da = (load(p).get("data") or {}).get("detectorAttempts")
        if not da:
            continue
        n_runs += 1
        n_miss = 0
        cur_s, cur_len = None, 0
        for a in da:
            s = a.get("status") or "unknown"
            stat[s] += 1
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
            f = sc.get("flags") or {}
            if f:
                flagn[s] += 1
                for k, v in f.items():
                    flags[s][k] += bool(v)
            reg = a.get("initialSearchRegion")
            if isinstance(reg, dict):
                ar = (reg.get("w") or 0) * (reg.get("h") or 0)
                area_by[s].append(ar)
                if ar >= 0.999:
                    full_frame[s] += 1
            n_miss += s == "missing"
            if s == cur_s:
                cur_len += 1
            else:
                if cur_s == "missing":
                    miss_runs.append(cur_len)
                if cur_s == "flipRejected":
                    flip_runs.append(cur_len)
                cur_s, cur_len = s, 1
        if cur_s == "missing":
            miss_runs.append(cur_len)
        if cur_s == "flipRejected":
            flip_runs.append(cur_len)
        run_missing.append(n_miss / len(da))

    n_att = sum(stat.values())
    print(f"\n== attempt funnel ({n_runs} runs, {n_att} attempts) ==")
    print("statuses:", {k: f"{v} ({v / n_att:.1%})" for k, v in stat.most_common()})
    print("selectionMethod:", dict(selmeth.most_common()))
    if reacq["attempted"]:
        print(f"reacquire: {dict(reacq)} | success {pct(reacq['succeeded'], reacq['attempted'])}")
    rm = sorted(run_missing)
    if rm:
        print(f"per-run missing: median {rm[len(rm) // 2]:.1%} | p90 {rm[int(0.9 * len(rm))]:.1%} "
              f"| runs>50% missing: {sum(1 for x in rm if x > 0.5)}")
    print(f"missing run-lengths: med {med(miss_runs)} max {max(miss_runs, default=0)} | "
          f"flipRejected run-lengths: med {med(flip_runs)} max {max(flip_runs, default=0)}")

    print("\n== search conditions by status ==")
    for s in ("accepted", "missing", "flipRejected", "qualityRejected"):
        c = cond[s]
        if not c:
            continue
        print(f"  {s:16s} n={len(c['overall.mean'])} "
              f"overall(luma={med(c['overall.mean'])},sharp={med(c['overall.sharpness'])}) "
              f"climber(luma={med(c['climber.mean'])},sharp={med(c['climber.sharpness'])})")
    for s in ("accepted", "missing", "flipRejected"):
        if flagn[s]:
            top = {k: f"{v / flagn[s]:.1%}" for k, v in flags[s].most_common() if v}
            print(f"  {s:16s} flags (n={flagn[s]}): {top}")
    print("initialSearchRegion area (median) by status:",
          {s: med(v, 4) for s, v in area_by.items()})
    print("full-frame initial regions by status:", dict(full_frame))

    # ---- 5. flip-rejection correctness ---------------------------------------
    ok = bad = unk = 0
    for _, d in recs:
        for e in (d["frameQuality"].get("frames") or []):
            if e.get("detectorAttemptStatus") == "flipRejected":
                cd = e.get("centroidDist")
                if cd is None:
                    unk += 1
                elif cd <= 0.10:
                    ok += 1
                else:
                    bad += 1
    print(f"\n== flipRejected vs truth ==")
    print(f"raw centroid <=0.10 of truth (plausibly good pose rejected): {ok} | "
          f">0.10: {bad} | truth-absent/unknown: {unk}")
    if ok + bad:
        print(f"over-rejection share (truth-checkable): {pct(ok, ok + bad)}")

    # ---- 6. per-run table -----------------------------------------------------
    print(f"\n== per-run table (sorted by miss%) ==")
    rows = []
    for r, d in recs:
        parts = Path(r).parts
        agr = d.get("agreement") or {}
        pres = agr.get("presence") or {}
        st = d["frameQuality"].get("detectorAttemptStatusCounts") or {}
        n = sum(st.values()) or 1
        pc = pt = 0
        for jd in (agr.get("perJoint") or {}).values():
            pc += jd["pck"]["correct"]
            pt += jd["pck"]["total"]
        pd_, pu = pres.get("presentDetected", 0), pres.get("presentUndetected", 0)
        ad, au = pres.get("absentDetected", 0), pres.get("absentUndetected", 0)
        rows.append((parts[-4], parts[-3][:28],
                     "Y" if (d.get("conformance") or {}).get("conforms") else "N",
                     pct(pc, pt), pct(pd_, pd_ + pu), pct(ad, ad + au),
                     st.get("missing", 0) / n, st.get("flipRejected", 0) / n))
    rows.sort(key=lambda x: -x[6])
    print(f"{'route':22s} {'video':28s} conf {'pck':>6s} {'det':>6s} {'hall':>6s} {'miss':>6s} {'flip':>6s}")
    for rt, vd, cf, pk, dt, hl, ms, fl in rows:
        print(f"{rt:22s} {vd:28s} {cf:>4s} {pk:>6s} {dt:>6s} {hl:>6s} {ms:>6.1%} {fl:>6.1%}")


if __name__ == "__main__":
    main()
