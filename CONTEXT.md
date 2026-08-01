# Context — analysis harness glossary

The canonical vocabulary for the Beta Scanner **analysis harness**. Use these
terms exactly in issues, code, hypotheses, and reports. This file is a glossary,
not a spec — it carries no implementation detail.

## Core objects

- **Bundle** — a self-contained per-video record at
  `analysis/<route_folder>/<video_key>/`: the video, `final_frame.png`,
  `metadata.json`, `setup.json`, and timestamped detection files. May also carry a
  `vitpose.json` **scaffold** (below) and a `mediapipe.status.json` **MediaPipe job
  status** (below). The unit of ingest.
- **ViTPose scaffold** (`vitpose.json`) — per-frame ViTPose++ Climber keypoints the
  downloader writes to seed beta-scanner's human-authored Ground Truth. A *seed, not
  truth*: the human still corrects and owns it. It is **not** a detection Run — no
  `detections/*_pose.json` is produced. Emitted by `POST /api/vitpose`; see
  `docs/adr/0003`. The seed request is the **decoupled seed contract** (below).
- **Decoupled seed contract** — the `POST /api/vitpose` seed request of record:
  **`seed_tap`** (the tap anchoring **Climber Identity**, with an optional `t` that
  picks the nearest tapped frame) plus **`seed_region`** (the **seed gate** deciding
  which track is the climber). `seed_region` is *decoupled from the Climber Crop* — the
  crop is a Video Stats input, not the gate. The gate is an **overlap** test against
  the expanded seed region, not a test of the candidate box's centre. Legacy
  `climber_point` / `climber_crop` are backward-compatible aliases; the new fields win
  when both are present. Scanners gate on `GET /api/contract` →
  `capabilities.decoupledSeed`. See `docs/adr/0006`.
- **Setup tap** vs **seed tap** — two distinct calibration values, initially equal, and
  never again "the tap". The **setup tap** (`setup.json.climberPoint`) is frozen at
  initial calibration: it seeds MediaPipe and it defines the **climb start**. The
  **seed tap** (`setup.json.seedTap`, sent as `seed_tap`) identifies the Climber for
  the ViTPose scaffold only, is free to move on every re-seed, and propagates its
  correction backwards over the whole trajectory as well as forwards. They were one
  field until issue #101, which is how re-seeding dragged 27 Bundles' setup taps into
  the middle of the climb. Scanners gate on `capabilities.splitTaps`. See
  `docs/adr/0007`.
- **Climb window** — the `[climb start, climb end]` span of a Bundle in which frames
  can be evidence at all. The start comes from the frozen setup tap (or an explicit
  `climbStart`), the end from an explicit `climbEnd` marker — there is no gesture to
  infer a topout from. Either bound may be absent, and an absent window admits every
  frame, so a Bundle with no end marked behaves exactly as it did before the window
  existed. Frames outside it are never tracked, never posed, and never scored: a
  Climber walking away from a finished problem is out of scope, not a detection
  failure.
- **Seed hash** — the identity of everything a ViTPose scaffold is a function of: the
  seed tap, the seed region, the climb window, the video binary, and — when they deviate
  from the declared defaults — the detector settings and the identity of both models.
  Stamped into `vitpose.json` so a scaffold records *which seed it was built from*, and
  compared on every request so unchanged inputs skip the job. This is what makes a
  **stale scaffold** detectable; `setupHash` structurally could not, because it matches
  whether or not a re-seed moved the tap. Model identity is read from the backends that
  actually ran, never from the request, so the hash records what produced the scaffold
  rather than what a caller claimed. One thing it deliberately cannot cover: changing a
  *declared default* itself leaves hashes matching, so such a change must be paired with
  a forced re-seed.
- **Seed failure reason** — why seeding found no Climber, recorded in the status
  sidecar so diagnosis never requires re-running the job: `no-detections` /
  `no-candidates` (the detector found nobody — the video is genuinely hard), or
  `no-frames-in-window` / `region-gated` (candidates existed and the harness refused
  them — repairable by re-tapping or widening the seed region). Only the repairable
  classes are worth spending effort on; the audit behind #101 found only 5 of 15
  truthless Bundles were genuinely hard.
- **Trajectory divergence** — the non-conformance cause meaning the scanner's poses and
  the truth's do not fit each other, on a run with ample accepted detections. It says the
  two disagree; it says **nothing about which of them is wrong**. Named
  `suspected-mistrack` until v15, which asserted the truth side and put eleven sound
  bundles on a truth-repair worklist.
- **Attribution** — which side a divergence is attributable to, and only where there is
  positive evidence: `truth-identity` when the run's truth carries human-attested
  wrong-person frames, `unattributed` otherwise. `unattributed` means *nobody knows* — it
  is never a verdict against the scanner, and it is where a laterality defect lives
  because the review that produces the flags cannot see one. Truth-repair worklists key
  off the attribution, never off the cause.
- **Climber Identity** — which of the people in a frame is *the Climber*, held across the
  whole clip. Anchored by the **seed tap** and propagated both forwards and backwards
  from it. Identity is a separate question from pose quality, and the two fail
  independently: a frame can carry an accurate skeleton on the wrong person. That — not
  joint error — is the dominant truth defect in this corpus, and the only one the human
  review loop is asked to catch.
- **Appearance signature** — the clothing-colour description of a tracked person,
  compared frame to frame to hold **Climber Identity** where position alone is ambiguous.
  It is none of the crops: not the **Climber Crop** (a Video Stats input, decoupled from
  seeding by ADR 0006) and not the scanner's **Adaptive Crop** (a search region). Only
  the signature carries identity. A candidate with no signature scores neutrally, so a
  Bundle where signatures are weak or absent degrades to nearest-box association — which
  is how a trajectory latches onto a bystander at the base of the wall.
- **Ground Truth** (`ground-truth.json`) — beta-scanner's per-frame pose truth
  artifact, authored from the ViTPose scaffold plus human flags. New artifacts carry
  top-level `setupHash` and per-frame `review` provenance. `review: "auto"` is
  agreement-tier evidence; human-flagged frames are excluded from every tier's
  scoring, so **no `review` value is accuracy-tier evidence** — the accuracy tier
  has no attestation source and stays empty.
  Ground Truth stays **pure keypoints**: per-frame metadata derived by the harness
  (the **absence reason**, the climb window) lives on the calibration or in the
  evaluation record, never in the truth artifact. See `docs/adr/0004`, `docs/adr/0005`.
- **Agreement tier** — pose scoring of a Run against unchallenged ViTPose scaffold
  truth. It measures distance from an independently-seeded *scaffold*, never from
  reality: the scaffold is unverified, so agreement is not accuracy.
- **Accuracy tier** — pose scoring against human-attested truth. Structurally present
  and **permanently empty**: no `review` value is a positive attestation. It is kept,
  and reported as explicitly *not computable*, so an unmeasured quantity can never be
  read as a measured-and-poor one.
- **Video Stats** (`video-stats.json` + `metadata.json.video_stats`) — computed
  image-statistic Predictors, two-phased: whole-frame *source stats* stamped into
  `metadata.json` at download/import (never stale), and crop-aware *region stats*
  in `video-stats.json` stamped with the `setupHash` they were computed under
  (stale exactly like Ground Truth when recalibration mints a new hash). Emitted
  by `POST /api/video-stats`; also carries the ViTPose-derived `cameraAngle`
  estimate. Continuous stats are Predictors; the *suggested labels* derived from
  them only prefill the hand labels. The saved `analysisInputs` layer is advisory
  metadata with per-label provenance (auto-accepted vs human-overridden) recorded
  by the scanner; it is not Ground Truth and must not be treated as the authority
  for main detector scoring.
- **Route** — a physical climb, identified by its `route_folder`. Multiple
  **Videos** of the same Route are the norm (different sessions/angles/lighting).
- **Run** — one detection execution on one Video, recorded as a paired
  `<run_ts>_pose.json` + `<run_ts>_orb.json`. **The Run is the unit of
  statistical inference** — coefficients are summarized across Runs, not pooled
  across frames.
- **Experimental Run** — a Run this harness produced by running MediaPipe itself
  (`mediapipe_job.py`), rather than one the scanner posted. Same *shape* as any other
  Run — written through the same writer, read unchanged by evaluation, the conformance
  gate, the tiers and the report — and deliberately different in *provenance*:
  `diagnostics.origin` is `harness-mediapipe`. The two must never pool, because whether
  a browser-WASM run and a Python run agree is the open **parity** question, not an
  assumption. Its ORB half is an explicit `notComputed` artifact: this module has no
  cross-match to compute, and an honest empty beats a fabricated one. See `docs/adr/0012`
  and PRD #156.
- **Arm** — one experimental condition: a detection mode, an ordered list of
  preprocessing steps and their parameters, a crop policy, and the module version,
  hashed into a `configHash` stamped in every Run it produced. **Two Runs differing in
  any factor must not share an arm stamp**, or they pool as one and the experiment
  degrades back into the observational corpus it exists to escape (issue #149's failure
  mode, on the detection side). Repeats within an arm share the stamp and differ only in
  `passIndex` — that is what makes them a variance floor rather than two arms.
- **Cycle** (`analysis/cycles/<cycle_id>.json`) — the comparison group: the set of
  batches whose Arms are meant to be read against each other. Batches are **mode-major**
  (one per mode, run in sequence), so anything that changes between the first batch and
  the last is perfectly confounded with mode; a Cycle is what makes that drift *detected
  exactly* rather than designed around. It is **opened** before the first batch — which
  snapshots every eligible Bundle's truth identity, the model pins, the module version
  and each crop trajectory — and **closed** after the last, which re-verifies them. A
  Bundle whose inputs moved is **excluded from that Cycle's comparison and named**, never
  silently dropped. Only Bundles in `comparableBundles` may be pooled across the Cycle's
  Arms. A tracked artifact, so a published comparison can be audited after the fact.
- **Cycle posture** — what the Arm comparison does with the Cycle it found, and the report
  states which of the four it applied (issue #176). `certified` → **gate**: the pooled Arm
  summary and the deltas are computed over `comparableBundles` only, because those are
  truth-fit numbers and a Bundle whose truth moved mid-Cycle yields a delta that silently
  contains a truth change; the per-Bundle table keeps every Bundle as a **covariate**
  column, so the gate removes rows from the pooled lines and never from the evidence.
  `failed` / `refused` → **refuse**: no pooled comparison is published at all. `close_cycle`
  writes `comparableBundles` even when it fails, so the gate keys on `certified` — keying on
  the list's presence would publish exactly what the artifact forbids. `open` → **in
  flight**: `comparableBundles` does not exist until close, so the comparison renders as
  provisional and never as certified. No Cycle → **label, don't gate**: the whole pre-#168
  corpus, reported in full with an explicit *not drift-checked* marker. The Cycle's criterion
  is an **operational event** (a truth re-seed, a recalibration) rather than a detector
  outcome, which is why it gates more freely than the #15 conformance gate does under #132.
- **Cycle window** — `(openedRunTs, closedRunTs)`, and the only durable join between a Run
  and its Cycle: nothing stamps the `cycleId` into a Run, so a Run is placed by the base
  timestamp in its `exp-<ts>-<arm8>-p<n>` id. A Run outside the window does not pool into
  the Cycle, and is **named** rather than merely subtracted — a timestamp window is a weaker
  join than a stamp, which is why the exclusions have to be readable. A Run predating the
  `exp-` convention is `unplaceable`, which is a different statement from out-of-window: the
  Cycle's own run census cannot see it either. The batch sidecar records the `cycleId` the
  sweep ran inside, for the operator watching a batch in flight.
- **Determinism canary** — one designated Bundle run on one fixed Arm at Cycle open and
  again at Cycle close, with the pose frames compared **byte-for-byte**. The harness
  detector is bit-deterministic, so any difference at all — weights, module, environment,
  crop trajectory — fails the Cycle, which is a strictly more sensitive drift instrument
  than re-running a baseline and comparing metrics, at about two minutes. **The canary
  Arm must crop**: full-frame MediaPipe detects 0% on the canary Bundle, and empty output
  is byte-identical under any weights, so an uncropped canary would certify a model swap
  it never saw. A canary detecting under half its sampled frames therefore **refuses to
  certify** rather than passing.
- **Arm comparison** — reading one Arm against another, **paired on the Bundles both
  ran**. Bundles differ from each other far more than Arms differ on one Bundle (tracked
  crop ranged 59–100% across six), so a difference of pooled means over non-identical
  Bundle sets measures which videos each Arm happened to run rather than the condition;
  two Arms sharing no Bundle yield *no comparison*, which is reported as a named gap
  rather than a number. The reference is the **baseline Arm**: the one applying the fewest
  preprocessing steps, then the one on the most Bundles. Agreement PCK is the primary
  outcome — its *absolute* stays uninterpretable under ADR 0010, but both Arms score
  against the same fixed truth, so truth error is common-mode and cancels in the
  *difference*. Emitted per-Bundle first and pooled only afterwards. See issue #164.
- **Sampling error** — the uncertainty a harness Arm's PCK actually carries, from scoring
  a `12·√n` sample of the Bundle's truth grid instead of the full grid: median 0.0017 /
  p90 0.0056 |ΔPCK| across 55 Bundles. **Common-mode across Arms** — the sample is a
  deterministic function of the Bundle, so every Arm scores the same frames against the
  same truth — which is why it is attached to per-video *absolutes* and discounted in Arm
  *deltas*. Not a run-to-run floor: the harness detector's run-to-run scatter is exactly
  **0** (bit-deterministic, #159, re-verified by the Determinism canary). Recorded in
  `analysis_pipeline/floors.py`.
- **Noise floor** — how much a metric moves when *nothing* changes: the within-Bundle
  range across genuine repeat Runs. #134's figures are **scanner-side** and must be
  labelled as such wherever displayed — attaching the scanner's 0.0055 PCK scatter to a
  harness Arm is a category error, two producers confused for one. They are also
  **provisional**: the historical corpus held six genuine repeat groups (27 of 33 apparent
  ones were a single detection pass re-exported), which is below what a p90 needs to mean
  anything, and the repeat set does not survive the corpus reset — so the measurement is
  frozen in `floors.py` with its caveats, with `scripts/measure_variance_floor.py` kept as
  the derivation.
- **Frame-set integrity** — the check that every harness Run scored the frame set its
  Bundle *prescribes* (`12·√n` of its truth grid), reported per Run in the Arm section.
  The Arm identity deliberately does **not** name the frame set — that omission is what
  makes Sampling error common-mode and cancel in an Arm delta — so an off-rule Run carries
  a stamp indistinguishable from an on-rule one, and any delta computed across the mismatch
  is partly a frame-set artifact. Checked against each Run's stamped `frameCount`, so it
  needs no second Run to disagree with: this is the half the repeat-integrity check (#164)
  cannot reach, which only fires when a sampled Run and a full-grid Run collide on one
  (Arm, Bundle). A
  Run predating the `frameCount` stamp reads as **unknown**, never as mismatched. See
  issue #178.
- **MediaPipe job status** (`mediapipe.status.json`) — the sidecar recording an
  experimental batch's `running` → `done` / `error`, on the `vitpose.status.json` model.
  A failure carries the exception type and traceback, and a batch that dies part-way
  still names the Runs that reached disk, so a partial batch is never mistaken for a
  complete one.
- **Detector Attempt** — one scanner-owned MediaPipe attempt on the sampled
  100 ms analysis timeline: the initial search region, whether full-frame
  reacquire ran, the selected raw Climber pose when MediaPipe returned one, the
  accepted pose when the scanner kept it, the rejection/missing status, compact
  candidate-selection metadata, and scanner-computed pixel conditions for the
  searched region. Detector Attempts are evidence, not recommendations; the
  harness joins them to Ground Truth to derive Detection Errors.
- **Attempt funnel** — how a Run's Detector Attempts split across `accepted` /
  `missing` / `flipRejected` / `qualityRejected` (plus `unknown` for a status
  outside the vocabulary), read as a funnel: what the detector kept, then the
  three ways it didn't. Scanner behavior only — no Ground Truth is consulted, so
  it is a description of what the detector *did*, never of whether it was right.
  Every pooled share is reported beside its run-unit distribution (median, p90,
  and **tail runs** — runs where one status took more than half the attempts),
  because the Run is the unit of inference and one very long collapsed Run moves
  a pooled share as much as a dozen ordinary ones. No confidence intervals:
  attempts within a Run are correlated.
- **Rejection Verdict** — the harness's judgement of one *rejected* Detector
  Attempt: was the scanner's flip/quality gate right to discard that raw pose?
  Only the harness can answer it, because only the harness holds Ground Truth.
  One of `goodPoseRejected` (the discarded pose agreed with truth — the gate
  over-rejected), `badPoseRejected` (the pose diverged from truth, or landed on a
  Climber-absent frame where no pose belongs), or `truthUnknown` (no raw pose, or
  no usable truth geometry to check against).
- **Truth bbox** — the padded extent of a truth frame's core joints, computed
  backend-side. The padding exists because 13 joints do not span a Climber's
  silhouette (no crown of the head, no hands past the wrists, no feet past the
  ankles). It is what the scanner's Adaptive Crop *should* have covered, so it is
  the reference for every crop-quality measure.
- **Crop containment** vs **crop IoU** — two different questions about one
  Adaptive Crop, deliberately not merged. Containment is the share of the Truth
  bbox inside the searched region ("did we look where the Climber was"); IoU also
  penalises a region far larger than the Climber ("did we look *tightly*"). A
  correctly-placed but oversized crop scores perfectly on the first and poorly on
  the second, so reporting only IoU would read crop *size* as crop *error*.
- **Miss Cause** — why one missing Detector Attempt found no Climber:
  `climber-absent` (truth says nobody is there — a correct miss),
  `crop-misplaced` (the crop excluded the Climber *and* was the only place
  searched), `adverse-conditions` (everything was searched and condition flags
  fired), `unexplained` (everything searched, conditions clean, still lost).
  `crop-misplaced` is a causal claim, so it requires that no full-frame reacquire
  ran: when the scanner also searched the whole frame and still failed, the crop
  cannot be what lost the Climber, however badly placed it was. Crop placement is
  measured on every miss regardless — the two facts are reported side by side and
  neither may be inferred from the other.
- **Over-rejection rate** — the share of *truth-checkable* rejections whose
  verdict is `goodPoseRejected`. Reported per Run so scanner flip-gate changes are
  measurable batch-over-batch, and under two denominators: over all checkable
  rejections, and over Climber-present ones only. Climber-absent rejections are
  correct by construction, so including them measures how much of the scanner's
  rejecting is aimed at empty frames rather than how well the gate judges a pose.
- **Absence reason** — why one truth frame is *absent*, derived by the harness from
  evidence already on disk and never authored into Ground Truth: `out-of-scope`
  (outside the **climb window**), `not-sampled` (the ViTPose scaffold's grid never
  reached it), `untracked` (the scaffold's tracker lost or never acquired the
  Climber), `confirmed-absent` (the residual — the Climber really is not there), or
  `unknown` (nothing to derive from). **Only `confirmed-absent` enters the presence
  2×2 and the hallucination split**; the rest are counted and held out, never dropped
  and never promoted. One label used to flatten all four, and the difference between
  them is the difference between four different fixes — only `confirmed-absent`
  implies presence gating. See `docs/adr/0008`.
- **Scaffold drift** — Ground Truth is authored *from* the ViTPose scaffold, so the
  two should record roughly the same Climber-present frames. When the scaffold is
  regenerated, the truth on disk keeps describing the superseded one, and every frame
  the new scaffold poses that the old truth calls absent becomes a **phantom absence**.
  Nothing else detects it: `setupHash` tracks *calibration*, and a re-seed does not
  change the calibration, so a drifted truth still pairs as current on both sides —
  the same blind spot ADR 0007 closed for scaffolds, one layer up. Measured by
  comparing present-frame counts, which is a heuristic; the durable fix is for Ground
  Truth to stamp the scaffold `seedHash` it was authored from.
- **Truth sufficiency** — the conformance gate's floor on truth-present **frames**,
  distinct from its floor on joint-*pairs*. A bundle whose near-perfect fit rests on
  eleven frames is not a conforming bundle, and counting joint-pairs let exactly that
  pass. Measured in the unit the gate is trying to measure.
- **Non-conformance cause** — why a bundle failed the conformance gate (the
  near-identity fit of scanner coordinates onto Ground Truth that quarantines a
  bundle from pooled metrics), which the gate's own pass/fail verdict cannot
  say: `rate-mismatch` (the scaffold sampled far coarser than the truth grid, so
  most truth frames were never looked at — a *data* defect, fixed by regenerating
  the scaffold), `sparse-match` (the detector supplied too little
  to fit — too few matched-present frames, or too small a share of present
  Detector Attempts accepted) or `suspected-mistrack` (ample accepted detections
  and the fit still misses identity — the appearance-stitch signature the gate was
  built for). The verdict is unchanged by the split; the cause only routes the
  record. Only `suspected-mistrack` reaches the **truth-repair worklist**,
  because re-seeding Ground Truth for a run whose detector found almost nothing
  repairs nothing — and `rate-mismatch` must not be read as either of the others,
  because neither worklist can fix it.
- **Evidence generation** — which detector evidence an evaluation record was
  scored from: `attempts` (the canonical **Detector Attempt** stream) or
  `legacy-frames` (the dense playback `frames[]` fallback used before the scanner
  exported attempts). Records written before the marker existed read as
  `unknown` — not attempt-backed, and never claimed as such. When one
  **video + truth revision** pairing carries both generations, only the
  attempt-backed record pools: the pairing is counted once and the two
  generations never blend (the legacy record's appVersion differs, so a
  generation change would otherwise read as a scanner change). **Superseded**
  records are held out of aggregation only — they stay on disk, readable, and are
  listed by name in the report. Every pooled report section states the
  generation(s) it aggregates, because a pool can still legitimately span
  generations across *different* videos.
- **Build identity** — which scanner code produced a run, and the unit every
  cross-batch comparison groups by. Two fields, both from the pose diagnostics:
  `appVersion` is the commit the dev server was *started* from, and
  `detectorCodeHash` is a digest of the detector source that actually executed.
  They differ because `NEXT_PUBLIC_APP_VERSION` resolves once at server start, so
  a hot reload moves the code and not the stamp — the defect that left the
  07-25/26 batch stamped `c305954` while running a later build. Grouping is
  **hash-first**: the `detectorCodeHash` where one exists, falling back to
  `appVersion` where it doesn't. That way two commits sharing a hash are one
  group (a commit that did not touch detection — pooling them *increases* usable
  n), while one stamp covering two hashes splits into the groups it really is.
  A **build-identity conflict** is that second case, and is reported over *every*
  pose run on disk rather than the scored subset — the corpus's only conflict
  sits entirely in runs no evaluation record scored. A missing hash is *unknown
  provenance*, never a conflict, and never merged with a hashed group: it might
  be the same code, but nothing on disk says so.
- **Measurement basis** — what a pooled number rests on, and therefore what it can be
  compared against: the **schema version(s)** the records were scored under (what was
  counted) plus the **build identity** set they were collected from (what was measured).
  The basis is *frozen* for one baseline cycle — collect → score → analyse → act —
  because the schema moved v8 → v11 → v12 → v13 → v14 in about two weeks and no two
  baselines were ever scored on the same one; the "88% no-candidates" miss split survived
  four of them before turning out to be a pooling artifact. `BASELINE_CYCLE_SCHEMA`
  declares the frozen basis, and `SCHEMA_VERSION` moving away from it is a **mid-cycle
  bump**: permitted, never silent, and it demands re-scoring the *whole* compared
  population (`evaluate --mode all`) rather than just the new batch. Every pooled section
  states its own basis, for the same reason it states its **evidence generation** — a
  number read out of the middle of the report carries its own provenance. A pool spanning
  versions is **flagged, not refused** (a corpus mid-migration legitimately spans bases),
  and a record that does not stamp one reads as `unknown`, never as the frozen version.
  See `docs/adr/0009`.

## The condition → detection vocabulary

- **Predictor** — a *condition* of the video that might drive detection quality:
  a computed image stat (reference/per-frame luma mean, stdDev, Laplacian
  **sharpness**), motion magnitude, climber coverage, or a **hand label**
  (route orientation, camera angle, occlusion, camera stability, …). Hand labels
  are written by the scanner at calibration into `setup.json.analysisInputs`
  (snake_case keys matching `runs.LABEL_KEYS`) and are advisory context only.
  They may stratify reports or preserve human notes, but computed pixel
  conditions and Detector Attempts are the primary analysis predictors. The
  harness upload no longer collects these labels.
- **Outcome** — a measure of *how good detection actually was*. The trusted pose
  Outcome is **`overlayQuality`** (the scanner's end-to-end 0..1 verdict) plus
  **`badStretches`** (spans the overlay was visibly wrong). The ORB Outcome is
  **cross-match separation** (below). An Outcome is validated against human
  judgment, never assumed.
- **Symptom** — a self-reported detector *reaction* that is often mistaken for an
  Outcome but is partly circular: `detectionRate`, `flipRate`, `confidence`,
  `gapsRefined`, `limbExpandedFrames`. Symptoms are Predictors of interest, not
  ground truth. (A high `flipRate` is the flip detector firing hard, not proof the
  pose was wrong.)
- **Proxy** — the per-frame `kp_count` / `mean_score` derived from *exported*
  frames. Because exported frames are already interpolated / gap-filled /
  smoothed, the Proxy is **not raw detector output**. Distinguished from
  raw-detect success, which comes from per-frame **provenance**
  (`source: raw | interpolated | filled | flipDiscarded | limbExpanded`).
- **Detection Error** — a per-frame discrepancy derived by the harness after
  joining scanner evidence to Ground Truth: missing Climber, hallucinated pose on
  an absent frame, wrong subject, flip/rotation, distortion, drift, or stale raw
  detector output. Causes are discovered by correlating Detection Errors against
  Detector Attempts, crops, and computed pixel conditions, not by accepting
  hand-authored frame metadata as truth.
- **Condition Band** — a quantile slice of one Predictor (terciles by default),
  reported with the failure/flagged rate of the frames that fall in it. The rate
  pools frames, but the **confidence interval is computed at the Run unit**: a
  band's frames come from a few dozen Runs at most and are heavily correlated
  within each, so the bootstrap resamples Runs and the per-Run median/p90 is
  reported beside the pooled rate. A band with many frames and few Runs is weak
  evidence, and its interval says so.

## ORB cross-match

- **Cross-match** — matching one Video's wall-crop features (train) against
  another Video's `final_frame.png` (query), over all ordered pairs.
- **Cross-match ground truth** — a pair is a **same-route** positive iff the two
  Videos share `route_folder`, else a **cross-route** negative.
- **Route-ID separation** — the gap between the same-route and cross-route
  inlier-ratio distributions. Wide separation = ORB robustly identifies a wall
  under real condition variation; narrow = it doesn't. The headline ORB Outcome.
