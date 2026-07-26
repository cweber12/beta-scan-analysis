# Handoff: split the setup tap from the ViTPose seed tap (ADR 0007)

**Audience:** an agent working in the **Beta Scanner** repo (the Next.js pose/ORB app).
You do not need the analysis harness repo open — this is a *delta* on the calibration
you write into `setup.json` and on the `POST /api/vitpose` request you send.

**Companion docs:**
[scanner-seed-contract-adr0006.md](scanner-seed-contract-adr0006.md) (the
`seed_tap` / `seed_region` split this amends),
[scanner-data-contract.md](scanner-data-contract.md) (bundle layout and the
`/api/contract` probe you gate on), and
[scanner-calibration-freshness.md](scanner-calibration-freshness.md) (`setupHash`
staleness — unchanged by this delta).

**Harness refs:** ADR 0007 (`docs/adr/0007-setup-tap-and-seed-tap-are-separate.md`) and
issue [#101](https://github.com/cweber12/beta-scan-analysis/issues/101).

---

## The defect this fixes, in one paragraph

`setup.json.climberPoint` is currently **both** the setup tap (the calibration gesture
that anchors MediaPipe's Climber selection) **and** the ViTPose seed tap (the point
that tells the scaffold which tracked person is the Climber). They start equal — they
were the same gesture — but re-seeding writes back over `climberPoint`, so every
re-tap drags the setup tap forward. Measured across the corpus: **27 Bundles now have
a setup tap sitting mid-climb**, and in 24 of them the Climber's hips had already
risen 5–47% of frame height before the tap. Nothing detects it, because the calibration
hash matches either way. That is also why the harness could not adopt the setup tap as
the climb start: doing it today would discard real climbing in 24 Bundles, up to 28
seconds in the worst case.

---

## What the scanner must change

### 1. Stop writing re-seed taps into `climberPoint`

`setup.json.climberPoint` is the **setup tap**. Write it once, at initial calibration.
**A re-seed must never touch it.**

### 2. Write the re-seed tap to `setup.json.seedTap`

```jsonc
{
  "setupHash": "...",
  "climberPoint": { "x": 0.51, "y": 0.88, "t": 3.5 },   // setup tap — FROZEN
  "seedTap":     { "x": 0.47, "y": 0.22, "t": 41.0 },   // seed tap — moves freely
  "climbEnd": 58.0,                                      // NEW, see §4
  // ...unchanged: climberCrop, wallCrop, seedRegion, panning, analysisInputs
}
```

On a Bundle that has never been re-seeded, write `seedTap` equal to `climberPoint` (or
omit it — the harness falls back to `climberPoint` and behaves exactly as today).

### 3. Send the re-seed tap as `seed_tap` on `POST /api/vitpose`

Unchanged from ADR 0006 in shape — what changes is *which* value goes in it. Send the
current **seed tap**, not the setup tap:

```jsonc
{
  "video_path": ".../<route>/<video_key>/<file>.mp4",
  "route_folder": "...", "video_key": "...",
  "frames": [ { "timestamp": 12.0 }, ... ],

  "seed_tap":    { "x": 0.47, "y": 0.22, "t": 41.0 },   // the SEED tap
  "seed_region": { "x": 0.30, "y": 0.20, "w": 0.40, "h": 0.55 },

  "climb_start": 3.5,     // NEW — the SETUP tap's t (see §4)
  "climb_end":   58.0,    // NEW — the end-of-climb marker
  "force":       false,   // NEW — see §5

  "panning": false, "setup_hash": "..."
}
```

Both climb fields also accept camelCase (`climbStart` / `climbEnd`). Both are optional.
If you omit them, the harness reads them from the bundle's `setup.json`, so writing
the calibration correctly (§2) is enough on its own — sending them is the explicit path
for a job that wants to override.

**Validation:** `climb_end` must be greater than `climb_start`, and both must be ≥ 0.
A violation is a 422 from the endpoint, not a silently ignored field.

### 4. Capture an end-of-climb marker (the UX work)

This is the one piece that needs scanner UI. The harness needs a timestamp for **where
the climb ends** — a topout, or the point where the attempt is over. There is no
gesture to infer it from, so it must be captured explicitly and written to
`setup.json.climbEnd`.

The climb **start** needs no new gesture: it is the setup tap's `t`, which the human
already gave you at calibration.

Until a Bundle has `climbEnd`, the harness treats the window as open on that side and
behaves exactly as it does today. So this can ship after the rest.

### 5. Expect a new response: the unchanged-seed skip

`POST /api/vitpose` now stamps a **seed hash** into `vitpose.json` covering the seed
tap, seed region, climb window and video binary. When a request's seed hash matches the
scaffold already on disk, the harness **skips the job** and says so **synchronously**:

```jsonc
// 200 OK  (instead of the usual 202 + poll)
{
  "status": "skipped",
  "reason": "unchanged-seed",
  "seedHash": "a1b2c3d4e5f60718",
  "artifactPath": "analysis/<route>/<video_key>/vitpose.json"
}
```

Treat a 200 `skipped` as success with the artifact already present — do **not** poll
for a job that will never write a status sidecar. A 202 still means "running, poll the
sidecar" exactly as before. Send `"force": true` to re-run regardless.

This is also a correctness fix on your side: before it, a re-calibration that moved the
seed left a **stale scaffold** behind and nothing could tell, because `setupHash`
matched either way. `seedHash` is the signal.

### 6. Gate on the capability flag

`GET /api/contract` now advertises:

```jsonc
"capabilities": { "decoupledSeed": true, "splitTaps": true }
```

`apiVersion` stays `1` — this is additive. Gate the new fields on `splitTaps` so a
mixed-version deployment degrades visibly rather than silently writing a `seedTap` an
old harness ignores.

---

## What the harness changed on its side

For context — no action needed from the scanner:

- **The seed-region gate is now an overlap test**, not a test of the candidate box's
  centre. Measured on the worst Bundle, the centre rule rejected all 182 candidates in
  the seed window on a video where a person is visible in 98% of sampled frames.
- **Detections with no tracker id are kept for seeding.** ByteTrack assigns no ids on
  some frames and the adapter used to discard every detection on such a frame; on the
  worst Bundle that threw away 82 raw detections across the seed window.
- **Seeding failures now record a reason** in `vitpose.status.json`
  (`seedDebug.seedFailureReason`): `no-detections` / `no-candidates` mean the video is
  genuinely hard, `no-frames-in-window` / `region-gated` mean the harness refused
  candidates that existed and the Bundle is repairable.
- **The climb window bounds tracking and posing**, so out-of-climb footage costs no GPU
  time and produces no scaffold frames.

Neither seeding change is permitted to manufacture truth: a Bundle with no detectable
Climber still fails, and the 5 genuinely-hard Bundles are the regression fixture for
that.

---

## Migration note

**Existing Bundles are not migrated, deliberately.** No migration can recover the
intent of a tap that was already overwritten — the original setup tap is gone. That is
why issue #101 ends in a corpus reset: every Bundle is regenerated under the new
contracts rather than repaired under the old ones. Once the scanner adopts this delta,
re-calibrating a Bundle writes a correct setup tap for the first time.
