# Handoff: decoupled ViTPose seed contract — `seed_tap` + `seed_region` (ADR 0006)

**Audience:** an agent working in the **Beta Scanner** repo (the Next.js pose/ORB
app). You do not need the analysis harness repo open — this doc is a *delta* on the
`POST /api/vitpose` seed request you send to the harness to build the ViTPose
scaffold.

**Companion docs:**
[scanner-data-contract.md](scanner-data-contract.md) (the bundle layout and the
`/api/contract` probe you gate on) and
[scanner-calibration-freshness.md](scanner-calibration-freshness.md) (the `setupHash`
staleness rules — unchanged by this delta).

**Harness refs:** ADR 0006 (`docs/adr/0006-decoupled-seed-contract.md`), which amends
the ADR 0003 seed request shape, and issues
[#55](https://github.com/cweber12/beta-scan-analysis/issues/55) (PRD) /
[#56](https://github.com/cweber12/beta-scan-analysis/issues/56) (harness alignment).

---

## What changed and why

ADR 0003 expressed the Climber selection as `climber_point` + `climber_crop`. Those two
fields did double duty: the tap anchored **Climber Identity**, and the **Climber Crop**
was reused as the **seed gate** (the region a candidate track had to fall inside to be
accepted as the seed).

That coupled two independent things. The Climber Crop is a **Video Stats** input — it
bounds the region whose luma/Laplacian stats the harness computes — and is drawn to
frame the body for *condition* measurement. The seed gate only needs to disambiguate
*which* track is the climber, wants to follow the tap, and wants a slightly larger
neighborhood than a tight body crop.

So the harness (ADR 0006) now treats a dedicated **`seed_tap` + `seed_region`** as the
seed contract of record, decoupled from the Climber Crop, and advertises a
`decoupledSeed` capability. `climber_point` / `climber_crop` stay accepted as
backward-compatible **aliases** during migration.

---

## The amended request: `POST /api/vitpose`

```jsonc
{
  "video_path": ".../<route>/<video_key>/<file>.mp4",
  "route_folder": "...",
  "video_key": "...",
  "frames": [ { "timestamp": 12.0 }, ... ],

  "seed_tap":    { "x": 0.51, "y": 0.62, "t": 8.0 },   // NEW — contract of record
  "seed_region": { "x": 0.30, "y": 0.20, "w": 0.40, "h": 0.55 },  // NEW

  "panning": false,
  "setup_hash": "..."
}
```

- **`seed_tap`** `{x, y, t?}` — the normalized tap that anchors Climber Identity. Its
  optional **`t`** anchors candidate selection to the **nearest tapped frame**: set it
  to the timestamp the human actually tapped on, so a later-frame retap seeds the right
  climber when several people cluster near the clip start. camelCase alias `seedTap`
  is also accepted.
- **`seed_region`** `{x, y, w, h}` — the normalized **seed gate**, drawn *for seeding*,
  independent of the Climber Crop. A candidate track passes when its box center falls
  inside `seed_region` (expanded by a fixed pad). camelCase alias `seedRegion` accepted.
- **Legacy aliases:** `climber_point` → `seed_tap`, `climber_crop` → `seed_region`.
  Still accepted. **When both a new field and its legacy alias are present, the new
  field wins** (deterministic precedence). So don't send conflicting pairs — send the
  new fields.

`wall_crop` and any Video Stats `climber_crop` are **unrelated to seeding** and
unchanged — keep sending your genuine Climber Crop to `POST /api/video-stats`; it is not
the seed gate anymore.

### Null-seed fallback (unchanged robustness)

- **`seed_tap` null (or `seed_tap.t` null):** seeding falls back to the prior
  global/full-frame selection — no error. You get the largest/most-central track.
- **`seed_region` null:** the gate is open (any track eligible).

The cross-program **`seedDebug`** block the harness returns is **unchanged**: keys stay
`tap` / `crop` / `mode` / `seedFound`. Your existing debug reader keeps working as-is;
`seedFound: false` still means no track passed the gate for that seed.

---

## What the scanner must do

### 1. Send `seed_tap` + `seed_region` (required, gated on capability)

Probe `GET {HARNESS_API_BASE}/api/contract` at startup and read
`capabilities.decoupledSeed`:

- **`decoupledSeed === true`** → send `seed_tap` + `seed_region`. This is the harness of
  record; prefer the new fields.
- **flag absent / probe fails** → the harness predates this contract. Fall back to
  `climber_point` + `climber_crop` and **surface a visible "harness out of date"
  notice** rather than silently drawing a `seed_region` an old harness will ignore. Do
  not assume the capability — gate on it (same rule as `/api/video-stats`; see
  [scanner-data-contract.md](scanner-data-contract.md)).

`apiVersion` stays `1` — do **not** treat the new fields as an `apiVersion` bump; the
capability flag is the signal.

### 2. Draw `seed_region` separately from the Climber Crop (required)

Give the human a seed gate they draw for *seeding* — loose enough to admit the climber's
track, not the tight body crop used for Video Stats. If your UI only has one crop today,
you may reuse the Climber Crop as `seed_region` (identical to legacy behavior), but the
contract now lets you decouple them, which is the point of ADR 0006.

### 3. Set `seed_tap.t` to the tapped frame's timestamp (recommended)

Anchoring `t` to the frame the human tapped on is what makes a later-frame retap seed
the correct Climber Identity. Leave `t` null only if you genuinely have no frame anchor;
null falls back to global seeding.

### 4. Keep legacy fields working during migration (transition)

Older scanner builds that still send `climber_point` / `climber_crop` keep seeding
exactly as before — the harness aliases them. Don't send a new field **and** a
conflicting legacy field in the same request; if you must send both (e.g. a shared
serializer), keep them equal, since the new field wins.

---

## Deprecation path

The legacy alias path (`climber_point` / `climber_crop` on `POST /api/vitpose`) is a
**transition affordance**, not permanent. Once the scanner fleet reports the new fields
everywhere:

1. Scanner stops sending the legacy aliases (send only `seed_tap` / `seed_region`).
2. A future harness ADR drops the alias resolution and may bump `apiVersion` then.

Until then both are accepted; gate on `decoupledSeed`, prefer the new fields, and keep
the legacy path alive for old clients.

---

## Scope boundary with issue #45 (read this)

This contract is **only** the seed request (`POST /api/vitpose`) — climber anchoring and
seed gating for ViTPose scaffolding. It is **not** related to
[#45](https://github.com/cweber12/beta-scan-analysis/issues/45)
(`detectionAnnotations` ingest), which is a `setupHash`-stamped block the scanner
*writes into the bundle* to refine per-frame detection quality (distractor / failure
class), layered on the Ground Truth review provenance of ADR 0004/0005. They share only
this handoff *style*, not any field. Do not implement one while touching the other:
`seed_tap`/`seed_region` never appear in `detectionAnnotations`, and vice versa.

---

## Don't break (unchanged)

- **`setup_hash`** on the request and **`setupHash`** stamped into `vitpose.json` — the
  evaluate trusted-pairing rules (ADR 0004) are unchanged by this delta.
- **Verbatim timestamps**, full-frame-normalized `[0,1]` coordinates, the 13 COCO core
  joint names, `keypoints: []` for an untracked frame — all the ADR 0003 output
  guarantees still hold.
- The `seedDebug` shape (`tap` / `crop` / `mode` / `seedFound`).

---

## Acceptance

- The scanner probes `/api/contract`, and when `decoupledSeed` is true it sends
  `seed_tap` + `seed_region` (new fields), degrading visibly to the legacy fields
  otherwise.
- `seed_region` is drawn/derivable independently of the Video Stats Climber Crop.
- `seed_tap.t` carries the tapped-frame timestamp when available; a later-frame retap
  seeds the intended climber.
- A null `seed_tap` still produces a scaffold via the global fallback (no error).
- No request sends a new field alongside a conflicting legacy alias.
