# Blender wave-pool add-ons

This folder ships two complementary, dependency-free add-ons:

- **Surf Pool** (`surf_pool.py`) — a *modern surfing wave pool* (Surf-Ranch /
  Wavegarden style): one clean wave that propagates across the width and **peels
  down the line**, breaking over a shaped reef, with a foil carriage tracking
  along. Procedural traveling wave (no fluid bake). **Use this for surfing.**
- **Wave Pool** (`wave_pool.py`) — a generic recreational wave pool using the
  built-in Ocean modifier (directional swell + foam over a sloped beach).

---

# Surf Pool

A modern surfing wave pool. The surface is a single procedural traveling wave
evaluated each frame from a closed-form function — cheap (no solver/bake), fully
art-directable, and every wave parameter is a live slider.

Geometry & axes: a long, narrow basin with a deep **foil channel** on one side
shoaling up to a **reef shelf** on the other. `X` = wave propagation (width),
`Y` = the line the wave peels along (length). The wave crest is sheared along
`Y` so the break sweeps down the line as it shoals on the reef.

Install via *Install from Disk…* (`surf_pool.py`), enable **Add Mesh: Surf
Pool**, then `N-panel > "Surf Pool" tab > Create / Rebuild Surf Pool` and press
**Play** (or scrub the timeline) to watch it peel — the wave animates from a
`frame_change` handler, so no baking is needed.

| Section | Behaviour |
|---|---|
| **Basin** | line length, width, water level, walls, grid resolution. Edits need **Rebuild**. |
| **Wave (live)** | height, wavelength, speed, **peel rate**, crest sharpness, ambient chop, foam. Edit the wave **instantly**. |
| **Extras** | water colour, foil carriage, sun. |

Dialing it in: keep **Wavelength** near the **Width** for one clean wall; raise
**Crest Sharpness** for a steeper, more barreling face; **Peel Rate** controls
how fast the break runs down the line (0 = closes out everywhere at once).

---

# Wave Pool — Blender add-on

A cost-effective, fully editable wave-pool generator. It builds a complete pool
scene and animates the water with Blender's built-in **Ocean modifier** — no
fluid bake required, so it stays cheap to compute and every wave parameter is a
slider you can scrub in real time.

Target: **Blender 5.x** (also runs on 4.2+; attribute writes are guarded so
minor API differences degrade gracefully).

## Install

1. `Edit > Preferences > Add-ons`
2. The drop-down (▾) `> Install from Disk…`
3. Pick `wave_pool/wave_pool.py`
4. Enable **Add Mesh: Wave Pool**

## Use

`3D Viewport > press N > "Wave Pool" tab > Create / Rebuild Wave Pool`.

What it generates (in a `WavePool` collection):

- **Floor** — flat & deep at the wavemaker end, ramping up into a sloped beach
  so waves visibly run up the shallow end (Solidify modifier for thickness).
- **Walls** — perimeter walls to the rim, with freeboard above the waterline.
- **Water** — a grid plane with an Ocean modifier (DISPLACE) and a glassy,
  transmissive water material; optional foam wired from the Ocean foam layer.
- **Wavemaker** paddle (optional) at the deep end, and a **Sun** (optional, only
  if the scene has no lights).

## Editing

| Section | Behaviour |
|---|---|
| **Pool Dimensions** | length / width / depth / water level / beach / walls / mesh resolution. Changing these needs a **Rebuild** (click the button again). |
| **Waves** | strength, speed, choppiness, directionality, direction, seed, viewport/render detail. These edit the **live** Ocean modifier instantly — no rebuild. |
| **Foam** | toggle + coverage; drives the water material's foam blend. |
| **Extras** | water colour, wavemaker paddle, sun. |

Tips for keeping it cheap:

- Leave **Detail (Viewport)** low (≈7) and raise **Detail (Render)** only for
  final frames. Both are Ocean FFT resolutions (2ⁿ grid), so cost rises fast.
- **Wave Speed** drives a driver on the Ocean `time` value (`frame × speed/fps`),
  so playback animates with zero simulation. Use **Bake Ocean** to cache frames
  for the fastest scrubbing/rendering.
- The Ocean modifier and all objects stay fully editable in the normal modifier
  stack — the panel is a convenience layer, not a lock-in.

## Notes

This add-on was authored against the Blender 5.x Python API and syntax-checked,
but not run inside a built Blender in this environment. The wave/material code
sets modifier attributes and shader sockets defensively (skipping anything a
given version doesn't expose) so it should load cleanly across 4.2–5.x; please
report any version-specific socket/attribute that needs adjusting.
