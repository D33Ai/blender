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

A modern surfing wave pool built as a **kinematic system**: a foil carriage
sweeps down the line at constant velocity and the wave is *generated from that
motion* — each row of water carries a crest launched when the foil passed it,
propagating across the width at the set celerity. So the peel isn't a fudge
factor; it falls out of the kinematics (**peel rate = celerity / foil speed**).

The wave itself is a **Gerstner (trochoidal)** wave — points are displaced
horizontally *and* vertically — so as it shoals over the reef and steepens, the
crest pitches forward and folds into a real **barrel/tube** (a plain height
field can't overhang). Defaults are tuned for a **heavy barrel**: a ~3 m face
that throws over with foam at the lip. Still cheap: it's a closed-form function
evaluated each frame in a `frame_change` handler — no fluid solve, no bake.

Geometry & axes: a long, narrow basin with a deep **foil channel** on one side
shoaling up to a **reef shelf** on the other. `X` = wave propagation (width),
`Y` = the line the wave peels along (length).

Install via *Install from Disk…* (`surf_pool.py`), enable **Add Mesh: Surf
Pool**, then `N-panel > "Surf Pool" tab > Create / Rebuild Surf Pool` and press
**Play** to watch the foil run down the line and the barrel peel behind it
(one ride per cycle).

| Section | Behaviour |
|---|---|
| **Basin** | line length, width, water level, walls, grid resolution. Edits need **Rebuild**. |
| **Wave (live)** | height, wavelength, **wave celerity**, **foil speed**, **steepness**, **reef steepening**, ambient chop, foam. Edit instantly; the panel shows the resulting peel rate. |
| **Extras** | water colour, foil carriage, sun. |

**Presets** (one click — set everything and rebuild):

- **Beginner** — mellow rolling wall over a gradual reef.
- **Performance** — punchy, peeling barrel (the standard heavy-barrel tune).
- **Slab** — a heavy, square slab: flat deep water hits a **sudden ledge**
  (`Reef Abruptness` ≈ 0.9) and jacks into a thick ~3 m barrel that draws below
  sea level over the reef. Think Shipstern / Cyclops.

The reef shape is set by **Reef Position** (where across the width the shelf
rises) and **Reef Abruptness** (gradual point-break reef → sudden slab ledge).

Dialing the barrel: heaviness/throw rises with **Wave Height**, **Steepness**
and **Reef Steepening** and with *shorter* **Wavelength** (the curl needs
`steepness × amplitude × 2π/wavelength > 1`). Lower **Foil Speed** (relative to
celerity) makes the wave stand up taller and peel slower; raise it for a faster,
more drawn-out wall. Use a **higher Width segment count** to resolve the tube.

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
