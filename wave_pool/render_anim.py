"""Headless animation of the Surf Pool 'Slab' barrel peeling down the line.

    SURF_CLIP=TRACK blender --background --factory-startup --python render_anim.py
    SURF_CLIP=POV   blender --background --factory-startup --python render_anim.py

TRACK = external camera trailing the peel; POV = inside the barrel looking out.
Renders one clip per run (kept cheap so it fits a modest CPU budget) to an MP4
in ./surf_render/. The wave animates via the add-on's frame_change handler.
"""
import importlib.util
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("render_slab", os.path.join(HERE, "render_slab.py"))
rs = importlib.util.module_from_spec(spec)
sys.modules["render_slab"] = rs
spec.loader.exec_module(rs)          # importable: render_slab.main() is __main__-guarded

CLIP = os.environ.get("SURF_CLIP", "TRACK").upper()
OUT = rs.OUT
FPS = 24
F0, F1 = 200, 259                    # ~2.5 s; foil sweeps the mid-pool section


def keyframe_camera(p, loc_fn, tgt_fn, lens):
    empty = bpy.data.objects.new("CamTarget", None)
    bpy.context.scene.collection.objects.link(empty)
    cd = bpy.data.cameras.new("Cam"); cd.lens = lens
    cam = bpy.data.objects.new("Cam", cd)
    bpy.context.scene.collection.objects.link(cam)
    con = cam.constraints.new("TRACK_TO")
    con.target = empty
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    bpy.context.scene.camera = cam
    # linear motion to stay in sync with the constant-velocity foil (set on the
    # preference so we avoid the 5.x layered-action fcurve API entirely)
    try:
        bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
    except Exception:
        pass
    for fr in (F0, F1):
        cam.location = loc_fn(fr); cam.keyframe_insert("location", frame=fr)
        empty.location = tgt_fn(fr); empty.keyframe_insert("location", frame=fr)


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mod = rs.load_addon()
    p = rs.build_slab(mod, frame=F0)
    rs.water_material()
    rs.style_objects()
    rs.add_sky_and_sun()
    L, W, wl = p.pool_length, p.pool_width, p.water_level

    def yf(fr):                       # kinematic foil position at this frame
        return -L / 2.0 + p.foil_speed * (fr / FPS)

    if CLIP == "POV":
        loc_fn = lambda fr: (W * 0.30, yf(fr) - 9.0, wl + 0.7)
        tgt_fn = lambda fr: (W * 0.20, yf(fr) + 7.0, wl + 0.5)
        lens = 28.0
    else:  # TRACK
        loc_fn = lambda fr: (-W * 0.50, yf(fr) - 14.0, wl + 1.7)
        tgt_fn = lambda fr: (W * 0.18, yf(fr) - 6.0, wl + 1.1)
        lens = 38.0
    keyframe_camera(p, loc_fn, tgt_fn, lens)

    rs.setup_render(samples=40, res=(960, 540), exposure=-1.1)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = F0, F1
    scene.render.fps = FPS
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    try:
        scene.render.ffmpeg.constant_rate_factor = "HIGH"
    except Exception:
        pass
    scene.render.filepath = os.path.join(OUT, f"slab_{CLIP.lower()}.mp4")
    bpy.ops.render.render(animation=True)
    print("WROTE", scene.render.filepath)


main()
