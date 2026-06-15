"""Headless high-quality render of the Surf Pool 'Slab' preset (Cycles).

Run from this folder with a portable/installed Blender 5.x:

    blender --background --factory-startup --python render_slab.py

Writes stills to ./surf_render/ (override with the SURF_OUT env var). The
helper functions are importable (e.g. by render_anim.py) — main() only runs
when this file is executed directly.
"""
import importlib.util
import math
import os
import sys

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("SURF_OUT", os.path.join(os.getcwd(), "surf_render"))
os.makedirs(OUT, exist_ok=True)


def load_addon():
    spec = importlib.util.spec_from_file_location("surf_pool", os.path.join(HERE, "surf_pool.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["surf_pool"] = mod
    spec.loader.exec_module(mod)
    mod.register()
    return mod


def build_slab(mod, frame=240):
    scene = bpy.context.scene
    p = scene.surf_pool
    for k, v in mod.PRESETS["SLAB"].items():
        setattr(p, k, v)
    p.add_lighting = False
    p.add_caissons = True
    p.foam_amount = 0.45          # confine whitewater to the lip, keep a blue body
    mod._clear()
    coll = mod._ensure_collection(bpy.context)
    mod._build_floor(p, coll)
    mod._build_walls(p, coll)
    mod._build_water(p, coll)
    mod._build_caissons(p, coll)
    scene.frame_set(frame)
    mod.compute_surface(scene)
    return p


def water_material():
    """Deep, partly-transmissive water with a punchy white foam breakout."""
    mat = bpy.data.materials.get("SurfPool_Water") or bpy.data.materials.new("SurfPool_Water")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    base = (0.0, 0.13, 0.20, 1.0)
    bsdf.inputs["Base Color"].default_value = base
    bsdf.inputs["Roughness"].default_value = 0.04
    if "IOR" in bsdf.inputs:
        bsdf.inputs["IOR"].default_value = 1.333
    for t in ("Transmission Weight", "Transmission"):
        if t in bsdf.inputs:
            bsdf.inputs[t].default_value = 0.55
            break

    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "foam"
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.50
    ramp.color_ramp.elements[1].position = 0.82
    nt.links.new(attr.outputs["Fac"], ramp.inputs["Fac"])

    mix_c = nt.nodes.new("ShaderNodeMix"); mix_c.data_type = "RGBA"
    mix_c.inputs["A"].default_value = base
    mix_c.inputs["B"].default_value = (1.0, 1.0, 1.0, 1.0)
    nt.links.new(ramp.outputs["Color"], mix_c.inputs["Factor"])
    nt.links.new(mix_c.outputs["Result"], bsdf.inputs["Base Color"])

    mix_r = nt.nodes.new("ShaderNodeMix"); mix_r.data_type = "FLOAT"
    mix_r.inputs["A"].default_value = 0.04
    mix_r.inputs["B"].default_value = 0.65
    nt.links.new(ramp.outputs["Color"], mix_r.inputs["Factor"])
    nt.links.new(mix_r.outputs["Result"], bsdf.inputs["Roughness"])

    for t in ("Transmission Weight", "Transmission"):
        if t in bsdf.inputs:
            mix_t = nt.nodes.new("ShaderNodeMix"); mix_t.data_type = "FLOAT"
            mix_t.inputs["A"].default_value = 0.55
            mix_t.inputs["B"].default_value = 0.05
            nt.links.new(ramp.outputs["Color"], mix_t.inputs["Factor"])
            nt.links.new(mix_t.outputs["Result"], bsdf.inputs[t])
            break

    w = bpy.data.objects.get("SurfPool_Water")
    if w is not None:
        w.data.materials.clear()
        w.data.materials.append(mat)


def style_objects():
    floor = bpy.data.objects.get("SurfPool_Floor")
    if floor is not None:
        m = bpy.data.materials.new("Reef"); m.use_nodes = True
        b = m.node_tree.nodes.get("Principled BSDF")
        if b:
            b.inputs["Base Color"].default_value = (0.18, 0.30, 0.32, 1.0)
            b.inputs["Roughness"].default_value = 0.85
        floor.data.materials.append(m)
    walls = bpy.data.objects.get("SurfPool_Walls")
    if walls is not None:
        m = bpy.data.materials.new("Concrete"); m.use_nodes = True
        b = m.node_tree.nodes.get("Principled BSDF")
        if b:
            b.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1.0)
            b.inputs["Roughness"].default_value = 0.85
        walls.data.materials.append(m)
    # caissons: one shared dark material
    cmat = bpy.data.materials.new("Caisson"); cmat.use_nodes = True
    cb = cmat.node_tree.nodes.get("Principled BSDF")
    if cb:
        cb.inputs["Base Color"].default_value = (0.04, 0.04, 0.05, 1.0)
        cb.inputs["Roughness"].default_value = 0.7
    for o in bpy.data.objects:
        if o.name.startswith("SurfPool_Caisson"):
            o.data.materials.append(cmat)


def add_sky_and_sun():
    scene = bpy.context.scene
    world = bpy.data.worlds.new("Sky"); world.use_nodes = True
    nt = world.node_tree; nt.nodes.clear()
    bg = nt.nodes.new("ShaderNodeBackground")
    out = nt.nodes.new("ShaderNodeOutputWorld")
    sky = nt.nodes.new("ShaderNodeTexSky")
    try:
        sky.sky_type = "NISHITA"
        sky.sun_elevation = math.radians(12.0)
        sky.sun_rotation = math.radians(80.0)
    except Exception:
        pass
    nt.links.new(sky.outputs[0], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.4
    nt.links.new(bg.outputs[0], out.inputs["Surface"])
    scene.world = world

    light = bpy.data.lights.new("Sun", "SUN")
    light.energy = 3.5
    light.angle = math.radians(1.5)
    sun = bpy.data.objects.new("Sun", light)
    sun.rotation_euler = (math.radians(62), math.radians(4), math.radians(125))
    scene.collection.objects.link(sun)


def add_camera(loc, target, lens=35.0):
    cd = bpy.data.cameras.new("Cam"); cd.lens = lens
    cam = bpy.data.objects.new("Cam", cd)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = Vector(loc)
    cam.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    return cam


def setup_render(samples=180, res=(1920, 1080), exposure=-1.1):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    try:
        scene.cycles.device = "CPU"
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
    except Exception:
        pass
    try:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.exposure = exposure
        scene.view_settings.look = "AgX - High Contrast"
    except Exception:
        pass
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = 100


def render_still(path, **kw):
    setup_render(**kw)
    scene = bpy.context.scene
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print("WROTE", path)


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    frame = 200
    mod = load_addon()
    p = build_slab(mod, frame=frame)
    water_material()
    style_objects()
    add_sky_and_sun()
    L, W, wl = p.pool_length, p.pool_width, p.water_level

    # find where the wave is breaking on the reef at this frame: the caisson whose
    # pulse has just travelled the full width (c*age == W) is breaking now.
    xi, yi, _ = mod.caisson_positions(p)
    tc = (frame / 24.0) % p.firing_period
    i_break = int(min(max((tc - W / p.wave_celerity) / max(p.firing_delay, 1e-4), 0), len(yi) - 1))
    yb = float(yi[i_break])

    # Hero: low in the channel, facing the breaking section across the reef
    add_camera(loc=(-W * 0.48, yb - 4.0, wl + 1.6),
               target=(W * 0.22, yb + 1.0, wl + 1.1), lens=40.0)
    render_still(os.path.join(OUT, "slab_hero.png"))

    # Scenic: elevated 3/4 down the line showing the caisson array + the peel
    add_camera(loc=(-W * 0.95, -L * 0.60, wl + W * 0.50),
               target=(W * 0.15, -L * 0.05, wl), lens=30.0)
    render_still(os.path.join(OUT, "slab_scenic.png"))


if __name__ == "__main__":
    main()
