# SPDX-License-Identifier: GPL-2.0-or-later
"""
Surf Pool — a modern surfing wave pool generator for Blender.

Models a SurfLoch / Wavebender-style facility: a long, narrow basin with a deep
caisson channel on one side shoaling up to a reef shelf on the other, and a row
of **pneumatic caissons** that fire in sequence to launch a single clean wave
that propagates across the width and **peels down the line**, barreling as it
shoals over the reef.

How the wave is made (matching the real tech): each caisson is a chamber that,
when fired, displaces a slug of water -> a wave pulse that travels across the
pool. Firing the caissons with a small progressive delay makes the pulses
superpose into one wave that peels along the line; the delay sets the peel rate.
On the **Wavebender arc** the caissons sit on a concave curve so the breaking
line bends like a reef pass. The surface is the closed-form superposition of the
caisson pulses, evaluated each frame -> cheap (no fluid solve), fully editable.

Install:  Edit > Preferences > Add-ons > Install from Disk... > pick this file,
          then enable "Add Mesh: Surf Pool".
Use:      3D Viewport > N-panel > "Surf Pool" tab > Create / Rebuild Surf Pool,
          then press Play (or scrub) to watch the caissons fire and the wave peel.

Target: Blender 5.x (tolerant of 4.2–5.x; attribute/socket writes are guarded).
"""

bl_info = {
    "name": "Surf Pool",
    "author": "QSP",
    "version": (2, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar (N) > Surf Pool",
    "description": "Modern surfing wave pool driven by sequenced pneumatic caissons",
    "category": "Add Mesh",
}

import math

import numpy as np
import bpy
import bmesh
from bpy.app.handlers import persistent
from bpy.props import (
    BoolProperty, FloatProperty, IntProperty, FloatVectorProperty, PointerProperty, StringProperty,
)
from bpy.types import Operator, Panel, PropertyGroup

COLLECTION = "SurfPool"
OBJ_FLOOR = "SurfPool_Floor"
OBJ_WALLS = "SurfPool_Walls"
OBJ_WATER = "SurfPool_Water"
OBJ_CAISSON = "SurfPool_Caisson_"
OBJ_SUN = "SurfPool_Sun"
FOAM_LAYER = "foam"
FLAG = "is_surf_pool"

# One-click wave personalities. Geometry + wave settings applied then rebuilt.
PRESETS = {
    "BEGINNER": {
        "pool_length": 160.0, "pool_width": 60.0, "water_level": 2.0,
        "reef_ledge": 0.55, "reef_abruptness": 0.15, "num_caissons": 28, "arc_layout": True,
        "arc_depth": 4.0, "firing_delay": 0.18, "firing_period": 12.0, "wave_celerity": 7.0,
        "wave_height": 0.7, "wavelength": 40.0, "steepness": 0.5, "reef_steepen": 0.5,
        "ambient_chop": 0.25, "foam_amount": 0.8},
    "PERFORMANCE": {
        "pool_length": 180.0, "pool_width": 55.0, "water_level": 2.5,
        "reef_ledge": 0.60, "reef_abruptness": 0.45, "num_caissons": 28, "arc_layout": True,
        "arc_depth": 6.0, "firing_delay": 0.12, "firing_period": 12.0, "wave_celerity": 8.0,
        "wave_height": 1.6, "wavelength": 22.0, "steepness": 1.0, "reef_steepen": 1.0,
        "ambient_chop": 0.25, "foam_amount": 1.2},
    "SLAB": {
        "pool_length": 150.0, "pool_width": 46.0, "water_level": 2.2,
        "reef_ledge": 0.66, "reef_abruptness": 0.92, "num_caissons": 28, "arc_layout": True,
        "arc_depth": 7.0, "firing_delay": 0.11, "firing_period": 13.0, "wave_celerity": 7.0,
        "wave_height": 1.7, "wavelength": 18.0, "steepness": 1.0, "reef_steepen": 1.4,
        "ambient_chop": 0.12, "foam_amount": 1.5},
}


# ---------------------------------------------------------------------------
# Helpers (version-tolerant)
# ---------------------------------------------------------------------------
def _set(obj, attr, value):
    if hasattr(obj, attr):
        try:
            setattr(obj, attr, value)
            return True
        except Exception:
            return False
    return False


def _set_input(node, names, value):
    if isinstance(names, str):
        names = (names,)
    for n in names:
        if n in node.inputs:
            try:
                node.inputs[n].default_value = value
                return True
            except Exception:
                pass
    return False


def _fps(scene):
    return max(1.0, scene.render.fps / max(1, scene.render.fps_base))


def _reef_height(props):
    return props.water_level * 0.95  # reef shelf crests just under the surface


def _shoal_profile(props, xn):
    """0 in the deep channel rising to 1 over the reef. A logistic centred on
    the reef position whose sharpness is the abruptness: low = gradual reef,
    high = a sudden ledge (slab). Works for scalars and numpy arrays."""
    edge = props.reef_ledge
    sharp = 1.5 + 30.0 * props.reef_abruptness
    g = 1.0 / (1.0 + np.exp(-sharp * (xn - edge)))
    g0 = 1.0 / (1.0 + np.exp(-sharp * (0.0 - edge)))
    g1 = 1.0 / (1.0 + np.exp(-sharp * (1.0 - edge)))
    return np.clip((g - g0) / max(g1 - g0, 1e-6), 0.0, 1.0)


def _floor_z(props, x):
    half_w = props.pool_width / 2.0
    xn = min(max((x + half_w) / max(props.pool_width, 1e-6), 0.0), 1.0)
    return _reef_height(props) * float(_shoal_profile(props, xn))


def _ensure_collection(context):
    coll = bpy.data.collections.get(COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(COLLECTION)
        context.scene.collection.children.link(coll)
    return coll


def _clear():
    for obj in [o for o in bpy.data.objects if o.name.startswith("SurfPool_")]:
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data is not None and getattr(data, "users", 1) == 0:
            if isinstance(data, bpy.types.Mesh):
                bpy.data.meshes.remove(data)
            elif isinstance(data, bpy.types.Light):
                bpy.data.lights.remove(data)
    coll = bpy.data.collections.get(COLLECTION)
    if coll is not None and len(coll.objects) == 0:
        bpy.data.collections.remove(coll)


def _new_mesh_object(name, coll):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    coll.objects.link(obj)
    return obj, mesh


# ---------------------------------------------------------------------------
# Geometry
#   X = wave propagation (basin width).  -X = deep caisson channel, +X = reef.
#   Y = the line the wave peels along    (basin length, the long axis).
# ---------------------------------------------------------------------------
def _build_floor(props, coll):
    obj, mesh = _new_mesh_object(OBJ_FLOOR, coll)
    L, W = props.pool_length, props.pool_width
    x0, x1 = -W / 2.0, W / 2.0
    y0, y1 = -L / 2.0, L / 2.0
    nx = max(2, props.width_segments)
    ny = max(2, props.length_segments // 2)
    bm = bmesh.new()
    grid = []
    for j in range(ny + 1):
        y = y0 + (y1 - y0) * j / ny
        row = []
        for i in range(nx + 1):
            x = x0 + (x1 - x0) * i / nx
            row.append(bm.verts.new((x, y, float(_floor_z(props, x)))))
        grid.append(row)
    for j in range(ny):
        for i in range(nx):
            bm.faces.new((grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]))
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    sol = obj.modifiers.new("Thickness", "SOLIDIFY")
    sol.thickness = props.wall_thickness
    sol.offset = -1.0
    return obj


def _build_walls(props, coll):
    obj, mesh = _new_mesh_object(OBJ_WALLS, coll)
    L, W = props.pool_length, props.pool_width
    x0, x1 = -W / 2.0, W / 2.0
    y0, y1 = -L / 2.0, L / 2.0
    zb = -props.wall_thickness
    zt = props.water_level + props.freeboard
    bm = bmesh.new()
    bottom = [bm.verts.new(c) for c in ((x0, y0, zb), (x1, y0, zb), (x1, y1, zb), (x0, y1, zb))]
    top = [bm.verts.new((v.co.x, v.co.y, zt)) for v in bottom]
    for i in range(4):
        a, b = i, (i + 1) % 4
        bm.faces.new((bottom[a], bottom[b], top[b], top[a]))
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    sol = obj.modifiers.new("Thickness", "SOLIDIFY")
    sol.thickness = props.wall_thickness
    sol.offset = 1.0
    return obj


def caisson_positions(props):
    """Return (xi, yi, xback) for the caisson array. Straight = constant x;
    Wavebender arc = a concave parabola so the breaking line bends."""
    n = max(1, props.num_caissons)
    L, W = props.pool_length, props.pool_width
    margin = max(L * 0.04, 3.0)
    yi = np.linspace(-L / 2.0 + margin, L / 2.0 - margin, n)
    xback = -W / 2.0 + props.wall_thickness + 1.2
    if props.arc_layout and n > 1:
        s = yi / max(L / 2.0, 1e-6)
        xi = xback + props.arc_depth * (1.0 - s * s)   # center set forward -> concave arc
    else:
        xi = np.full(n, xback)
    return xi, yi, xback


def _caisson_base_z(props):
    return props.water_level * 0.5


def _build_caissons(props, coll):
    xi, yi, _ = caisson_positions(props)
    n = len(yi)
    spacing = abs(yi[1] - yi[0]) if n > 1 else props.pool_length
    cz0 = _caisson_base_z(props)
    h = props.water_level + props.freeboard
    for ci in range(n):
        obj, mesh = _new_mesh_object(f"{OBJ_CAISSON}{ci:02d}", coll)
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bm.to_mesh(mesh)
        bm.free()
        obj.scale = (2.2, spacing * 0.72, h)
        obj.location = (float(xi[ci]), float(yi[ci]), cz0)


def _build_water(props, coll):
    obj, mesh = _new_mesh_object(OBJ_WATER, coll)
    L, W = props.pool_length, props.pool_width
    inset = props.wall_thickness
    x0, x1 = -W / 2.0 + inset, W / 2.0 - inset
    y0, y1 = -L / 2.0 + inset, L / 2.0 - inset
    z = props.water_level
    nx = max(2, props.width_segments)
    ny = max(2, props.length_segments)
    bm = bmesh.new()
    grid = []
    for j in range(ny + 1):
        y = y0 + (y1 - y0) * j / ny
        row = []
        for i in range(nx + 1):
            x = x0 + (x1 - x0) * i / nx
            row.append(bm.verts.new((x, y, z)))
        grid.append(row)
    for j in range(ny):
        for i in range(nx):
            bm.faces.new((grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]))
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    for poly in mesh.polygons:
        poly.use_smooth = True
    try:
        mesh.color_attributes.new(name=FOAM_LAYER, type="FLOAT_COLOR", domain="POINT")
    except Exception:
        pass
    try:
        ra = mesh.attributes.new(name="rest_pos", type="FLOAT_VECTOR", domain="POINT")
        rest = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", rest)
        ra.data.foreach_set("vector", rest)
    except Exception:
        pass
    obj[FLAG] = True
    obj.data.materials.append(_build_material(props))
    return obj


# ---------------------------------------------------------------------------
# Wave engine — superposition of sequenced caisson pulses, per frame
# ---------------------------------------------------------------------------
def firing_front(scene, props):
    """(x, y) of the caisson currently firing this cycle (for camera tracking)."""
    xi, yi, xback = caisson_positions(props)
    n = len(yi)
    t = scene.frame_current / _fps(scene)
    tc = t % max(props.firing_period, 1e-3)
    idx = int(np.clip(tc / max(props.firing_delay, 1e-4), 0, n - 1))
    return float(xi[idx]), float(yi[idx])


def compute_surface(scene):
    obj = bpy.data.objects.get(OBJ_WATER)
    if obj is None or not obj.get(FLAG):
        return
    me = obj.data
    p = getattr(scene, "surf_pool", None)
    if p is None:
        return
    nv = len(me.vertices)
    if nv == 0 or "rest_pos" not in me.attributes:
        return

    rest = np.empty(nv * 3, dtype=np.float64)
    me.attributes["rest_pos"].data.foreach_get("vector", rest)
    rest = rest.reshape(nv, 3)
    X0 = rest[:, 0]
    Y0 = rest[:, 1]

    half_w = max(p.pool_width / 2.0, 1e-6)
    lam = max(p.wavelength, 1e-3)
    k = 2.0 * np.pi / lam
    c = max(p.wave_celerity, 1e-3)

    xn = np.clip((X0 + half_w) / (2.0 * half_w), 0.0, 1.0)
    g = _shoal_profile(p, xn)
    shoal = 0.30 + 1.70 * g
    Q = p.steepness + p.reef_steepen * g
    depth = np.maximum(p.water_level - _reef_height(p) * g, 0.15)

    xi, yi, xback = caisson_positions(p)
    n = len(yi)
    spacing = abs(yi[1] - yi[0]) if n > 1 else p.pool_length
    sigma_y = max(spacing * 0.85, 1e-3)
    t = scene.frame_current / _fps(scene)
    tc = t % max(p.firing_period, 1e-3)

    Z = np.zeros(nv)
    DX = np.zeros(nv)
    for ci in range(n):
        age = tc - ci * p.firing_delay          # seconds since caisson ci fired this cycle
        if age <= 0.0:
            continue
        cx = xi[ci]
        crest_x = cx + c * age                  # this pulse's crest has travelled c*age across
        env_y = np.exp(-((Y0 - yi[ci]) / sigma_y) ** 2)        # caisson's band along the line
        packet = np.exp(-((X0 - crest_x) / (0.7 * lam)) ** 2)  # single travelling crest
        fwd = np.clip((X0 - cx) / (0.3 * lam), 0.0, 1.0)       # only in front of the caisson
        amp = p.wave_height * shoal * env_y * packet * fwd
        theta = k * (X0 - crest_x)
        Z += amp * np.cos(theta)
        DX += -(Q * amp) * np.sin(theta)

    z = p.water_level + Z
    if p.ambient_chop > 0.0:
        z += p.ambient_chop * 0.05 * np.sin(2.0 * np.pi * (0.5 * X0 + 0.7 * Y0) / 2.6 + 3.0 * tc)

    co = np.empty((nv, 3), dtype=np.float64)
    co[:, 0] = X0 + DX
    co[:, 1] = Y0
    co[:, 2] = z
    me.vertices.foreach_set("co", co.reshape(-1))
    me.update()

    # depth-limited breaking: foam where the crest height approaches local depth
    if FOAM_LAYER in me.color_attributes:
        crest_h = np.maximum(z - p.water_level, 0.0)
        break_i = np.clip((crest_h / depth - 0.55) * 2.0, 0.0, 1.0) * p.foam_amount
        foam = np.clip(break_i, 0.0, 1.0).astype(np.float32)
        rgba = np.ones((nv, 4), dtype=np.float32)
        rgba[:, 0] = foam
        rgba[:, 1] = foam
        rgba[:, 2] = foam
        try:
            me.color_attributes[FOAM_LAYER].data.foreach_set("color", rgba.reshape(-1))
        except Exception:
            pass

    # caisson firing motion: a quick plunge as each fires, in sequence
    cz0 = _caisson_base_z(p)
    for ci in range(n):
        cobj = bpy.data.objects.get(f"{OBJ_CAISSON}{ci:02d}")
        if cobj is None:
            continue
        age = tc - ci * p.firing_delay
        plunge = math.sin(math.pi * age / 0.6) * 0.5 if 0.0 <= age <= 0.6 else 0.0
        cobj.location.z = cz0 - plunge


@persistent
def _surf_frame_handler(scene, depsgraph=None):
    try:
        compute_surface(scene)
    except Exception:
        pass


def _live_update(self, context):
    if context is not None and context.scene is not None:
        compute_surface(context.scene)


# ---------------------------------------------------------------------------
# Material (glassy water + foam from the color attribute)
# ---------------------------------------------------------------------------
def _build_material(props):
    mat = bpy.data.materials.get("SurfPool_Water")
    if mat is None:
        mat = bpy.data.materials.new("SurfPool_Water")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (300, 0)
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    base = (props.water_color[0], props.water_color[1], props.water_color[2], 1.0)
    _set_input(bsdf, "Base Color", base)
    _set_input(bsdf, "Roughness", 0.03)
    _set_input(bsdf, "IOR", 1.333)
    _set_input(bsdf, ("Transmission Weight", "Transmission"), 0.6)

    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.location = (-300, -180)
    attr.attribute_name = FOAM_LAYER
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (-90, -180)
    if "Fac" in attr.outputs:
        nt.links.new(attr.outputs["Fac"], ramp.inputs["Fac"])
    mix_c = nt.nodes.new("ShaderNodeMix"); mix_c.location = (70, 0); mix_c.data_type = "RGBA"
    _set_input(mix_c, ("A", "Color1"), base)
    _set_input(mix_c, ("B", "Color2"), (1.0, 1.0, 1.0, 1.0))
    nt.links.new(ramp.outputs["Color"], mix_c.inputs["Factor"])
    if "Result" in mix_c.outputs:
        nt.links.new(mix_c.outputs["Result"], bsdf.inputs["Base Color"])
    return mat


def _build_sun(props, coll):
    light = bpy.data.lights.new(OBJ_SUN, "SUN")
    light.energy = 3.5
    obj = bpy.data.objects.new(OBJ_SUN, light)
    obj.rotation_euler = (math.radians(55), math.radians(8), math.radians(150))
    coll.objects.link(obj)
    return obj


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------
class SurfPoolProps(PropertyGroup):
    # --- basin & reef (rebuild) ---
    pool_length: FloatProperty(name="Line Length", default=150.0, min=10.0, soft_max=800.0, unit="LENGTH")
    pool_width: FloatProperty(name="Width", default=46.0, min=5.0, soft_max=200.0, unit="LENGTH")
    water_level: FloatProperty(name="Water Level", default=2.2, min=0.2, soft_max=20.0, unit="LENGTH")
    wall_thickness: FloatProperty(name="Wall Thickness", default=0.4, min=0.01, soft_max=5.0, unit="LENGTH")
    freeboard: FloatProperty(name="Freeboard", default=0.8, min=0.0, soft_max=10.0, unit="LENGTH")
    reef_ledge: FloatProperty(name="Reef Position", default=0.66, min=0.05, max=0.98,
                              description="Across-width position where the reef shelf rises")
    reef_abruptness: FloatProperty(name="Reef Abruptness", default=0.92, min=0.0, max=1.0,
                                   description="0 = gradual reef, 1 = sudden ledge (slab) that jacks the wave")
    length_segments: IntProperty(name="Segments (Line)", default=220, min=4, soft_max=600)
    width_segments: IntProperty(name="Segments (Width)", default=96, min=4, soft_max=400)

    # --- caisson array (rebuild) ---
    num_caissons: IntProperty(name="Caissons", default=28, min=1, soft_max=64,
                              description="Number of pneumatic caisson wave engines along the back")
    arc_layout: BoolProperty(name="Wavebender Arc", default=True,
                             description="Concave arc layout so the breaking line bends like a reef pass")
    arc_depth: FloatProperty(name="Arc Depth", default=7.0, min=0.0, soft_max=40.0, unit="LENGTH",
                             description="How far the arc bows; 0 = straight wall")

    # --- firing sequence + wave (live) ---
    firing_delay: FloatProperty(name="Firing Delay", default=0.11, min=0.0, soft_max=1.0, update=_live_update,
                                description="Delay between adjacent caissons firing; sets the peel rate")
    firing_period: FloatProperty(name="Set Interval", default=13.0, min=0.5, soft_max=60.0, update=_live_update,
                                 description="Seconds between waves (one full firing sweep per cycle)")
    wave_celerity: FloatProperty(name="Wave Celerity", default=7.0, min=0.1, soft_max=40.0, update=_live_update,
                                 description="Speed each caisson pulse travels across the width")
    wave_height: FloatProperty(name="Wave Height", default=1.7, min=0.0, soft_max=8.0, update=_live_update)
    wavelength: FloatProperty(name="Wavelength", default=18.0, min=1.0, soft_max=300.0, update=_live_update,
                              description="Crest width of each pulse; shorter steepens the barrel")
    steepness: FloatProperty(name="Steepness", default=1.0, min=0.0, soft_max=3.0, update=_live_update,
                             description="Gerstner steepness; higher pitches the face toward a barrel")
    reef_steepen: FloatProperty(name="Reef Steepening", default=1.4, min=0.0, soft_max=3.0, update=_live_update,
                                description="Extra steepness as the wave shoals over the reef")
    ambient_chop: FloatProperty(name="Ambient Chop", default=0.12, min=0.0, soft_max=3.0, update=_live_update)
    foam_amount: FloatProperty(name="Foam", default=1.5, min=0.0, soft_max=3.0, update=_live_update)

    # --- extras (rebuild) ---
    water_color: FloatVectorProperty(name="Water Color", subtype="COLOR", size=3,
                                     default=(0.0, 0.22, 0.30), min=0.0, max=1.0)
    add_caissons: BoolProperty(name="Show Caissons", default=True)
    add_lighting: BoolProperty(name="Add Sun", default=True)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
class SURFPOOL_OT_create(Operator):
    bl_idname = "surfpool.create"
    bl_label = "Create / Rebuild Surf Pool"
    bl_description = "Build (or rebuild) the surf pool from the current settings"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.surf_pool
        _clear()
        coll = _ensure_collection(context)
        _build_floor(props, coll)
        _build_walls(props, coll)
        _build_water(props, coll)
        if props.add_caissons:
            _build_caissons(props, coll)
        if props.add_lighting and not any(o.type == "LIGHT" for o in context.scene.objects):
            _build_sun(props, coll)
        compute_surface(context.scene)
        self.report({"INFO"}, "Surf pool built — press Play to fire the caissons")
        return {"FINISHED"}


class SURFPOOL_OT_preset(Operator):
    bl_idname = "surfpool.preset"
    bl_label = "Apply Surf Preset"
    bl_description = "Apply a wave preset and rebuild the pool"
    bl_options = {"REGISTER", "UNDO"}

    preset: StringProperty(default="SLAB")

    def execute(self, context):
        vals = PRESETS.get(self.preset)
        if not vals:
            self.report({"WARNING"}, f"Unknown preset: {self.preset}")
            return {"CANCELLED"}
        props = context.scene.surf_pool
        for key, value in vals.items():
            setattr(props, key, value)
        bpy.ops.surfpool.create()
        self.report({"INFO"}, f"{self.preset.title()} preset applied")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
class SURFPOOL_PT_panel(Panel):
    bl_label = "Surf Pool"
    bl_idname = "SURFPOOL_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Surf Pool"

    def draw(self, context):
        layout = self.layout
        props = context.scene.surf_pool

        box = layout.box()
        box.label(text="Presets", icon="PRESET")
        row = box.row(align=True)
        row.operator("surfpool.preset", text="Beginner").preset = "BEGINNER"
        row.operator("surfpool.preset", text="Performance").preset = "PERFORMANCE"
        row.operator("surfpool.preset", text="Slab").preset = "SLAB"

        layout.operator("surfpool.create", icon="MOD_WAVE")

        box = layout.box()
        box.label(text="Basin & Reef", icon="MESH_GRID")
        col = box.column(align=True)
        col.prop(props, "pool_length")
        col.prop(props, "pool_width")
        col.prop(props, "water_level")
        col.prop(props, "wall_thickness")
        col.prop(props, "freeboard")
        sub = box.column(align=True)
        sub.prop(props, "reef_ledge")
        sub.prop(props, "reef_abruptness")
        sub = box.column(align=True)
        sub.prop(props, "length_segments")
        sub.prop(props, "width_segments")

        box = layout.box()
        box.label(text="Caisson Array", icon="MOD_ARRAY")
        col = box.column(align=True)
        col.prop(props, "num_caissons")
        col.prop(props, "arc_layout")
        sub = col.column(align=True)
        sub.enabled = props.arc_layout
        sub.prop(props, "arc_depth")
        box.label(text="(basin/reef/caisson edits need a Rebuild)", icon="INFO")

        box = layout.box()
        box.label(text="Firing & Wave (live)", icon="MOD_WAVE")
        col = box.column(align=True)
        col.prop(props, "firing_delay")
        col.prop(props, "firing_period")
        col.prop(props, "wave_celerity")
        col.prop(props, "wave_height")
        col.prop(props, "wavelength")
        col.prop(props, "steepness")
        col.prop(props, "reef_steepen")
        col.prop(props, "ambient_chop")
        col.prop(props, "foam_amount")
        spacing = (props.pool_length - 2 * max(props.pool_length * 0.04, 3.0)) / max(props.num_caissons - 1, 1)
        peel = spacing / props.firing_delay if props.firing_delay > 1e-4 else 0.0
        box.label(text=f"Peel speed ≈ {peel:.1f} m/s along the line", icon="INFO")

        box = layout.box()
        box.label(text="Extras", icon="SETTINGS")
        box.prop(props, "water_color")
        box.prop(props, "add_caissons")
        box.prop(props, "add_lighting")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
_classes = (
    SurfPoolProps,
    SURFPOOL_OT_create,
    SURFPOOL_OT_preset,
    SURFPOOL_PT_panel,
)


def _add_handler():
    handlers = bpy.app.handlers.frame_change_pre
    for h in list(handlers):
        if getattr(h, "__name__", "") == "_surf_frame_handler":
            handlers.remove(h)
    handlers.append(_surf_frame_handler)


def _remove_handler():
    handlers = bpy.app.handlers.frame_change_pre
    for h in list(handlers):
        if getattr(h, "__name__", "") == "_surf_frame_handler":
            handlers.remove(h)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.surf_pool = PointerProperty(type=SurfPoolProps)
    _add_handler()


def unregister():
    _remove_handler()
    del bpy.types.Scene.surf_pool
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
