# SPDX-License-Identifier: GPL-2.0-or-later
"""
Surf Pool — a modern surfing wave pool generator for Blender.

Models a Surf-Ranch / Wavegarden-style facility: a long, narrow basin with a
deep foil channel on one side shoaling up to a reef shelf on the other, and a
single clean wave that propagates across the width and **peels down the line**,
breaking (with foam) as it shoals over the reef. A foil carriage tracks along
the line in sync.

Why not the Ocean modifier? That produces a stochastic open-ocean spectrum —
many chaotic waves — not the single peeling wall a surf pool makes. Instead the
surface is a procedural traveling wave evaluated per frame from a closed-form
function: cheap (no fluid bake), fully art-directable, and every parameter is a
live slider.

Install:  Edit > Preferences > Add-ons > Install from Disk... > pick this file,
          then enable "Add Mesh: Surf Pool".
Use:      3D Viewport > N-panel > "Surf Pool" tab > Create / Rebuild Surf Pool,
          then press Play (or scrub) to see the wave peel.

Target: Blender 5.x (tolerant of 4.2–5.x; attribute/socket writes are guarded).
"""

bl_info = {
    "name": "Surf Pool",
    "author": "QSP",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar (N) > Surf Pool",
    "description": "Generate a modern surfing wave pool with a procedural peeling wave",
    "category": "Add Mesh",
}

import numpy as np
import bpy
import bmesh
from bpy.app.handlers import persistent
from bpy.props import (
    BoolProperty, FloatProperty, IntProperty, FloatVectorProperty, PointerProperty, StringProperty,
)
from bpy.types import Operator, Panel, PropertyGroup

# One-click wave personalities. Geometry + wave settings applied then rebuilt.
PRESETS = {
    "BEGINNER": {  # mellow rolling wall over a gradual reef
        "pool_length": 160.0, "pool_width": 60.0, "water_level": 2.0,
        "reef_ledge": 0.55, "reef_abruptness": 0.15,
        "wave_height": 0.7, "wavelength": 40.0, "wave_celerity": 7.0, "foil_speed": 8.0,
        "steepness": 0.5, "reef_steepen": 0.5, "ambient_chop": 0.25, "foam_amount": 0.8},
    "PERFORMANCE": {  # punchy, peeling barrel
        "pool_length": 180.0, "pool_width": 55.0, "water_level": 2.5,
        "reef_ledge": 0.60, "reef_abruptness": 0.45,
        "wave_height": 1.6, "wavelength": 22.0, "wave_celerity": 8.0, "foil_speed": 9.0,
        "steepness": 1.0, "reef_steepen": 1.0, "ambient_chop": 0.25, "foam_amount": 1.2},
    "SLAB": {  # heavy, square slab jacking over a sudden ledge
        "pool_length": 150.0, "pool_width": 46.0, "water_level": 2.2,
        "reef_ledge": 0.66, "reef_abruptness": 0.92,
        "wave_height": 1.7, "wavelength": 18.0, "wave_celerity": 7.0, "foil_speed": 7.5,
        "steepness": 1.0, "reef_steepen": 1.4, "ambient_chop": 0.12, "foam_amount": 1.5},
}

COLLECTION = "SurfPool"
OBJ_FLOOR = "SurfPool_Floor"
OBJ_WALLS = "SurfPool_Walls"
OBJ_WATER = "SurfPool_Water"
OBJ_FOIL = "SurfPool_Foil"
OBJ_SUN = "SurfPool_Sun"
FOAM_LAYER = "foam"
FLAG = "is_surf_pool"


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


def _ensure_collection(context):
    coll = bpy.data.collections.get(COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(COLLECTION)
        context.scene.collection.children.link(coll)
    return coll


def _clear():
    for name in (OBJ_FLOOR, OBJ_WALLS, OBJ_WATER, OBJ_FOIL, OBJ_SUN):
        obj = bpy.data.objects.get(name)
        if obj is not None:
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if data is not None and data.users == 0:
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
#   X = wave propagation (basin width).  -X = deep foil channel, +X = reef.
#   Y = the line the wave peels along    (basin length, the long axis).
# ---------------------------------------------------------------------------
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
    """Deep foil channel on -X, shoaling up to the reef shelf on +X."""
    half_w = props.pool_width / 2.0
    xn = min(max((x + half_w) / max(props.pool_width, 1e-6), 0.0), 1.0)
    return _reef_height(props) * float(_shoal_profile(props, xn))


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
    # foam color attribute (POINT domain) consumed by the material
    try:
        mesh.color_attributes.new(name=FOAM_LAYER, type="FLOAT_COLOR", domain="POINT")
    except Exception:
        pass
    # rest lattice: the kinematic Gerstner solver displaces from these original
    # (X,Y) each frame, so horizontal motion never drifts/accumulates
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


def _build_foil(props, coll):
    """The foil carriage. Its Y position is set kinematically by the handler;
    the wave is derived from that position, so the foil *generates* the wave."""
    obj, mesh = _new_mesh_object(OBJ_FOIL, coll)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bm.to_mesh(mesh)
    bm.free()
    obj.scale = (props.pool_width * 0.10, props.pool_length * 0.05, props.water_level + props.freeboard)
    x = _foil_x(props)
    obj.location = (x, -props.pool_length / 2.0, (props.water_level + props.freeboard) / 2.0)
    return obj


# ---------------------------------------------------------------------------
# Procedural wave — evaluated each frame
# ---------------------------------------------------------------------------
def _foil_x(props):
    return -props.pool_width / 2.0 + props.pool_width * 0.12


def foil_state(scene, p):
    """Kinematic foil: constant-velocity sweep down the line, one ride per cycle.
    Returns (x_foil, y_foil, t_cycle, v)."""
    L = p.pool_length
    v = p.foil_speed if abs(p.foil_speed) > 1e-6 else 1e-6
    period = abs(L / v)
    t = scene.frame_current / _fps(scene)
    tc = (t % period) if period > 0 else t
    y_foil = -L / 2.0 + v * tc
    return _foil_x(p), y_foil, tc, v


def compute_surface(scene):
    """Kinematic Gerstner wave: generated at the moving foil, propagating across
    the width and peeling down the line, steepening into a barrel on the reef."""
    obj = bpy.data.objects.get(OBJ_WATER)
    if obj is None or not obj.get(FLAG):
        return
    me = obj.data
    p = getattr(scene, "surf_pool", None)
    if p is None:
        return
    n = len(me.vertices)
    if n == 0 or "rest_pos" not in me.attributes:
        return

    # displace from the REST lattice each frame (horizontal motion can't drift)
    rest = np.empty(n * 3, dtype=np.float64)
    me.attributes["rest_pos"].data.foreach_get("vector", rest)
    rest = rest.reshape(n, 3)
    X0 = rest[:, 0]
    Y0 = rest[:, 1]

    half_w = max(p.pool_width / 2.0, 1e-6)
    lam = max(p.wavelength, 1e-3)
    k = 2.0 * np.pi / lam
    c = max(p.wave_celerity, 1e-3)
    omega = k * c

    x_foil, y_foil, tc, v = foil_state(scene, p)
    foil = bpy.data.objects.get(OBJ_FOIL)
    if foil is not None:                       # kinematically position the carriage
        loc = list(foil.location)
        loc[1] = y_foil
        foil.location = loc

    # kinematic coupling: a row at Y0 was passed by the foil tau seconds ago, and
    # the wave it launched has since propagated c*tau across the width.
    tau = (y_foil - Y0) / v if v != 0 else np.zeros_like(Y0)
    tau_pos = np.maximum(tau, 0.0)
    passed = (Y0 <= y_foil + 1e-6).astype(np.float64)
    reach = x_foil + c * tau_pos                       # how far the front has travelled
    window = np.clip((reach - X0) / (0.5 * lam), 0.0, 1.0)   # 0 ahead of the front
    xn = np.clip((X0 + half_w) / (2.0 * half_w), 0.0, 1.0)
    g = _shoal_profile(p, xn)                          # 0 in channel -> 1 over reef/ledge
    shoal = 0.25 + 1.75 * g                            # wave is small in the deep, jacks on the ledge

    active = window * passed
    A = p.wave_height * shoal * active
    Q = np.clip(p.steepness + p.reef_steepen * g, 0.0, 5.0)  # steepening concentrated at the ledge
    theta = k * (X0 - x_foil) - omega * tau_pos

    # Gerstner: vertical lift + horizontal pull toward the crest (lets it pitch/curl)
    dX = -(Q * A) * np.sin(theta)
    z = p.water_level + A * np.cos(theta)
    if p.ambient_chop > 0.0:
        z += p.ambient_chop * 0.05 * np.sin(2.0 * np.pi * (0.5 * X0 + 0.7 * Y0) / 2.6 + 3.0 * tc)

    co = np.empty((n, 3), dtype=np.float64)
    co[:, 0] = X0 + dX
    co[:, 1] = Y0
    co[:, 2] = z
    me.vertices.foreach_set("co", co.reshape(-1))
    me.update()

    # foam: the curling lip (overhang strength Q*A*k) on the front face, plus
    # whitewater on the older, already-broken sections behind the foil
    if FOAM_LAYER in me.color_attributes:
        curl = np.clip(Q * A * k - 0.8, 0.0, 1.0)
        front = np.clip(np.sin(theta), 0.0, 1.0)
        age = np.clip(tau_pos / 2.5, 0.0, 1.0)
        foam = np.clip((curl * front + 0.5 * curl + 0.3 * age * xn) * active * p.foam_amount, 0.0, 1.0)
        rgba = np.ones((n, 4), dtype=np.float32)
        rgba[:, 0] = foam
        rgba[:, 1] = foam
        rgba[:, 2] = foam
        try:
            me.color_attributes[FOAM_LAYER].data.foreach_set("color", rgba.reshape(-1))
        except Exception:
            pass


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
    _set_input(bsdf, "Roughness", 0.02)
    _set_input(bsdf, "IOR", 1.333)
    _set_input(bsdf, ("Transmission Weight", "Transmission"), 1.0)

    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.location = (-300, -180)
    attr.attribute_name = FOAM_LAYER
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (-90, -180)
    if "Fac" in attr.outputs:
        nt.links.new(attr.outputs["Fac"], ramp.inputs["Fac"])
    mix_c = nt.nodes.new("ShaderNodeMix")
    mix_c.location = (70, 0)
    mix_c.data_type = "RGBA"
    _set_input(mix_c, ("A", "Color1"), base)
    _set_input(mix_c, ("B", "Color2"), (1.0, 1.0, 1.0, 1.0))
    nt.links.new(ramp.outputs["Color"], mix_c.inputs["Factor"])
    if "Result" in mix_c.outputs:
        nt.links.new(mix_c.outputs["Result"], bsdf.inputs["Base Color"])
    mix_r = nt.nodes.new("ShaderNodeMix")
    mix_r.location = (70, -260)
    mix_r.data_type = "FLOAT"
    _set_input(mix_r, "A", 0.02)
    _set_input(mix_r, "B", 0.7)
    nt.links.new(ramp.outputs["Color"], mix_r.inputs["Factor"])
    if "Result" in mix_r.outputs:
        nt.links.new(mix_r.outputs["Result"], bsdf.inputs["Roughness"])
    return mat


def _build_sun(props, coll):
    light = bpy.data.lights.new(OBJ_SUN, "SUN")
    light.energy = 3.5
    obj = bpy.data.objects.new(OBJ_SUN, light)
    obj.rotation_euler = (0.5, 0.15, 0.9)
    coll.objects.link(obj)
    return obj


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------
class SurfPoolProps(PropertyGroup):
    # --- basin (rebuild) ---
    pool_length: FloatProperty(name="Line Length", default=180.0, min=10.0, soft_max=800.0, unit="LENGTH",
                               description="Long axis the wave peels along")
    pool_width: FloatProperty(name="Width", default=55.0, min=5.0, soft_max=200.0, unit="LENGTH",
                              description="Short axis the wave propagates across")
    water_level: FloatProperty(name="Water Level", default=2.5, min=0.2, soft_max=20.0, unit="LENGTH")
    wall_thickness: FloatProperty(name="Wall Thickness", default=0.4, min=0.01, soft_max=5.0, unit="LENGTH")
    freeboard: FloatProperty(name="Freeboard", default=0.8, min=0.0, soft_max=10.0, unit="LENGTH")
    reef_ledge: FloatProperty(name="Reef Position", default=0.6, min=0.05, max=0.98, update=_live_update,
                              description="Across-width position (0=channel, 1=reef wall) where the shelf rises")
    reef_abruptness: FloatProperty(name="Reef Abruptness", default=0.4, min=0.0, max=1.0, update=_live_update,
                                   description="0 = gradual point-break reef, 1 = sudden ledge (slab) that jacks the wave")
    length_segments: IntProperty(name="Segments (Line)", default=220, min=4, soft_max=600,
                                 description="Water grid resolution along the line")
    width_segments: IntProperty(name="Segments (Width)", default=96, min=4, soft_max=400,
                                description="Water grid resolution across the width (resolves the barrel curl)")

    # --- wave: kinematic Gerstner (live) ---
    wave_height: FloatProperty(name="Wave Height", default=1.6, min=0.0, soft_max=8.0, update=_live_update,
                               description="Crest amplitude before shoaling (heavy = large)")
    wavelength: FloatProperty(name="Wavelength", default=22.0, min=1.0, soft_max=300.0, update=_live_update,
                              description="Crest-to-crest distance; shorter steepens the barrel")
    wave_celerity: FloatProperty(name="Wave Celerity", default=8.0, min=0.1, soft_max=40.0, update=_live_update,
                                 description="Speed the wave propagates across the width toward the reef")
    foil_speed: FloatProperty(name="Foil Speed", default=9.0, min=-40.0, soft_max=40.0, update=_live_update,
                              description="Kinematic speed of the foil carriage down the line; "
                                          "peel rate = celerity / foil speed")
    steepness: FloatProperty(name="Steepness", default=1.0, min=0.0, soft_max=3.0, update=_live_update,
                             description="Gerstner steepness; higher pitches the face forward toward a barrel")
    reef_steepen: FloatProperty(name="Reef Steepening", default=1.0, min=0.0, soft_max=3.0, update=_live_update,
                                description="Extra steepness as the wave shoals over the reef (makes it throw/barrel)")
    ambient_chop: FloatProperty(name="Ambient Chop", default=0.25, min=0.0, soft_max=3.0, update=_live_update)

    # --- foam ---
    foam_amount: FloatProperty(name="Foam", default=1.2, min=0.0, soft_max=3.0, update=_live_update)

    # --- extras (rebuild) ---
    water_color: FloatVectorProperty(name="Water Color", subtype="COLOR", size=3,
                                     default=(0.0, 0.22, 0.30), min=0.0, max=1.0)
    add_foil: BoolProperty(name="Foil Carriage", default=True)
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
        if props.add_foil:
            _build_foil(props, coll)
        if props.add_lighting and not any(o.type == "LIGHT" for o in context.scene.objects):
            _build_sun(props, coll)
        compute_surface(context.scene)            # show the wave at the current frame
        self.report({"INFO"}, "Surf pool built — press Play to see it peel")
        return {"FINISHED"}


class SURFPOOL_OT_preset(Operator):
    bl_idname = "surfpool.preset"
    bl_label = "Apply Surf Preset"
    bl_description = "Apply a wave preset and rebuild the pool"
    bl_options = {"REGISTER", "UNDO"}

    preset: StringProperty(default="PERFORMANCE")

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
        box.label(text="(basin/reef edits need a Rebuild)", icon="INFO")

        box = layout.box()
        box.label(text="Wave — kinematic Gerstner (live)", icon="MOD_WAVE")
        col = box.column(align=True)
        col.prop(props, "wave_height")
        col.prop(props, "wavelength")
        col.prop(props, "wave_celerity")
        col.prop(props, "foil_speed")
        col.prop(props, "steepness")
        col.prop(props, "reef_steepen")
        col.prop(props, "ambient_chop")
        col.prop(props, "foam_amount")
        peel = props.wave_celerity / props.foil_speed if props.foil_speed else 0.0
        box.label(text=f"Peel rate (celerity / foil) ≈ {peel:.2f}", icon="INFO")

        box = layout.box()
        box.label(text="Extras", icon="SETTINGS")
        box.prop(props, "water_color")
        box.prop(props, "add_foil")
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
