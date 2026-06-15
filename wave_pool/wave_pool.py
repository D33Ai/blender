# SPDX-License-Identifier: GPL-2.0-or-later
"""
Wave Pool — a cost-effective, fully editable wave-pool generator for Blender.

Builds a complete pool scene (sloped-beach basin + walls + wavemaker) and an
animated water surface driven by the built-in **Ocean modifier**. No fluid
bake required: every wave parameter is exposed in a sidebar panel and edits the
live modifier instantly, so it is cheap to compute and easy to art-direct.

Install:  Edit > Preferences > Add-ons > Install from Disk... > pick this file,
          then enable "Add Mesh: Wave Pool".
Use:      3D Viewport > N-panel > "Wave Pool" tab > Create / Rebuild Wave Pool.

Tested target: Blender 5.x (uses the generic Mix shader node + Principled
"Transmission Weight" socket). Attribute writes are guarded so minor API drift
across 4.2–5.x degrades gracefully instead of erroring.
"""

bl_info = {
    "name": "Wave Pool",
    "author": "QSP",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar (N) > Wave Pool",
    "description": "Generate an editable, low-cost wave pool using the Ocean modifier",
    "category": "Add Mesh",
}

import bpy
import bmesh
from bpy.props import (
    BoolProperty, FloatProperty, IntProperty, FloatVectorProperty, PointerProperty,
)
from bpy.types import Operator, Panel, PropertyGroup

COLLECTION = "WavePool"
OBJ_FLOOR = "WavePool_Floor"
OBJ_WALLS = "WavePool_Walls"
OBJ_WATER = "WavePool_Water"
OBJ_PADDLE = "WavePool_Wavemaker"
OBJ_SUN = "WavePool_Sun"
OCEAN_MOD = "Ocean"


# ---------------------------------------------------------------------------
# Small helpers (version-tolerant)
# ---------------------------------------------------------------------------
def _set(obj, attr, value):
    """Set an attribute only if it exists and accepts the value."""
    if hasattr(obj, attr):
        try:
            setattr(obj, attr, value)
            return True
        except Exception:
            return False
    return False


def _set_input(node, names, value):
    """Set the first matching node input socket by name (handles renames)."""
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


def _ensure_collection(context):
    coll = bpy.data.collections.get(COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(COLLECTION)
        context.scene.collection.children.link(coll)
    return coll


def _clear():
    """Remove a previous build so Create acts as a clean rebuild."""
    for name in (OBJ_FLOOR, OBJ_WALLS, OBJ_WATER, OBJ_PADDLE, OBJ_SUN):
        obj = bpy.data.objects.get(name)
        if obj is not None:
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            # remove now-orphaned mesh/light datablocks
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
# Geometry builders (bmesh — no operators, no context dependence)
# ---------------------------------------------------------------------------
def _build_floor(props, coll):
    """Sloped floor: flat & deep at the wavemaker end, ramping up to a beach."""
    obj, mesh = _new_mesh_object(OBJ_FLOOR, coll)
    L, W = props.pool_length, props.pool_width
    x0, x1 = -W / 2.0, W / 2.0
    y0, y1 = -L / 2.0, L / 2.0           # -Y = wavemaker (deep), +Y = beach (shallow)
    beach_start = y1 - max(props.beach_length, 0.0)
    beach_top = props.water_level * 1.05  # beach emerges just above the waterline

    def zfunc(y):
        if y <= beach_start or props.beach_length <= 0.0:
            return 0.0
        t = (y - beach_start) / max(props.beach_length, 1e-6)
        return beach_top * min(max(t, 0.0), 1.0)

    nx = max(2, props.pool_width_segments)
    ny = max(2, props.pool_length_segments)
    bm = bmesh.new()
    grid = []
    for j in range(ny + 1):
        y = y0 + (y1 - y0) * j / ny
        row = []
        for i in range(nx + 1):
            x = x0 + (x1 - x0) * i / nx
            row.append(bm.verts.new((x, y, zfunc(y))))
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
    """Four perimeter walls from the floor up to the rim (open top)."""
    obj, mesh = _new_mesh_object(OBJ_WALLS, coll)
    L, W = props.pool_length, props.pool_width
    x0, x1 = -W / 2.0, W / 2.0
    y0, y1 = -L / 2.0, L / 2.0
    zb = -props.wall_thickness                       # base slightly below floor
    zt = props.pool_depth + props.freeboard          # rim above the water

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


def _build_paddle(props, coll):
    """A static wavemaker paddle at the deep (-Y) end (visual cue)."""
    obj, mesh = _new_mesh_object(OBJ_PADDLE, coll)
    W = props.pool_width
    x0, x1 = -W / 2.0 + props.wall_thickness, W / 2.0 - props.wall_thickness
    y = -props.pool_length / 2.0 + props.wall_thickness + 0.05
    th = 0.12
    zb, zt = 0.0, props.pool_depth + props.freeboard
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bm.to_mesh(mesh)
    bm.free()
    obj.scale = ((x1 - x0), th, (zt - zb))
    obj.location = (0.0, y, (zb + zt) / 2.0)
    return obj


def _build_water(props, coll):
    """Flat water grid + Ocean modifier (DISPLACE) + water material."""
    obj, mesh = _new_mesh_object(OBJ_WATER, coll)
    L, W = props.pool_length, props.pool_width
    inset = props.wall_thickness
    x0, x1 = -W / 2.0 + inset, W / 2.0 - inset
    y0, y1 = -L / 2.0 + inset, L / 2.0 - inset
    z = props.water_level
    n = max(2, props.water_resolution)

    bm = bmesh.new()
    grid = []
    for j in range(n + 1):
        y = y0 + (y1 - y0) * j / n
        row = []
        for i in range(n + 1):
            x = x0 + (x1 - x0) * i / n
            row.append(bm.verts.new((x, y, z)))
        grid.append(row)
    for j in range(n):
        for i in range(n):
            bm.faces.new((grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]))
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    for poly in mesh.polygons:
        poly.use_smooth = True

    mod = obj.modifiers.new(OCEAN_MOD, "OCEAN")
    _set(mod, "geometry_mode", "DISPLACE")
    apply_wave_params(props, mod)

    obj.data.materials.append(_build_water_material(props))
    return obj


# ---------------------------------------------------------------------------
# Ocean / wave parameters — applied on build AND live edits
# ---------------------------------------------------------------------------
def apply_wave_params(props, mod):
    """Push the panel's wave settings onto an Ocean modifier (guarded)."""
    # spatial size sets wavelength scale relative to pool size
    _set(mod, "spatial_size", max(1, int(round(max(props.pool_length, props.pool_width)))))
    _set(mod, "viewport_resolution", props.wave_detail_viewport)
    _set(mod, "resolution", props.wave_detail_render)
    _set(mod, "wind_velocity", props.wave_strength)
    _set(mod, "choppiness", props.choppiness)
    _set(mod, "wave_alignment", props.wave_alignment)
    _set(mod, "wave_direction", props.wave_direction)
    _set(mod, "depth", props.pool_depth)
    _set(mod, "random_seed", props.random_seed)
    _set(mod, "use_foam", props.use_foam)
    if props.use_foam:
        _set(mod, "foam_coverage", props.foam_coverage)
        _set(mod, "foam_layer_name", "foam")

    # animate: drive 'time' from the frame so the surface rolls without a bake
    fps = max(1.0, bpy.context.scene.render.fps / max(1, bpy.context.scene.render.fps_base))
    per_frame = props.wave_speed / fps
    try:
        mod.driver_remove("time")
    except Exception:
        pass
    if abs(per_frame) > 1e-9:
        try:
            drv = mod.driver_add("time").driver
            drv.type = "SCRIPTED"
            var = drv.variables.new()
            var.name = "f"
            var.type = "SINGLE_PROP"
            tgt = var.targets[0]
            tgt.id_type = "SCENE"
            tgt.id = bpy.context.scene
            tgt.data_path = "frame_current"
            drv.expression = f"f * {per_frame:.8f}"
        except Exception:
            _set(mod, "time", 1.0)


def _live_update(self, context):
    """Panel callback: re-apply wave settings to the existing modifier live."""
    obj = bpy.data.objects.get(OBJ_WATER)
    if obj is None:
        return
    mod = obj.modifiers.get(OCEAN_MOD)
    if mod is not None:
        apply_wave_params(self, mod)


# ---------------------------------------------------------------------------
# Water material (generic Mix node + Principled, 4.x/5.x sockets)
# ---------------------------------------------------------------------------
def _build_water_material(props):
    mat = bpy.data.materials.get("WavePool_Water")
    if mat is None:
        mat = bpy.data.materials.new("WavePool_Water")
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

    if props.use_foam:
        attr = nt.nodes.new("ShaderNodeAttribute")
        attr.location = (-300, -200)
        attr.attribute_name = "foam"
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.location = (-100, -200)
        if "Fac" in attr.outputs:
            nt.links.new(attr.outputs["Fac"], ramp.inputs["Fac"])
        # color: blend base -> white where foam
        mix_c = nt.nodes.new("ShaderNodeMix")
        mix_c.location = (60, 0)
        mix_c.data_type = "RGBA"
        _set_input(mix_c, ("A", "Color1"), base)
        _set_input(mix_c, ("B", "Color2"), (1.0, 1.0, 1.0, 1.0))
        nt.links.new(ramp.outputs["Color"], mix_c.inputs["Factor"])
        if "Result" in mix_c.outputs:
            nt.links.new(mix_c.outputs["Result"], bsdf.inputs["Base Color"])
        # roughness: foam is rougher than glassy water
        mix_r = nt.nodes.new("ShaderNodeMix")
        mix_r.location = (60, -250)
        mix_r.data_type = "FLOAT"
        _set_input(mix_r, "A", 0.02)
        _set_input(mix_r, "B", 0.6)
        nt.links.new(ramp.outputs["Color"], mix_r.inputs["Factor"])
        if "Result" in mix_r.outputs:
            nt.links.new(mix_r.outputs["Result"], bsdf.inputs["Roughness"])
    return mat


def _build_sun(props, coll):
    light = bpy.data.lights.new(OBJ_SUN, "SUN")
    light.energy = 3.0
    obj = bpy.data.objects.new(OBJ_SUN, light)
    obj.rotation_euler = (0.6, 0.1, 0.8)
    coll.objects.link(obj)
    return obj


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------
class WavePoolProps(PropertyGroup):
    # --- pool dimensions (rebuild required) ---
    pool_length: FloatProperty(name="Length", default=25.0, min=2.0, soft_max=200.0, unit="LENGTH")
    pool_width: FloatProperty(name="Width", default=12.0, min=2.0, soft_max=200.0, unit="LENGTH")
    pool_depth: FloatProperty(name="Depth", default=2.0, min=0.2, soft_max=20.0, unit="LENGTH")
    water_level: FloatProperty(name="Water Level", default=1.7, min=0.0, soft_max=20.0, unit="LENGTH")
    beach_length: FloatProperty(name="Beach Length", default=6.0, min=0.0, soft_max=100.0, unit="LENGTH",
                                description="Sloped shallow end where waves run up")
    wall_thickness: FloatProperty(name="Wall Thickness", default=0.3, min=0.01, soft_max=5.0, unit="LENGTH")
    freeboard: FloatProperty(name="Freeboard", default=0.4, min=0.0, soft_max=10.0, unit="LENGTH",
                             description="Wall height above the waterline")
    pool_length_segments: IntProperty(name="Floor Segments (L)", default=24, min=2, soft_max=200)
    pool_width_segments: IntProperty(name="Floor Segments (W)", default=12, min=2, soft_max=200)
    water_resolution: IntProperty(name="Water Mesh Res", default=64, min=2, soft_max=400,
                                  description="Subdivisions of the water grid (cost vs detail)")

    # --- waves (live edit) ---
    wave_strength: FloatProperty(name="Wave Strength", default=4.0, min=0.0, soft_max=40.0, update=_live_update,
                                 description="Wind speed driving wave size (Ocean modifier)")
    wave_speed: FloatProperty(name="Wave Speed", default=1.0, min=-20.0, soft_max=20.0, update=_live_update,
                              description="Animation speed of the surface (drives Ocean 'time')")
    choppiness: FloatProperty(name="Choppiness", default=0.7, min=0.0, soft_max=4.0, update=_live_update)
    wave_alignment: FloatProperty(name="Directionality", default=0.7, min=0.0, max=10.0, update=_live_update,
                                  description="How aligned waves are to one direction (pool-like rolling sets)")
    wave_direction: FloatProperty(name="Direction", default=90.0, min=0.0, max=360.0, update=_live_update,
                                  description="Wave travel direction in degrees (toward the beach)")
    random_seed: IntProperty(name="Seed", default=0, min=0, soft_max=1000, update=_live_update)
    wave_detail_viewport: IntProperty(name="Detail (Viewport)", default=7, min=1, max=12, update=_live_update,
                                      description="Ocean FFT resolution in the viewport (keep low for speed)")
    wave_detail_render: IntProperty(name="Detail (Render)", default=9, min=1, max=14, update=_live_update)

    # --- foam ---
    use_foam: BoolProperty(name="Foam", default=True, update=_live_update)
    foam_coverage: FloatProperty(name="Foam Coverage", default=0.4, min=-1.0, max=1.0, update=_live_update)

    # --- extras (rebuild) ---
    water_color: FloatVectorProperty(name="Water Color", subtype="COLOR", size=3,
                                     default=(0.0, 0.18, 0.25), min=0.0, max=1.0)
    add_wavemaker: BoolProperty(name="Wavemaker Paddle", default=True)
    add_lighting: BoolProperty(name="Add Sun", default=True)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
class WAVEPOOL_OT_create(Operator):
    bl_idname = "wavepool.create"
    bl_label = "Create / Rebuild Wave Pool"
    bl_description = "Build (or rebuild) the wave pool from the current settings"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.wave_pool
        _clear()
        coll = _ensure_collection(context)
        _build_floor(props, coll)
        _build_walls(props, coll)
        _build_water(props, coll)
        if props.add_wavemaker:
            _build_paddle(props, coll)
        if props.add_lighting and not any(o.type == "LIGHT" for o in context.scene.objects):
            _build_sun(props, coll)
        self.report({"INFO"}, "Wave pool built")
        return {"FINISHED"}


class WAVEPOOL_OT_bake(Operator):
    bl_idname = "wavepool.bake"
    bl_label = "Bake Ocean"
    bl_description = "Bake the Ocean modifier to cache for faster playback / render"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        obj = bpy.data.objects.get(OBJ_WATER)
        return obj is not None and obj.modifiers.get(OCEAN_MOD) is not None

    def execute(self, context):
        obj = bpy.data.objects.get(OBJ_WATER)
        mod = obj.modifiers.get(OCEAN_MOD)
        _set(mod, "frame_start", context.scene.frame_start)
        _set(mod, "frame_end", context.scene.frame_end)
        try:
            with context.temp_override(object=obj, active_object=obj,
                                       selected_objects=[obj], selected_editable_objects=[obj]):
                bpy.ops.object.ocean_bake(modifier=mod.name)
            self.report({"INFO"}, "Ocean baked")
        except Exception as exc:  # pragma: no cover - depends on runtime
            self.report({"WARNING"}, f"Bake failed: {exc}")
            return {"CANCELLED"}
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
class WAVEPOOL_PT_panel(Panel):
    bl_label = "Wave Pool"
    bl_idname = "WAVEPOOL_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Wave Pool"

    def draw(self, context):
        layout = self.layout
        props = context.scene.wave_pool

        layout.operator("wavepool.create", icon="MOD_OCEAN")

        box = layout.box()
        box.label(text="Pool Dimensions", icon="MESH_GRID")
        col = box.column(align=True)
        col.prop(props, "pool_length")
        col.prop(props, "pool_width")
        col.prop(props, "pool_depth")
        col.prop(props, "water_level")
        col.prop(props, "beach_length")
        col.prop(props, "wall_thickness")
        col.prop(props, "freeboard")
        sub = box.column(align=True)
        sub.prop(props, "water_resolution")
        sub.prop(props, "pool_length_segments")
        sub.prop(props, "pool_width_segments")
        box.label(text="(dimension edits need a Rebuild)", icon="INFO")

        box = layout.box()
        box.label(text="Waves (live)", icon="FORCE_HARMONIC")
        col = box.column(align=True)
        col.prop(props, "wave_strength")
        col.prop(props, "wave_speed")
        col.prop(props, "choppiness")
        col.prop(props, "wave_alignment")
        col.prop(props, "wave_direction")
        col.prop(props, "random_seed")
        sub = box.column(align=True)
        sub.prop(props, "wave_detail_viewport")
        sub.prop(props, "wave_detail_render")

        box = layout.box()
        box.label(text="Foam", icon="OUTLINER_OB_FORCE_FIELD")
        box.prop(props, "use_foam")
        row = box.row()
        row.enabled = props.use_foam
        row.prop(props, "foam_coverage")

        box = layout.box()
        box.label(text="Extras", icon="SETTINGS")
        box.prop(props, "water_color")
        box.prop(props, "add_wavemaker")
        box.prop(props, "add_lighting")

        layout.operator("wavepool.bake", icon="RENDER_ANIMATION")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
_classes = (
    WavePoolProps,
    WAVEPOOL_OT_create,
    WAVEPOOL_OT_bake,
    WAVEPOOL_PT_panel,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.wave_pool = PointerProperty(type=WavePoolProps)


def unregister():
    del bpy.types.Scene.wave_pool
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
