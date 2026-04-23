"""
WDAS cloud Blender scene generator — overhead sun, artistic rendering.

Usage:
    blender --background --python wdas_cloud_setup.py -- \
        --vdb /path/to/wdas_cloud_sixteenth.vdb \
        --out /path/to/wdas_cloud_scene.blend

Optional flags:
    --resolution 1920 1080
    --samples 256
    --density 3.0
    --sun-strength 6.0
    --render /path/to/output.png   (renders a still after saving)

Tested against Blender 4.x. Requires Cycles + a GPU with ≥8GB VRAM
for the sixteenth-resolution VDB (~180MB on disk, ~1-2GB allocated
with BVH + volume grid residency).
"""

import bpy
import sys
import os
import math
import argparse


# ---------- arg parsing (after `--` separator) ----------
def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--vdb", required=True, help="Path to wdas_cloud_sixteenth.vdb")
    p.add_argument("--out", required=True, help="Output .blend path")
    p.add_argument("--resolution", type=int, nargs=2, default=[1920, 1080])
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--density", type=float, default=0.4,
                   help="Density multiplier. 0.2-0.5 wispy/translucent, "
                        "1-2 medium, 3-5 solid.")
    p.add_argument("--anisotropy", type=float, default=0.8,
                   help="HG asymmetry 0-1. Higher = more forward scatter, "
                        "brighter transmission through the cloud.")
    p.add_argument("--rotation", type=float, default=0.0,
                   help="Rotate cloud around global Z axis in degrees. "
                        "Use to view different perspectives without moving "
                        "the camera or sun.")
    p.add_argument("--sun-strength", type=float, default=6.0)
    p.add_argument("--bottom-fill", type=float, default=1.5,
                   help="Strength of dim upward sun from below the cloud, "
                        "faking ground bounce. 0 disables. 1-3 brightens "
                        "the shadowed underside.")
    p.add_argument("--sky-strength", type=float, default=0.5,
                   help="Sky background strength.")
    p.add_argument("--volume-bounces", type=int, default=8,
                   help="Multi-scatter bounces. Higher = brighter shadowed "
                        "regions. 4 fast, 8 balanced, 16 reference-quality.")
    p.add_argument("--step-rate", type=float, default=1.0,
                   help="Volume step rate. Lower = finer sampling, slower.")
    p.add_argument("--render", default=None,
                   help="If set, render a still to this path after saving.")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU rendering. Use to isolate GPU-driver crashes.")
    p.add_argument("--no-denoise", action="store_true",
                   help="Disable denoiser. Use to isolate OIDN crashes.")
    p.add_argument("--eevee", action="store_true",
                   help="Use EEVEE instead of Cycles. Faster, no crashes on "
                        "problematic VDBs, but simplified scattering.")
    return p.parse_args(argv)


def reset_scene():
    """Wipe the default cube/light/camera."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def setup_render(args):
    scene = bpy.context.scene

    scene.render.resolution_x, scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False

    # Color management: Filmic + medium-high contrast looks right on clouds.
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'Medium High Contrast'
    scene.view_settings.exposure = 0.0

    if args.eevee:
        setup_eevee(scene, args)
    else:
        setup_cycles(scene, args)


def setup_eevee(scene, args):
    """EEVEE volume rendering. Fast, robust, less physically accurate."""
    # In Blender 4.2+ the engine is 'BLENDER_EEVEE_NEXT'. In 4.0/4.1 it's
    # 'BLENDER_EEVEE'. Try new name first, fall back to legacy.
    try:
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
        print("[INFO] Using EEVEE Next (4.2+)")
    except TypeError:
        scene.render.engine = 'BLENDER_EEVEE'
        print("[INFO] Using legacy EEVEE (4.0/4.1)")

    e = scene.eevee
    # Volumetric sampling — these are the key knobs for cloud quality.
    e.volumetric_tile_size = '4'           # finer = better detail, slower
    e.volumetric_samples = 128             # steps along view ray. 64-256 range.
    e.volumetric_sample_distribution = 0.8 # 0=linear, 1=exponential (more near)
    e.volumetric_light_clamp = 0.0         # no clamp, let highlights blow

    # Volumetric shadows — critical for self-shadowing on clouds.
    e.use_volumetric_shadows = True
    e.volumetric_shadow_samples = 64       # 16-128. Higher = crisper shadows.

    # Ambient occlusion adds depth to shadowed regions (cheap fake multi-scatter)
    try:
        e.use_gtao = True
        e.gtao_distance = 2.0
    except AttributeError:
        pass  # EEVEE Next removed GTAO


def setup_cycles(scene, args):
    scene.render.engine = 'CYCLES'

    if args.cpu:
        scene.cycles.device = 'CPU'
        print("[INFO] CPU rendering forced")
    else:
        # GPU if available; fall back silently to CPU.
        prefs = bpy.context.preferences.addons['cycles'].preferences
        for backend in ('OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI'):
            try:
                prefs.compute_device_type = backend
                prefs.get_devices()
                if any(d.use for d in prefs.devices):
                    scene.cycles.device = 'GPU'
                    print(f"[INFO] GPU rendering via {backend}")
                    break
            except TypeError:
                continue
        else:
            print("[INFO] No GPU backend available, using CPU")

    c = scene.cycles
    c.samples = args.samples
    c.use_adaptive_sampling = True
    c.adaptive_threshold = 0.01
    c.max_bounces = 12
    c.diffuse_bounces = 4
    c.glossy_bounces = 4
    c.transmission_bounces = 12
    c.volume_bounces = args.volume_bounces
    c.transparent_max_bounces = 8
    c.volume_step_rate = args.step_rate
    c.volume_max_steps = 1024

    # Denoiser: OIDN handles volume noise better than OptiX denoiser.
    if args.no_denoise:
        c.use_denoising = False
        print("[INFO] Denoising disabled")
    else:
        c.use_denoising = True
        try:
            c.denoiser = 'OPENIMAGEDENOISE'
            c.denoising_prefilter = 'FAST'
            c.denoising_input_passes = 'RGB'
        except (AttributeError, TypeError):
            pass


def import_vdb(vdb_path):
    """Import the OpenVDB as a Blender Volume object."""
    if not os.path.isfile(vdb_path):
        raise FileNotFoundError(f"VDB not found: {vdb_path}")
    bpy.ops.object.volume_import(filepath=vdb_path, align='WORLD',
                                 location=(0, 0, 0))
    vol = bpy.context.selected_objects[0]
    vol.name = "WDAS_Cloud"

    # WDAS cloud is ~kilometer-scale in native units — scale to fit a
    # ~10m framing for camera convenience. You can remove this if you
    # want physical scale.
    # NOTE: don't transform_apply on Volume objects in 4.0 — it warns
    # "Objects have no data to transform" because the VDB grid transform
    # is stored on the file, not the object data. Leave scale on the object.
    vol.scale = (0.01, 0.01, 0.01)
    bpy.context.view_layer.update()
    return vol


def build_cloud_material(vol_obj, density_mult, anisotropy):
    """Principled Volume driven by the 'density' grid via Attribute node.
    This is the correct 4.x path — density_attribute on the node doesn't
    exist; Volume Object Data has render-settings properties but those
    are for viewport display only, not the shader."""
    mat = bpy.data.materials.new("WDAS_CloudMat")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    pv = nt.nodes.new("ShaderNodeVolumePrincipled")
    attr = nt.nodes.new("ShaderNodeAttribute")
    mult = nt.nodes.new("ShaderNodeMath")

    attr.attribute_name = "density"
    mult.operation = 'MULTIPLY'
    mult.inputs[1].default_value = density_mult

    pv.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    pv.inputs["Absorption Color"].default_value = (0.98, 0.98, 1.0, 1.0)
    pv.inputs["Anisotropy"].default_value = anisotropy
    pv.inputs["Emission Strength"].default_value = 0.0

    out.location = (400, 0)
    pv.location = (150, 0)
    mult.location = (-100, 100)
    attr.location = (-350, 100)

    nt.links.new(attr.outputs["Fac"], mult.inputs[0])
    nt.links.new(mult.outputs[0], pv.inputs["Density"])
    nt.links.new(pv.outputs["Volume"], out.inputs["Volume"])

    vol_obj.data.materials.clear()
    vol_obj.data.materials.append(mat)
    return mat


def add_overhead_sun(strength, cloud_center=(0, 0, 0), cloud_size=10.0):
    """Sun overhead, positioned WELL outside cloud bounds.
    Sun lights are directional so location doesn't drive illumination,
    but Blender 4.0 has crash paths when a Sun is inside a dense volume
    during equiangular volume sampling. Place it high and far."""
    cx, cy, cz = cloud_center
    sun_z = cz + cloud_size * 5.0   # 5× cloud size above center
    bpy.ops.object.light_add(type='SUN', location=(cx, cy, sun_z))
    sun = bpy.context.object
    sun.name = "OverheadSun"
    # ~5° off zenith gives silhouette definition vs perfectly flat top light
    sun.rotation_euler = (math.radians(5), 0, math.radians(-30))
    sun.data.energy = strength
    sun.data.angle = math.radians(2.0)   # ~4× real sun; softens shadows
    sun.data.color = (1.0, 0.98, 0.92)   # faint warm
    return sun


def add_bottom_fill(strength, cloud_center=(0, 0, 0), cloud_size=10.0):
    """Dim upward-pointing sun from below — fakes ground bounce.
    Physically wrong (Earth's albedo is ~0.3 and diffuse, not a sun disk)
    but visually sells the look cheaply. Real path-traced solution would
    require a ground plane; this is 100× cheaper."""
    if strength <= 0:
        return None
    cx, cy, cz = cloud_center
    fill_z = cz - cloud_size * 5.0
    bpy.ops.object.light_add(type='SUN', location=(cx, cy, fill_z))
    fill = bpy.context.object
    fill.name = "BottomFill"
    # Pointing straight up (default sun points -Z; rotate 180° around X).
    fill.rotation_euler = (math.radians(180), 0, 0)
    fill.data.energy = strength
    fill.data.angle = math.radians(20.0)   # very soft
    fill.data.color = (0.75, 0.82, 0.95)   # cool, like reflected sky
    return fill


def setup_sky(strength):
    """Nishita physical sky for above-horizon, blended with a solid blue
    tint below horizon so the lower frame isn't pure black.
    Uses Geometry node's Incoming vector to detect ray direction."""
    world = bpy.data.worlds.new("Sky") if not bpy.data.worlds else bpy.data.worlds[0]
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    sky = nt.nodes.new("ShaderNodeTexSky")

    # Nodes for horizon blending: detect ray direction Z, mix sky above
    # with solid horizon color below.
    geom = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ramp = nt.nodes.new("ShaderNodeValToRGB")   # maps Z → blend factor
    mix_col = nt.nodes.new("ShaderNodeMixRGB")
    below_color = nt.nodes.new("ShaderNodeRGB")

    # Sky type: Blender 5.0 renamed NISHITA to SINGLE_SCATTERING etc.
    for sky_type in ('MULTIPLE_SCATTERING', 'SINGLE_SCATTERING', 'NISHITA'):
        try:
            sky.sky_type = sky_type
            print(f"[INFO] Sky type: {sky_type}")
            break
        except TypeError:
            continue

    for prop, value in [
        ("sun_elevation", math.radians(85)),
        ("sun_rotation", math.radians(-30)),
        ("altitude", 0.0),
        ("air_density", 1.0),
        ("dust_density", 0.3),
        ("ozone_density", 1.0),
        ("sun_disc", False),
        ("sun_size", math.radians(2.0)),
        ("sun_intensity", 0.0),
    ]:
        try:
            setattr(sky, prop, value)
        except (AttributeError, TypeError):
            pass

    # Below-horizon color: saturated deep blue, matches the Rayleigh sky
    # character rather than fading to gray.
    below_color.outputs[0].default_value = (0.18, 0.30, 0.60, 1.0)

    # Color ramp maps Z component of view ray to blend weight.
    # Z > 0 = looking up (use sky), Z < 0 = looking down (use below_color).
    # Soft transition across horizon.
    r = ramp.color_ramp
    r.interpolation = 'EASE'
    r.elements[0].position = 0.45     # just below horizon → fully below
    r.elements[0].color = (0, 0, 0, 1)
    r.elements[1].position = 0.55     # just above horizon → fully sky
    r.elements[1].color = (1, 1, 1, 1)

    # Separate Z from Incoming vector, remap to 0-1 for the ramp
    # (Z ranges -1 to 1; ramp expects 0-1, so use (Z + 1) / 2).
    map_range = nt.nodes.new("ShaderNodeMapRange")
    map_range.inputs['From Min'].default_value = -1.0
    map_range.inputs['From Max'].default_value = 1.0
    map_range.inputs['To Min'].default_value = 0.0
    map_range.inputs['To Max'].default_value = 1.0

    bg.inputs["Strength"].default_value = strength

    # Boost saturation — Nishita + filmic tonemap tends to read greyish.
    hsv = nt.nodes.new("ShaderNodeHueSaturation")
    hsv.inputs["Saturation"].default_value = 1.6

    nt.links.new(geom.outputs["Incoming"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"], map_range.inputs["Value"])
    nt.links.new(map_range.outputs["Result"], ramp.inputs["Fac"])
    nt.links.new(below_color.outputs["Color"], mix_col.inputs["Color1"])
    nt.links.new(sky.outputs["Color"], mix_col.inputs["Color2"])
    nt.links.new(ramp.outputs["Color"], mix_col.inputs["Fac"])
    nt.links.new(mix_col.outputs["Color"], hsv.inputs["Color"])
    nt.links.new(hsv.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def measure_cloud(vol_obj):
    """Measure cloud center and size after depsgraph evaluation.
    Returns (center_xyz, diagonal_length)."""
    from mathutils import Vector
    # Force depsgraph eval — without this, bound_box is (0,0,0)*8 in
    # --background mode because no viewport redraw ever triggers.
    deps = bpy.context.evaluated_depsgraph_get()
    vol_eval = vol_obj.evaluated_get(deps)

    bb_world = [vol_obj.matrix_world @ Vector(v) for v in vol_eval.bound_box]
    diag = max((bb_world[i] - bb_world[j]).length
               for i in range(8) for j in range(i + 1, 8))

    if diag < 0.001:
        print("[WARN] Degenerate bounds, using fallback 12m cube at origin")
        return (0.0, 0.0, 0.0), 12.0

    cx = sum(v[0] for v in bb_world) / 8
    cy = sum(v[1] for v in bb_world) / 8
    cz = sum(v[2] for v in bb_world) / 8
    return (cx, cy, cz), diag


def add_camera(cloud_center, cloud_diag):
    """Frame cloud from slightly below center — overhead-sun composition."""
    cx, cy, cz = cloud_center
    diag = cloud_diag

    cam_dist = diag * 1.4
    # Position camera well below the cloud, looking up.
    # z offset is negative (below) by ~1× the diagonal → strong upward tilt.
    cam_loc = (cx + cam_dist * 0.9, cy - cam_dist * 0.9, cz - diag * 1.0)
    bpy.ops.object.camera_add(location=cam_loc)
    cam = bpy.context.object
    cam.name = "CloudCam"
    cam.data.lens = 50
    cam.data.sensor_width = 36
    cam.data.clip_start = 0.01
    cam.data.clip_end = diag * 20   # ensure cloud is within clip range

    # Aim at cloud center.
    track = cam.constraints.new("TRACK_TO")
    empty = bpy.data.objects.new("CamTarget", None)
    bpy.context.collection.objects.link(empty)
    empty.location = (cx, cy, cz)
    track.target = empty
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'

    bpy.context.scene.camera = cam

    # EEVEE-specific: set volumetric rendering range on the scene.
    # EEVEE only renders volumes within [volumetric_start, volumetric_end]
    # along the view direction. Cloud must fit in this band.
    scene = bpy.context.scene
    if hasattr(scene, 'eevee'):
        scene.eevee.volumetric_start = max(0.01, cam_dist - diag)
        scene.eevee.volumetric_end = cam_dist + diag * 2
        print(f"[INFO] EEVEE volumetric range: "
              f"{scene.eevee.volumetric_start:.1f} to "
              f"{scene.eevee.volumetric_end:.1f}")

    print(f"[INFO] Camera at ({cam_loc[0]:.1f}, {cam_loc[1]:.1f}, {cam_loc[2]:.1f})")
    return cam


def mathutils_vec(v):
    from mathutils import Vector
    return Vector(v)


def main():
    args = parse_args()
    reset_scene()
    setup_render(args)

    vol = import_vdb(args.vdb)
    build_cloud_material(vol, args.density, args.anisotropy)

    # Rotate cloud around global Z axis. Done before measurement so
    # bounds reflect the rotated volume and the camera frames it correctly.
    vol.rotation_euler = (math.radians(90), 0, math.radians(args.rotation))
    bpy.context.view_layer.update()
    print(f"[INFO] Cloud rotation: {args.rotation}°")

    # Measure AFTER material is applied, so the grid is referenced.
    center, diag = measure_cloud(vol)
    print(f"[INFO] Cloud center ({center[0]:.1f}, {center[1]:.1f}, "
          f"{center[2]:.1f}), diagonal {diag:.1f}")

    add_overhead_sun(args.sun_strength, cloud_center=center, cloud_size=diag)
    add_bottom_fill(args.bottom_fill, cloud_center=center, cloud_size=diag)
    setup_sky(args.sky_strength)
    add_camera(center, diag)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.out))
    print(f"[OK] Saved {args.out}")

    if args.render:
        bpy.context.scene.render.filepath = os.path.abspath(args.render)
        bpy.ops.render.render(write_still=True)
        print(f"[OK] Rendered {args.render}")


if __name__ == "__main__":
    main()
