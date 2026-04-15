#!/usr/bin/env python3
"""Render VDB volumes with Blender (headless, Cycles CPU).

Camera is auto-fitted to the first VDB's bounding box; subsequent files
are rendered from the same viewpoint for side-by-side comparison.

Usage:
    python render_vdbs.py volume.vdb
    python render_vdbs.py a.vdb b.vdb c.vdb --resolution 1920 1080 --samples 256
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

BLENDER = "/workspace/bin/blender-4.5.8-linux-x64/blender"
LD_LIBRARY_PATH_EXTRA = (
    "/home/coder/micromamba/envs/openvdb/lib:"
    "/home/coder/micromamba/envs/openvdb/x86_64-conda-linux-gnu/sysroot/usr/lib64"
)

BLENDER_SCRIPT = textwrap.dedent(r'''
import bpy
import mathutils
import sys
import os
import math
import json


def setup_scene(res_x, res_y, samples):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.volume_step_rate = 0.25
    scene.cycles.volume_max_steps = 4096
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Camera: positioned later by first VDB's bounding box
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 35
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    # Key light
    sun = bpy.data.lights.new("Sun", type='SUN')
    sun.energy = 3.0
    sun_obj = bpy.data.objects.new("Sun", sun)
    scene.collection.objects.link(sun_obj)
    sun_obj.rotation_euler = (math.radians(45), 0, math.radians(30))

    # Fill light
    fill = bpy.data.lights.new("Fill", type='SUN')
    fill.energy = 1.0
    fill_obj = bpy.data.objects.new("Fill", fill)
    scene.collection.objects.link(fill_obj)
    fill_obj.rotation_euler = (math.radians(60), 0, math.radians(-120))

    # Dark background
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.02, 0.02, 0.04, 1.0)


def create_volume_material():
    mat = bpy.data.materials.new("Volume")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    output = nodes.new('ShaderNodeOutputMaterial')
    principled = nodes.new('ShaderNodeVolumePrincipled')
    principled.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    principled.inputs["Density"].default_value = 500.0
    principled.inputs["Anisotropy"].default_value = 0.0
    mat.node_tree.links.new(principled.outputs["Volume"], output.inputs["Volume"])
    return mat


def fit_camera(vol_obj, bbox=None):
    """Fit camera to the volume's bounding box.

    If *bbox* is given as ((min_x,min_y,min_z),(max_x,max_y,max_z)) it is
    used directly (e.g. computed from the source VDB index space, which
    is what Cycles actually renders even when vol.dimensions reports a
    transformed extent).
    """
    scene = bpy.context.scene
    cam = scene.camera

    if bbox is not None:
        bb_min = mathutils.Vector(bbox[0])
        bb_max = mathutils.Vector(bbox[1])
    else:
        bb = [mathutils.Vector(c) for c in vol_obj.bound_box]
        bb_min = mathutils.Vector(min(v[i] for v in bb) for i in range(3))
        bb_max = mathutils.Vector(max(v[i] for v in bb) for i in range(3))
    center = (bb_min + bb_max) / 2
    max_dim = max(bb_max - bb_min)

    # Place camera so the volume fills ~60% of the frame
    # Use sensor geometry to compute exact distance for any resolution
    aspect = scene.render.resolution_x / scene.render.resolution_y
    cam.data.lens = 35
    sensor_w = cam.data.sensor_width
    hfov = 2 * math.atan(sensor_w / (2 * cam.data.lens))
    vfov = 2 * math.atan(math.tan(hfov / 2) / aspect)
    min_half_fov = min(hfov, vfov) / 2
    dist = (max_dim / 2) / math.tan(min_half_fov) * 1.1

    cam.location = center + mathutils.Vector((-dist * 0.70, -dist * 0.60, dist * 0.2))
    cam.rotation_euler = (center - cam.location).to_track_quat('-Z', 'Y').to_euler()


def render_vdb(vdb_path, output_path, material):
    bpy.ops.object.volume_import(filepath=vdb_path)
    vol = bpy.context.active_object
    if vol is None:
        for obj in bpy.data.objects:
            if obj.type == 'VOLUME':
                vol = obj
                break
    if vol is None:
        print(f"ERROR: Could not load {vdb_path}")
        return

    if vol.data.materials:
        vol.data.materials[0] = material
    else:
        vol.data.materials.append(material)

    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.ops.render.render(write_still=True)
    print(f"Rendered: {output_path}")

    bpy.data.objects.remove(vol, do_unlink=True)
    for block in bpy.data.volumes:
        if block.users == 0:
            bpy.data.volumes.remove(block)


config = json.loads(sys.argv[sys.argv.index("--") + 1])
os.makedirs(config["out_dir"], exist_ok=True)
setup_scene(config["res_x"], config["res_y"], config["samples"])
material = create_volume_material()

# Fit camera to the LARGEST VDB (by file size as proxy for voxel count)
vdb_files = config["files"]
largest = max(vdb_files, key=lambda p: os.path.getsize(p))
bpy.ops.object.volume_import(filepath=largest)
fit_vol = bpy.context.active_object
if fit_vol is None:
    for obj in bpy.data.objects:
        if obj.type == 'VOLUME':
            fit_vol = obj
            break
if fit_vol is not None:
    fit_camera(fit_vol)
    bpy.data.objects.remove(fit_vol, do_unlink=True)
    for block in bpy.data.volumes:
        if block.users == 0:
            bpy.data.volumes.remove(block)

for vdb_path in vdb_files:
    name = os.path.splitext(os.path.basename(vdb_path))[0]
    out = os.path.join(config["out_dir"], name + ".png")
    print(f"\n{'='*60}\nRendering: {name}\n{'='*60}")
    render_vdb(vdb_path, out, material)

print(f"\nDone! {len(vdb_files)} renderings saved to {config['out_dir']}/")
''')


def omnitree_to_vdb_file(
    descriptor_file: str,
    values_file: str,
    out_dir: str,
    out_name: str,
    bbox_min=None,
    per_dim_levels=None,
    grid_name: str = "density",
    background: float = 0.0,
) -> str:
    """Reconstruct a VDB from a saved omnitree descriptor + leaf values.

    *descriptor_file* is the `_<dim>d.bin` file written by
    `RefinementDescriptor.to_file`; *values_file* is the matching `.npy`
    of leaf scalings.  Writes `<out_dir>/<out_name>.vdb` and returns its path.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np
    import openvdb as vdb
    import dyada.descriptor
    import dyada.discretization
    import dyada.linearization
    from compare_cloud import omnitree_to_vdb

    descriptor = dyada.descriptor.RefinementDescriptor.from_file(descriptor_file)
    discretization = dyada.discretization.Discretization(
        dyada.linearization.MortonOrderLinearization(), descriptor
    )
    leaf_values = np.load(values_file)

    if per_dim_levels is None:
        per_dim_levels = np.asarray(descriptor.get_maximum_level())
    else:
        per_dim_levels = np.asarray(per_dim_levels)
    if bbox_min is None:
        bbox_min = np.zeros(descriptor.get_num_dimensions(), dtype=np.int64)
    else:
        bbox_min = np.asarray(bbox_min)

    grid = omnitree_to_vdb(
        discretization, leaf_values, per_dim_levels, bbox_min,
        grid_name=grid_name, background=background,
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name + ".vdb")
    vdb.write(out_path, grids=[grid])
    print(f"  Reconstructed omnitree -> {out_path} "
          f"({len(descriptor)} nodes, {len(leaf_values)} leaves, "
          f"levels={per_dim_levels.tolist()})")
    return out_path


def _resolve_input(path: str) -> str:
    """If *path* is an omnitree descriptor `.bin`, reconstruct to a VDB beside it.

    Otherwise return *path* unchanged.  The matching values file is found by
    replacing the `_<dim>d.bin` suffix with `_values.npy`.  The resulting VDB
    is written next to the `.bin` with the same stem.
    """
    import re
    m = re.search(r"_(\d+)d\.bin$", path)
    if not m:
        return path
    values_path = path[: m.start()] + "_values.npy"
    if not os.path.exists(values_path):
        raise FileNotFoundError(
            f"Expected leaf-values file beside descriptor: {values_path}"
        )
    out_dir = os.path.dirname(path)
    out_name = os.path.splitext(os.path.basename(path))[0]
    return omnitree_to_vdb_file(path, values_path, out_dir, out_name)


def _ensure_float_vdb(vdb_path: str, tmp_dir: str) -> str:
    """If a VDB contains only BoolGrids, convert to FloatGrid (density=1.0).

    Returns the original path if already float, or a temp path with the
    converted grid.
    """
    try:
        import openvdb as vdb
    except ImportError:
        return vdb_path

    grids, _meta = vdb.readAll(vdb_path)
    needs_conversion = all(isinstance(g, vdb.BoolGrid) for g in grids)
    if not needs_conversion:
        return vdb_path

    float_grids = []
    for g in grids:
        fg = vdb.FloatGrid(0.0)
        fg.name = "density"
        fg.transform = g.transform
        for item in g.iterOnValues():
            fg.fill(item.min, item.max, 1.0)
        float_grids.append(fg)

    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, os.path.basename(vdb_path))
    vdb.write(tmp_path, grids=float_grids)
    print(f"  Converted BoolGrid -> FloatGrid: {tmp_path}")
    return tmp_path


def main():
    parser = argparse.ArgumentParser(description="Render VDB volumes with Blender (headless, Cycles CPU)")
    parser.add_argument("files", nargs="+", help="VDB files to render")
    parser.add_argument("--resolution", nargs=2, type=int, default=[1024, 780],
                        metavar=("W", "H"))
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="./vdb_renders")
    args = parser.parse_args()

    # Resolve inputs: reconstruct omnitree _<dim>d.bin files into temp VDBs,
    # and convert BoolGrid VDBs to FloatGrid for Blender compatibility.
    tmp_dir = "/tmp/render_vdbs_converted"
    resolved = [_resolve_input(str(Path(f).resolve())) for f in args.files]
    vdb_files = [_ensure_float_vdb(p, tmp_dir) for p in resolved]
    print(f"Will render {len(vdb_files)} VDB files:")
    for f in vdb_files:
        print(f"  {f}")

    config = json.dumps({
        "files": vdb_files,
        "out_dir": args.output_dir,
        "res_x": args.resolution[0],
        "res_y": args.resolution[1],
        "samples": args.samples,
    })

    script_path = "/tmp/blender_render_vdb.py"
    with open(script_path, "w") as f:
        f.write(BLENDER_SCRIPT)

    env = os.environ.copy()
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH_EXTRA + (":" + existing if existing else "")

    result = subprocess.run([BLENDER, "--background", "--python", script_path, "--", config], env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
