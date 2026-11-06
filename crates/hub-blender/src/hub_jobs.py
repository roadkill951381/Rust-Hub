"""Rust-Hub mesh jobs for Blender.

hub-blender runs this file in the background, one job per process:

    blender --background --factory-startup --python-exit-code 3 \
        --python hub_jobs.py -- job.json

or, with Blender's Python module installed (`pip install bpy`):

    python hub_jobs.py -- job.json

job.json holds {"op", "input", "output", "report", "params"}. The script reads
the input GLB (metres, Y-up), runs the operation on its meshes joined into one,
writes the output GLB, and always writes a JSON report:
{"ok": true, "blender": "5.2.3", "triangles": 1234} or {"ok": false, "error": "..."}.
"""

import json
import math
import sys
import traceback

import bpy


def job_path():
    argv = sys.argv
    rest = argv[argv.index("--") + 1 :] if "--" in argv else argv[1:]
    if not rest:
        raise SystemExit("usage: hub_jobs.py -- job.json")
    return rest[0]


def load(path):
    """Imports a GLB and returns its meshes joined into one object, transforms applied."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=path)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise ValueError("the input has no meshes")
    for o in meshes:
        world = o.matrix_world.copy()
        o.parent = None
        o.data.transform(world)
        o.matrix_world.identity()
    target = meshes[0]
    if len(meshes) > 1:
        with bpy.context.temp_override(
            active_object=target,
            object=target,
            selected_objects=meshes,
            selected_editable_objects=meshes,
        ):
            bpy.ops.object.join()
    for o in list(bpy.context.scene.objects):
        if o != target:
            bpy.data.objects.remove(o, do_unlink=True)
    weld(target.data)
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    return target


def weld(mesh):
    """Merges vertices at the same position.

    glTF splits vertices along UV and normal seams, which Blender would treat
    as cuts. Blender keeps UVs per face corner, so welding keeps the UVs and
    closes the seams. The tolerance matches the hub's: a millionth of the size.
    """
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(mesh)
    if bm.verts:
        lo = [min(v.co[i] for v in bm.verts) for i in range(3)]
        hi = [max(v.co[i] for v in bm.verts) for i in range(3)]
        diagonal = math.dist(lo, hi)
        bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=max(diagonal * 1e-6, 1e-9))
    bm.to_mesh(mesh)
    bm.free()


def apply_modifier(obj, kind, **settings):
    """Adds a modifier, bakes its result into the mesh, and removes it."""
    mod = obj.modifiers.new(name="hub", type=kind)
    for key, value in settings.items():
        setattr(mod, key, value)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    baked = bpy.data.meshes.new_from_object(obj.evaluated_get(depsgraph))
    obj.modifiers.clear()
    old = obj.data
    obj.data = baked
    bpy.data.meshes.remove(old)


def voxel_remesh(obj, params):
    apply_modifier(
        obj,
        "REMESH",
        mode="VOXEL",
        voxel_size=float(params["voxel"]),
        adaptivity=0.0,
        use_smooth_shade=True,
    )


def decimate(obj, params):
    apply_modifier(
        obj,
        "DECIMATE",
        decimate_type="COLLAPSE",
        ratio=float(params["ratio"]),
        use_collapse_triangulate=True,
    )


def uv_unwrap(obj, params):
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(
        angle_limit=math.radians(float(params["angle_deg"])),
        island_margin=float(params["margin"]),
        area_weight=0.0,
        correct_aspect=True,
        scale_to_bounds=False,
    )
    bpy.ops.object.mode_set(mode="OBJECT")


OPS = {"voxel_remesh": voxel_remesh, "decimate": decimate, "uv_unwrap": uv_unwrap}


def triangles(obj):
    obj.data.calc_loop_triangles()
    return len(obj.data.loop_triangles)


def save(path):
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format="GLB",
        use_selection=False,
        export_apply=True,
        export_yup=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="NONE",
        export_animations=False,
    )


def main():
    with open(job_path(), encoding="utf-8") as f:
        job = json.load(f)
    report = {"ok": False, "blender": bpy.app.version_string}
    try:
        op = job["op"]
        if op != "version":
            if op not in OPS:
                raise ValueError(f"unknown operation {op!r}")
            obj = load(job["input"])
            OPS[op](obj, job.get("params", {}))
            report["triangles"] = triangles(obj)
            save(job["output"])
        report["ok"] = True
    except Exception as e:  # reported to the hub as a readable message
        report["error"] = f"{type(e).__name__}: {e}"
        report["trace"] = traceback.format_exc()
    with open(job["report"], "w", encoding="utf-8") as f:
        json.dump(report, f)
    if not report["ok"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
