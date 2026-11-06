"""Rust-Hub mesh jobs for Blender.

hub-blender runs this file in the background, one job per process:

    blender --background --factory-startup --python-exit-code 3 \
        --python hub_jobs.py -- job.json

or, with Blender's Python module installed (`pip install bpy`):

    python hub_jobs.py -- job.json

job.json holds {"op", "input", "output", "report", "params"}. The mesh
operations read the input GLB (metres, Y-up), run on its meshes joined into
one, and write the output GLB. ``rig`` instead fits the input model to a
rigged reference character (see ``rig``). Every job writes a JSON report:
{"ok": true, "blender": "5.2.3", "triangles": 1234} or {"ok": false, "error": "..."}.
"""

import json
import math
import struct
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


# --- Rigging a new model like an existing character -------------------------
#
# The reference GLB's skeleton is kept byte for byte: the output is the
# reference file with its skinned meshes swapped for the new model, so bone
# names, rest rotations, inverse bind matrices and any animations stay exactly
# as the game expects. Blender only works out the new model's skin weights:
#
# 1. Scale and move the model onto the reference's rest pose, turning it round
#    if its feet point the other way.
# 2. Measure how far each arm sits from the reference's arms (generated
#    A-poses vary), and pose the reference to match.
# 3. Copy skin weights from the posed reference onto the model, then use those
#    weights to move the model's arms back into the reference's rest pose.
# 4. Split the model into skin and clothes materials by colour, named the way
#    the reference names its materials (e.g. ``surv_mei.body``).


def rig(job):
    import numpy as np
    from mathutils import Matrix

    params = job.get("params", {})
    reference = params["reference"]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=reference)
    scene = bpy.context.scene
    armatures = [o for o in scene.objects if o.type == "ARMATURE"]
    if len(armatures) != 1:
        raise ValueError(
            f"the reference must have exactly one skeleton; {reference} has {len(armatures)}"
        )
    arm = armatures[0]
    skinned = [
        o
        for o in scene.objects
        if o.type == "MESH" and any(m.type == "ARMATURE" and m.object == arm for m in o.modifiers)
    ]
    if not skinned:
        raise ValueError(f"{reference} has no meshes skinned to its skeleton")
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    prefix = params.get("materials") or body_prefix(skinned) or arm.name

    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=job["input"])
    added = [o for o in bpy.data.objects if o not in before]
    model = join_world([o for o in added if o.type == "MESH"], "model")
    for o in added:
        if o != model and o.name in bpy.data.objects:
            bpy.data.objects.remove(o, do_unlink=True)
    weld(model.data)
    bpy.context.view_layer.update()

    parts = [triangles_of(o) for o in skinned]
    offsets = np.cumsum([0] + [len(p) for p, _ in parts])
    ref_points = surface_points(
        np.concatenate([p for p, _ in parts]),
        np.concatenate([t + offsets[k] for k, (_, t) in enumerate(parts)]),
    )
    report = {}
    turned = fit_to(model, ref_points, params.get("facing", "auto"))
    report["turned"] = turned

    target = params.get("triangles")
    if target and triangles(model) > int(target):
        decimate(model, {"ratio": int(target) / triangles(model)})

    poses = fit_arms(arm, model)
    report["arm_offsets_deg"] = {side: round(math.degrees(a), 1) for side, (_, a) in poses.items()}
    for _side, (bone, angle) in poses.items():
        rotate_bone(arm, bone, angle)
    bpy.context.view_layer.update()

    source = evaluated_copy(skinned, arm)
    transfer_weights(source, model)
    bpy.data.objects.remove(source, do_unlink=True)

    # Back to the rest pose: deform the model by the opposite arm rotations.
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    for _side, (bone, angle) in poses.items():
        rotate_bone(arm, bone, -angle)
    bpy.context.view_layer.update()
    mod = model.modifiers.new("pose", "ARMATURE")
    mod.object = arm
    depsgraph = bpy.context.evaluated_depsgraph_get()
    posed = bpy.data.meshes.new_from_object(
        model.evaluated_get(depsgraph), preserve_all_data_layers=True, depsgraph=depsgraph
    )
    model.modifiers.clear()
    old = model.data
    model.data = posed
    bpy.data.meshes.remove(old)
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)

    skin, colour_note = skin_faces(model, arm)
    report["skin_fraction"] = round(float(skin.mean()), 3) if len(skin) else 0.0
    if colour_note:
        report["note"] = colour_note

    names = [f"{prefix}.body", f"{prefix}.clothes"]
    write_rigged(reference, job["input"], job["output"], model, skin, names)
    report["materials"] = names
    report["bones"] = len(arm.data.bones)
    report["triangles"] = triangles(model)
    return report


def body_prefix(meshes):
    for o in meshes:
        for slot in o.material_slots:
            if slot.material and slot.material.name.endswith(".body"):
                return slot.material.name[: -len(".body")]
    return None


def join_world(objects, name):
    """Bakes transforms into the meshes and joins them into one object."""
    if not objects:
        raise ValueError("the model has no meshes")
    for o in objects:
        world = o.matrix_world.copy()
        o.parent = None
        o.data.transform(world)
        o.matrix_world.identity()
    target = objects[0]
    if len(objects) > 1:
        with bpy.context.temp_override(
            active_object=target,
            object=target,
            selected_objects=objects,
            selected_editable_objects=objects,
        ):
            bpy.ops.object.join()
    target.name = name
    return target


def world_points(obj):
    """Vertex positions in world space, deformed by the current pose."""
    import numpy as np

    depsgraph = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(depsgraph)
    mesh = ev.to_mesh()
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    ev.to_mesh_clear()
    co = co.reshape(-1, 3)
    m = np.array(obj.matrix_world, dtype=np.float32)
    return co @ m[:3, :3].T + m[:3, 3]


def mesh_points(obj):
    import numpy as np

    co = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def feet_forward(points):
    """How far the feet reach along Y beyond the shins: negative when the toes
    point to -Y, Blender's front (glTF's +Z). ``points`` should cover the
    surface evenly (see ``surface_points``)."""
    z = points[:, 2]
    lo, hi = z.min(), z.max()
    h = hi - lo
    feet = points[z < lo + 0.06 * h]
    shins = points[(z > lo + 0.15 * h) & (z < lo + 0.28 * h)]
    if len(feet) == 0 or len(shins) == 0:
        return 0.0
    return float(feet[:, 1].mean() - shins[:, 1].mean())


def surface_points(points, faces, count=20000):
    """Random points spread evenly over a triangle mesh's surface."""
    import numpy as np

    tri = points[faces]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if area.sum() <= 0:
        return points
    rng = np.random.default_rng(7)
    pick = rng.choice(len(tri), size=count, p=area / area.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[pick]
    return t[:, 0] + (t[:, 1] - t[:, 0]) * u[:, None] + (t[:, 2] - t[:, 0]) * v[:, None]


def triangles_of(obj, world=True):
    """(points, triangle indices) of an object as currently posed."""
    import numpy as np

    depsgraph = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(depsgraph)
    mesh = ev.to_mesh()
    mesh.calc_loop_triangles()
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
    mesh.loop_triangles.foreach_get("vertices", tris)
    ev.to_mesh_clear()
    co = co.reshape(-1, 3)
    if world:
        m = np.array(obj.matrix_world, dtype=np.float32)
        co = co @ m[:3, :3].T + m[:3, 3]
    return co, tris.reshape(-1, 3)


def fit_to(model, ref_points, facing):
    """Scales the model to the reference's height and stands it on the same
    spot; turns it round when its feet point the other way. Returns whether it
    was turned."""
    import numpy as np
    from mathutils import Matrix

    pts = mesh_points(model)
    turned = False
    if facing == "turn":
        turned = True
    elif facing == "auto":
        mine = feet_forward(surface_points(*triangles_of(model)))
        theirs = feet_forward(ref_points)
        turned = mine * theirs < 0
    if turned:
        model.data.transform(Matrix.Rotation(math.pi, 4, "Z"))
        pts = mesh_points(model)
    lo, hi = pts.min(0), pts.max(0)
    rlo, rhi = ref_points.min(0), ref_points.max(0)
    scale = float((rhi[2] - rlo[2]) / max(hi[2] - lo[2], 1e-9))
    centre = (lo + hi) / 2
    rcentre = (rlo + rhi) / 2
    move = Matrix.Translation((rcentre[0], rcentre[1], rlo[2]))
    shift = Matrix.Translation((-centre[0], -centre[1], -lo[2]))
    model.data.transform(move @ Matrix.Scale(scale, 4) @ shift)
    model.data.update()
    del np
    return turned


def arm_bones(arm):
    """(upper arm, chain end) bone names per side, found by the usual names
    (Mixamo's LeftArm, MakeHuman's upperarm_l, Rigify's upper_arm.L)."""
    found = {}
    for b in arm.data.bones:
        n = b.name.lower().replace("mixamorig:", "").replace("_", "").replace(".", "")
        for side, marks in (("left", ("left", "l")), ("right", ("right", "r"))):
            if n in (f"{marks[0]}arm", f"upperarm{marks[1]}", f"upperarm{marks[0]}"):
                found[side] = b.name
    return found


def chain_points(arm, bone_name, samples=24):
    """World points along a bone and its first-child chain to the hand."""

    mw = arm.matrix_world
    bone = arm.pose.bones[bone_name]
    chain = [bone]
    while len(chain) < 3 and chain[-1].children:
        chain.append(max(chain[-1].children, key=lambda c: c.length))
    points = []
    for pb in chain:
        head, tail = mw @ pb.head, mw @ pb.tail
        for i in range(samples // len(chain)):
            t = (i + 0.5) / (samples // len(chain))
            points.append(head.lerp(tail, t))
    return mw @ bone.head, points


def fit_arms(arm, model):
    """For each arm, the rotation (radians, about the front-to-back axis
    through the shoulder) that best puts the reference's arm bones inside the
    model's arm."""
    from mathutils import Matrix
    from mathutils.bvhtree import BVHTree

    bvh = BVHTree.FromObject(model, bpy.context.evaluated_depsgraph_get())
    pts = mesh_points(model)
    height = float(pts[:, 2].max() - pts[:, 2].min())
    cap = 0.03 * height
    out = {}
    for side, name in arm_bones(arm).items():
        pivot, points = chain_points(arm, name)

        def score(angle, pivot=pivot, points=points):
            rot = Matrix.Rotation(angle, 3, "Y")
            total = 0.0
            for p in points:
                q = pivot + rot @ (p - pivot)
                loc, normal, _i, dist = bvh.find_nearest(q)
                if loc is None:
                    continue
                inside = (q - loc).dot(normal) < 0
                total += min(dist, cap) if inside else -min(dist, 3 * cap)
            return total

        best = max(range(-40, 41, 2), key=lambda d: (score(math.radians(d)), -abs(d)))
        fine = max(
            (best + x / 2 for x in range(-4, 5)),
            key=lambda d: (score(math.radians(d)), -abs(d)),
        )
        out[side] = (name, math.radians(fine))
    return out


def rotate_bone(arm, bone_name, angle):
    """Turns a pose bone (and its children) about the world front-to-back
    axis through its head."""
    from mathutils import Matrix

    bpy.context.view_layer.update()  # pose matrices are stale after edits
    pb = arm.pose.bones[bone_name]
    mw = arm.matrix_world
    pivot = mw @ pb.head
    world = Matrix.Translation(pivot) @ Matrix.Rotation(angle, 4, "Y") @ Matrix.Translation(-pivot)
    pb.matrix = mw.inverted() @ world @ mw @ pb.matrix
    bpy.context.view_layer.update()


def evaluated_copy(objects, arm):
    """One mesh object: the skinned meshes as currently posed, in world space,
    keeping their vertex groups."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    copies = []
    for o in objects:
        mesh = bpy.data.meshes.new_from_object(
            o.evaluated_get(depsgraph), preserve_all_data_layers=True, depsgraph=depsgraph
        )
        mesh.transform(o.matrix_world)
        copy = bpy.data.objects.new(o.name + "-source", mesh)
        bpy.context.scene.collection.objects.link(copy)
        for g in o.vertex_groups:
            copy.vertex_groups.new(name=g.name)
        copies.append(copy)
    target = copies[0]
    if len(copies) > 1:
        with bpy.context.temp_override(
            active_object=target,
            object=target,
            selected_objects=copies,
            selected_editable_objects=copies,
        ):
            bpy.ops.object.join()
    bones = {b.name for b in arm.data.bones}
    for g in list(target.vertex_groups):
        if g.name not in bones:
            target.vertex_groups.remove(g)
    return target


def transfer_weights(source, model):
    for g in source.vertex_groups:
        model.vertex_groups.new(name=g.name)
    mod = model.modifiers.new("weights", "DATA_TRANSFER")
    mod.object = source
    mod.use_vert_data = True
    mod.data_types_verts = {"VGROUP_WEIGHTS"}
    mod.vert_mapping = "POLYINTERP_NEAREST"
    mod.layers_vgroup_select_src = "ALL"
    mod.layers_vgroup_select_dst = "NAME"
    # Writing the GLB keeps each vertex's four strongest bones and normalizes them.
    with bpy.context.temp_override(object=model, active_object=model):
        bpy.ops.object.modifier_apply(modifier=mod.name)


def base_texture(obj):
    """The model's base colour image as an (h, w, 4) float array, or None."""
    import numpy as np

    for slot in obj.material_slots:
        mat = slot.material
        if not mat or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                links = node.inputs["Base Color"].links
                if links and links[0].from_node.type == "TEX_IMAGE":
                    image = links[0].from_node.image
                    if image is None:
                        continue
                    w, h = image.size
                    px = np.empty(w * h * 4, dtype=np.float32)
                    image.pixels.foreach_get(px)
                    return px.reshape(h, w, 4)
    return None


def face_colours(obj, image):
    """Each face's texture colour at its UV centre, as an (n, 3) array."""
    import numpy as np

    mesh = obj.data
    uv_layer = mesh.uv_layers.active
    n = len(mesh.polygons)
    if uv_layer is None:
        return None
    uvs = np.empty(len(mesh.loops) * 2, dtype=np.float32)
    uv_layer.data.foreach_get("uv", uvs)
    uvs = uvs.reshape(-1, 2)
    start = np.empty(n, dtype=np.int64)
    count = np.empty(n, dtype=np.int64)
    mesh.polygons.foreach_get("loop_start", start)
    mesh.polygons.foreach_get("loop_total", count)
    sums = np.add.reduceat(uvs, start, axis=0) if n else np.zeros((0, 2))
    centre = sums / count[:, None]
    h, w = image.shape[:2]
    x = np.clip((centre[:, 0] % 1.0) * (w - 1), 0, w - 1).astype(np.int64)
    y = np.clip((centre[:, 1] % 1.0) * (h - 1), 0, h - 1).astype(np.int64)
    return image[y, x, :3]


def skin_faces(obj, arm):
    """Which faces show skin: those coloured like the middle of the face.
    Returns (bool per face, a note when it could not tell)."""
    import numpy as np
    from mathutils import Vector

    n = len(obj.data.polygons)
    image = base_texture(obj)
    colours = face_colours(obj, image) if image is not None else None
    if colours is None:
        return np.zeros(n, dtype=bool), "the model has no texture, so it is all one material"
    head = next(
        (b for b in arm.data.bones if b.name.lower().replace("mixamorig:", "") == "head"), None
    )
    if head is None:
        return np.zeros(n, dtype=bool), "no Head bone to find the face by"
    mw = arm.matrix_world
    top = mw @ head.tail_local
    base = mw @ head.head_local
    aim = base.lerp(top, 0.35)
    toward = Vector((0.0, 1.0, 0.0))
    hit, _loc, _normal, index = obj.ray_cast(aim - 2.0 * toward, toward)
    if not hit:
        return np.zeros(n, dtype=bool), "could not find the face to sample skin colour"
    tone = colours[index]

    def chroma(c):
        total = c.sum(axis=-1, keepdims=True) + 1e-6
        return c[..., :2] / total, total[..., 0]

    want, bright = chroma(tone)
    got, level = chroma(colours)
    close = np.linalg.norm(got - want, axis=1) < 0.035
    lit = (level > bright * 0.55) & (level < bright * 1.6)
    skin = close & lit
    # Smooth out speckles: a face follows the majority of its neighbours.
    edges = {}
    for poly in obj.data.polygons:
        for key in poly.edge_keys:
            edges.setdefault(key, []).append(poly.index)
    neighbours = [[] for _ in range(n)]
    for faces in edges.values():
        for a in faces:
            neighbours[a].extend(f for f in faces if f != a)
    for _ in range(2):
        votes = np.array([skin[nb].mean() if nb else skin[i] for i, nb in enumerate(neighbours)])
        skin = np.where(votes > 0.5, True, np.where(votes < 0.5, False, skin))
    return skin, None


# --- Writing the rigged GLB --------------------------------------------------


def read_glb(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, _version, _length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValueError(f"{path} is not a binary glTF (.glb)")
    offset, doc, binary = 12, None, b""
    while offset < len(data):
        size, kind = struct.unpack_from("<I4s", data, offset)
        chunk = data[offset + 8 : offset + 8 + size]
        if kind == b"JSON":
            doc = json.loads(chunk)
        elif kind == b"BIN\x00":
            binary = chunk
        offset += 8 + size
    if doc is None:
        raise ValueError(f"{path} has no glTF JSON")
    return doc, binary


def view_bytes(doc, binary, index):
    view = doc["bufferViews"][index]
    start = view.get("byteOffset", 0)
    return binary[start : start + view["byteLength"]]


class GlbWriter:
    """Builds a new binary buffer, copying only what is still used."""

    def __init__(self):
        self.blob = bytearray()
        self.views = []

    def add(self, data, target=None, stride=None):
        while len(self.blob) % 4:
            self.blob.append(0)
        view = {"buffer": 0, "byteOffset": len(self.blob), "byteLength": len(data)}
        if target:
            view["target"] = target
        if stride:
            view["byteStride"] = stride
        self.blob.extend(data)
        self.views.append(view)
        return len(self.views) - 1


def matrix_of(node):
    import numpy as np

    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    t = node.get("translation", [0, 0, 0])
    r = node.get("rotation", [0, 0, 0, 1])
    s = node.get("scale", [1, 1, 1])
    x, y, z, w = r
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    m = np.eye(4)
    m[:3, :3] = rot * np.array(s)
    m[:3, 3] = t
    return m


def world_matrices(doc):
    import numpy as np

    nodes = doc.get("nodes", [])
    parent = {}
    for i, n in enumerate(nodes):
        for c in n.get("children", []):
            parent[c] = i
    cache = {}

    def world(i):
        if i not in cache:
            local = matrix_of(nodes[i])
            cache[i] = world(parent[i]) @ local if i in parent else local
        return cache[i]

    return [world(i) for i in range(len(nodes))], parent, np


def accessor_array(doc, binary, index):
    import numpy as np

    acc = doc["accessors"][index]
    comps = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[acc["type"]]
    dtype = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}[
        acc["componentType"]
    ]
    view = doc["bufferViews"][acc["bufferView"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    count = acc["count"]
    stride = view.get("byteStride")
    item = np.dtype(dtype).itemsize * comps
    if stride and stride != item:
        rows = [
            np.frombuffer(binary, dtype=dtype, count=comps, offset=start + k * stride)
            for k in range(count)
        ]
        return np.array(rows)
    return np.frombuffer(binary, dtype=dtype, count=count * comps, offset=start).reshape(
        count, comps
    )


def model_arrays(obj, joint_of):
    """Per-corner vertex data for glTF: positions and normals converted to
    Y-up, UVs flipped, four joints and weights, and triangles per material."""
    import numpy as np

    mesh = obj.data
    mesh.calc_loop_triangles()
    tris = mesh.loop_triangles
    nt = len(tris)
    loops = np.empty(nt * 3, dtype=np.int64)
    tris.foreach_get("loops", loops)
    verts = np.empty(nt * 3, dtype=np.int64)
    tris.foreach_get("vertices", verts)
    polys = np.empty(nt, dtype=np.int64)
    tris.foreach_get("polygon_index", polys)
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    normals = np.empty(len(mesh.loops) * 3, dtype=np.float32)
    mesh.corner_normals.foreach_get("vector", normals) if hasattr(
        mesh, "corner_normals"
    ) else mesh.loops.foreach_get("normal", normals)
    normals = normals.reshape(-1, 3)
    uv_layer = mesh.uv_layers.active
    if uv_layer is not None:
        uvs = np.empty(len(mesh.loops) * 2, dtype=np.float32)
        uv_layer.data.foreach_get("uv", uvs)
        uvs = uvs.reshape(-1, 2)
    else:
        uvs = np.zeros((len(mesh.loops), 2), dtype=np.float32)

    joints = np.zeros((len(mesh.vertices), 4), dtype=np.uint16)
    weights = np.zeros((len(mesh.vertices), 4), dtype=np.float32)
    names = {g.index: g.name for g in obj.vertex_groups}
    for v in mesh.vertices:
        pairs = sorted(
            (
                (g.weight, joint_of[names[g.group]])
                for g in v.groups
                if g.weight > 0 and names.get(g.group) in joint_of
            ),
            reverse=True,
        )[:4]
        total = sum(w for w, _ in pairs)
        for k, (w, j) in enumerate(pairs):
            joints[v.index, k] = j
            weights[v.index, k] = w / total
    unweighted = int((weights.sum(1) == 0).sum())

    def to_gltf(v):
        return np.stack([v[:, 0], v[:, 2], -v[:, 1]], axis=1)

    corner_pos = to_gltf(co[verts])
    corner_nrm = to_gltf(normals[loops])
    corner_uv = uvs[loops] * np.array([1.0, -1.0], dtype=np.float32) + np.array(
        [0.0, 1.0], dtype=np.float32
    )
    key = np.concatenate(
        [verts[:, None].astype(np.float64), np.round(corner_uv, 5), np.round(corner_nrm, 3)],
        axis=1,
    )
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    return {
        "position": corner_pos[first].astype(np.float32),
        "normal": corner_nrm[first].astype(np.float32),
        "uv": corner_uv[first].astype(np.float32),
        "joints": joints[verts[first]],
        "weights": weights[verts[first]],
        "indices": inverse.reshape(-1, 3).astype(np.uint32),
        "face_of_triangle": polys,
        "unweighted": unweighted,
    }


def write_rigged(reference, model_path, output, obj, skin, names):
    import numpy as np

    doc, binary = read_glb(reference)
    nodes = doc["nodes"]
    skinned_nodes = [i for i, n in enumerate(nodes) if "mesh" in n and "skin" in n]
    if not skinned_nodes:
        raise ValueError(f"{reference} has no skinned meshes")
    skin_index = nodes[skinned_nodes[0]]["skin"]
    skin_doc = doc["skins"][skin_index]
    joint_nodes = skin_doc["joints"]
    joint_of = {nodes[j].get("name", f"node{j}"): k for k, j in enumerate(joint_nodes)}

    # Where rest-pose world positions land in the mesh's own space: with the
    # usual inverse bind matrices this is the identity.
    worlds, _parent, _ = world_matrices(doc)
    ibm = (
        accessor_array(doc, binary, skin_doc["inverseBindMatrices"])
        .reshape(-1, 4, 4)
        .transpose(0, 2, 1)
        if "inverseBindMatrices" in skin_doc
        else np.tile(np.eye(4), (len(joint_nodes), 1, 1))
    )
    bind = worlds[joint_nodes[0]] @ ibm[0]
    to_mesh = np.linalg.inv(bind)

    data = model_arrays(obj, joint_of)
    pos = data["position"] @ to_mesh[:3, :3].T.astype(np.float32) + to_mesh[:3, 3].astype(
        np.float32
    )
    nrm = data["normal"] @ np.linalg.inv(to_mesh[:3, :3]).astype(np.float32)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)

    # The model's own texture, straight from its file.
    mdoc, mbin = read_glb(model_path)
    picture = None
    for mat in mdoc.get("materials", []):
        tex = mat.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        if tex is not None:
            image = mdoc["images"][mdoc["textures"][tex["index"]]["source"]]
            picture = (
                view_bytes(mdoc, mbin, image["bufferView"]),
                image.get("mimeType", "image/png"),
            )
            break

    out = GlbWriter()
    old_views = {}

    def keep_view(index):
        if index not in old_views:
            v = doc["bufferViews"][index]
            old_views[index] = out.add(
                view_bytes(doc, binary, index), v.get("target"), v.get("byteStride")
            )
        return old_views[index]

    # Keep every accessor something other than the old meshes uses.
    used = set()
    for s in doc.get("skins", []):
        if "inverseBindMatrices" in s:
            used.add(s["inverseBindMatrices"])
    for a in doc.get("animations", []):
        for sampler in a.get("samplers", []):
            used.update((sampler["input"], sampler["output"]))
    accessors, acc_map = [], {}
    for i in sorted(used):
        acc = dict(doc["accessors"][i])
        if "bufferView" in acc:
            acc["bufferView"] = keep_view(acc["bufferView"])
        acc_map[i] = len(accessors)
        accessors.append(acc)
    for s in doc.get("skins", []):
        if "inverseBindMatrices" in s:
            s["inverseBindMatrices"] = acc_map[s["inverseBindMatrices"]]
    for a in doc.get("animations", []):
        for sampler in a.get("samplers", []):
            sampler["input"] = acc_map[sampler["input"]]
            sampler["output"] = acc_map[sampler["output"]]

    def add_accessor(array, kind, component, target=None, minmax=False):
        array = np.ascontiguousarray(array)
        view = out.add(array.tobytes(), target)
        acc = {
            "bufferView": view,
            "componentType": component,
            "count": int(array.shape[0]),
            "type": kind,
        }
        if minmax:
            acc["min"] = array.min(0).tolist()
            acc["max"] = array.max(0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    attributes = {
        "POSITION": add_accessor(pos, "VEC3", 5126, 34962, minmax=True),
        "NORMAL": add_accessor(nrm, "VEC3", 5126, 34962),
        "TEXCOORD_0": add_accessor(data["uv"], "VEC2", 5126, 34962),
        "JOINTS_0": add_accessor(data["joints"], "VEC4", 5123, 34962),
        "WEIGHTS_0": add_accessor(data["weights"], "VEC4", 5126, 34962),
    }

    images, textures, samplers = [], [], []
    if picture:
        view = out.add(picture[0])
        images.append({"bufferView": view, "mimeType": picture[1]})
        samplers.append({"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071})
        textures.append({"sampler": 0, "source": 0})
    materials, primitives = [], []
    face_skin = skin[data["face_of_triangle"]]
    for k, name in enumerate(names):
        tris = data["indices"][face_skin == (k == 0)]
        if len(tris) == 0:
            continue
        pbr = {"metallicFactor": 0.0, "roughnessFactor": 0.85 if k == 0 else 0.95}
        if picture:
            pbr["baseColorTexture"] = {"index": 0}
        materials.append({"name": name, "pbrMetallicRoughness": pbr})
        primitives.append(
            {
                "attributes": attributes,
                "indices": add_accessor(tris.reshape(-1), "SCALAR", 5125, 34963),
                "material": len(materials) - 1,
                "mode": 4,
            }
        )

    # Swap the old skinned meshes for the new one, under the first one's parent.
    first = skinned_nodes[0]
    keep_node = {
        "name": names[0].rsplit(".", 1)[0],
        "mesh": 0,
        "skin": skin_index,
    }
    remove = set(skinned_nodes)
    # Other meshes (eyes, hair cards) skinned or attached to the old body go too.
    remove.update(i for i, n in enumerate(nodes) if "mesh" in n)
    renumber, new_nodes = {}, []
    for i, n in enumerate(nodes):
        if i in remove and i != first:
            continue
        renumber[i] = len(new_nodes)
        if i == first:
            n = {**{k: v for k, v in n.items() if k not in ("mesh", "skin")}, **keep_node}
        new_nodes.append(n)
    for n in new_nodes:
        if "children" in n:
            n["children"] = [renumber[c] for c in n["children"] if c in renumber]
            if not n["children"]:
                del n["children"]
    for s in doc.get("skins", []):
        s["joints"] = [renumber[j] for j in s["joints"]]
        if "skeleton" in s:
            s["skeleton"] = renumber[s["skeleton"]]
    for sc in doc.get("scenes", []):
        sc["nodes"] = [renumber[i] for i in sc.get("nodes", []) if i in renumber]
    for a in doc.get("animations", []):
        a["channels"] = [c for c in a["channels"] if c["target"].get("node") in renumber]
        for c in a["channels"]:
            c["target"]["node"] = renumber[c["target"]["node"]]

    doc["nodes"] = new_nodes
    doc["meshes"] = [{"name": keep_node["name"], "primitives": primitives}]
    doc["accessors"] = accessors
    doc["bufferViews"] = out.views
    doc["buffers"] = [{"byteLength": len(out.blob)}]
    doc["materials"] = materials
    for key, value in (("images", images), ("textures", textures), ("samplers", samplers)):
        if value:
            doc[key] = value
        else:
            doc.pop(key, None)
    doc.setdefault("asset", {})["generator"] = "Rust-Hub (rigged from a reference)"
    for key in ("extensionsUsed", "extensionsRequired"):
        doc.pop(key, None)

    text = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 4)
    blob = bytes(out.blob) + b"\x00" * (-len(out.blob) % 4)
    total = 12 + 8 + len(text) + 8 + len(blob)
    with open(output, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(text), b"JSON"))
        f.write(text)
        f.write(struct.pack("<I4s", len(blob), b"BIN\x00"))
        f.write(blob)
    if data["unweighted"]:
        raise ValueError(f"{data['unweighted']} vertices got no skin weights")


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
        if op == "rig":
            report.update(rig(job))
        elif op != "version":
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
