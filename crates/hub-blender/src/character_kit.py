"""Character kits for in-game character customizers.

hub-blender runs this file in Blender with the person's own preferences (not
``--factory-startup``: MPFB is a user extension and needs them):

    blender --background --python-exit-code 3 --python character_kit.py -- job.json

job.json holds {"op": "character_kit", "output": <kit folder>, "report": <path>,
"params": {"recipe": <kit recipe>, "bodies": [<body ids>] or null}}.

A kit is what a game needs to let a player make their own character: for each
base body (say male and female) one rigged body with every slider baked in as
a morph target, and the parts that go on it (hair, outfits, shoes, hats), each
rigged to the same skeleton and with the same morphs, so a slider moves the
body and everything on it together. ``kit.json`` lists it all for the game.

The bodies come from MPFB (MakeHuman in Blender) and its CC0 assets. Each
slider is baked the way MPFB's own panel would change the character: set the
macro (weight, muscle) or the shape targets to the slider's end, refit the
clothes and the skeleton, and record where every vertex and bone went. The
recipe (see the hub's README) says which sliders and parts a kit has; a
recipe with "source": "fake" builds a small stand-in human without MPFB, for
tests.

A recipe:

    {"name": "humans",
     "texture_max": 1024, "skin_texture_max": 2048,
     "slots": {"hair":   {"label": "Hair", "kind": "hair", "optional": true, "tint": "hair"},
               "outfit": {"label": "Outfit", "kind": "clothes"}},
     "attachment_tints": {"eyebrows": "hair"},       # brows take the hair colour
     "colours": {"hair": [{"id": "black", "rgb": [0.1, 0.09, 0.08]}]},   # passed to kit.json
     "labels": {"bob01": "Bob"},                     # names for parts, by asset
     "scale": {"label": "Height", "min": 0.92, "max": 1.08},   # the game scales the whole figure
     "sliders": [
       {"id": "weight", "group": "Body", "label": "Weight", "macro": "weight", "range": [0, 1]},
       {"id": "nose_width", "group": "Face", "label": "Nose width",
        "neg": ["nose-scale-horiz-decr"], "pos": ["nose-scale-horiz-incr"]},   # targets, -1 to 1
       {"id": "bust", "label": "Bust", "macro": "cupsize", "range": [0, 1], "bodies": ["female"]}],
     "bodies": {
       "female": {"from": "../starters/human.starter_caucasian_female.json",  # an MPFB human
                  "phenotype": {"weight": 0.5}, "targets": {"head-oval": 0.3},
                  "skin": "young_caucasian_female", "skins": ["young_african_female"],
                  "eyebrows": "eyebrow010",
                  "parts": {"hair": ["bob01", "long01"], "outfit": ["female_casualsuit01"]},
                  "defaults": {"hair": "bob01", "outfit": "female_casualsuit01"}}}}

Asset names are folders in MPFB's user data (hair/, clothes/, skins/...);
"kind" says which folder a slot's parts come from.

Every job writes a JSON report: {"ok": true, "blender": "5.2.3", "kit": ...}
or {"ok": false, "error": "..."}.
"""

import json
import math
import os
import sys
import time
import traceback

import bpy  # first: bmesh and mathutils come with it
import bmesh
import mathutils
import numpy as np

KIT_FORMAT = 1
# Morphs that move nothing more than this (metres) are left out.
MIN_MOVE = 1e-5
# Bone moves smaller than this (metres) are left out of kit.json.
MIN_BONE_MOVE = 2e-4
# Target sliders run from -1 to 1 unless the recipe says otherwise.
TARGET_RANGE = (-1.0, 1.0)


def job_path():
    argv = sys.argv
    rest = argv[argv.index("--") + 1 :] if "--" in argv else argv[1:]
    if not rest:
        raise SystemExit("usage: character_kit.py -- job.json")
    return rest[0]


# ── reading a recipe ──────────────────────────────────────────────────────────


def load_recipe(path):
    with open(path, encoding="utf-8") as f:
        recipe = json.load(f)
    recipe["_dir"] = os.path.dirname(os.path.abspath(path))
    if not recipe.get("bodies"):
        raise ValueError("the recipe has no bodies")
    ids = set()
    for s in recipe.get("sliders", []):
        if "id" not in s:
            raise ValueError(f"a slider has no id: {s}")
        if s["id"] in ids:
            raise ValueError(f"two sliders are called {s['id']!r}")
        ids.add(s["id"])
        if not s.get("macro") and not (s.get("pos") or s.get("neg")):
            raise ValueError(f"slider {s['id']!r} needs a macro or pos/neg targets")
    slots = recipe.get("slots", {})
    for body_id, body in recipe["bodies"].items():
        for slot in body.get("parts", {}):
            if slot not in slots:
                raise ValueError(f"body {body_id!r} has parts in slot {slot!r}, which slots does not list")
    return recipe


def sliders_for(recipe, body_id):
    return [s for s in recipe.get("sliders", []) if not s.get("bodies") or body_id in s["bodies"]]


# ── geometry helpers (plain Blender) ──────────────────────────────────────────


def world_coords(obj):
    """Vertex positions in world space with the shape keys mixed in and the
    modifiers left off, so there is one entry per vertex of the mesh."""
    saved = [(m, m.show_viewport) for m in obj.modifiers]
    for m, _ in saved:
        m.show_viewport = False
    try:
        dg = bpy.context.evaluated_depsgraph_get()
        ev = obj.evaluated_get(dg)
        me = ev.to_mesh()
        co = np.empty(len(me.vertices) * 3, dtype=np.float64)
        me.vertices.foreach_get("co", co)
        ev.to_mesh_clear()
    finally:
        for m, s in saved:
            m.show_viewport = s
    co = co.reshape(-1, 3)
    w = np.array(obj.matrix_world, dtype=np.float64)
    return co @ w[:3, :3].T + w[:3, 3]


def bone_heads(arm):
    """Rest-pose head of every bone, in world space."""
    w = np.array(arm.matrix_world, dtype=np.float64)
    return {b.name: np.array(b.head_local[:]) @ w[:3, :3].T + w[:3, 3] for b in arm.data.bones}


def to_gltf(v):
    """Blender's Z-up to glTF's Y-up."""
    return [float(v[0]), float(v[2]), float(-v[1])]


def to_local(obj, world):
    w = np.array(obj.matrix_world.inverted(), dtype=np.float64)
    return world @ w[:3, :3].T + w[:3, 3]


def bake_shape_keys(obj, base_world, morphs):
    """Replaces the object's shape keys with a basis at ``base_world`` and one
    key per entry of ``morphs`` ({name: world-space offsets}). Returns the
    names written."""
    obj.shape_key_clear()
    obj.data.vertices.foreach_set("co", to_local(obj, base_world).astype(np.float32).ravel())
    obj.data.update()
    written = []
    if not morphs:
        return written
    obj.shape_key_add(name="Basis", from_mix=False)
    for name, delta in morphs.items():
        key = obj.shape_key_add(name=name, from_mix=False)
        key.data.foreach_set("co", to_local(obj, base_world + delta).astype(np.float32).ravel())
        key.value = 0.0
        written.append(name)
    return written


def keep_vertices(obj, keep):
    """Deletes the vertices where ``keep`` is False, keeping shape keys,
    vertex groups, UVs and attributes on the rest."""
    if keep.all():
        return
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not keep[v.index]], context="VERTS")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def drop_modifiers(obj):
    for m in list(obj.modifiers):
        if m.type != "ARMATURE":
            obj.modifiers.remove(m)


def triangles(obj):
    obj.data.calc_loop_triangles()
    return len(obj.data.loop_triangles)


def images_of(obj):
    seen = []
    for slot in obj.material_slots:
        m = slot.material
        if m and m.node_tree:
            for n in m.node_tree.nodes:
                if n.type != "TEX_IMAGE" or not n.image or n.image in seen:
                    continue
                label = (n.name + " " + n.label + " " + n.image.name).lower()
                if "normal" in label or "rough" in label or "_ao" in label:
                    continue
                seen.append(n.image)
    return seen


def pixels(img):
    a = np.empty(img.size[0] * img.size[1] * 4, dtype=np.float32)
    img.pixels.foreach_get(a)
    return a.reshape(img.size[1], img.size[0], 4)


def make_tintable(img, level=0.55):
    """Turns a hair texture grey, at an even brightness, so the game's hair
    colour (which multiplies it) decides the colour."""
    a = pixels(img)
    grey = a[..., 0] * 0.3 + a[..., 1] * 0.59 + a[..., 2] * 0.11
    seen = a[..., 3] > 0.5
    mean = float(grey[seen].mean()) if seen.any() else float(grey.mean())
    k = level / max(mean, 1e-3)
    for c in range(3):
        a[..., c] = np.clip(grey * k, 0, 1)
    img.pixels.foreach_set(a.ravel())
    img.pack()


def cap_textures(images, cap):
    for img in images:
        w, h = img.size
        if w > cap or h > cap:
            k = cap / max(w, h)
            img.scale(max(1, int(w * k)), max(1, int(h * k)))
            img.pack()


def export_glb(path, objects, arm):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in objects + [arm]:
        o.hide_set(False)
        o.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format="GLB",
        use_selection=True,
        export_apply=False,  # applying modifiers would drop the shape keys
        export_yup=True,
        export_skins=True,
        export_morph=True,
        export_morph_normal=False,  # halves the morph data; three.js recomputes nothing, looks the same
        export_try_sparse_sk=True,
        export_attributes=True,  # the _hide attribute
        export_animations=False,
        export_image_format="AUTO",  # hair and brows are cut-outs: PNG keeps their alpha
    )
    return os.path.getsize(path)


# ── sources of humans ─────────────────────────────────────────────────────────


class Human:
    """One built body: the base mesh, the meshes that are always on it
    (eyes, brows, lashes, teeth), the parts that can be swapped, and the rig."""

    def __init__(self, body, keep, attachments, parts, arm):
        self.body = body
        self.keep = keep  # vertices of body that are the person, not helpers
        self.attachments = attachments  # {name: obj}
        self.parts = parts  # {(slot, id): obj}
        self.arm = arm

    def meshes(self):
        out = {"body": self.body}
        out.update({"attachment:" + k: v for k, v in self.attachments.items()})
        out.update({"part:%s:%s" % k: v for k, v in self.parts.items()})
        return out

    def snapshot(self):
        return {
            "meshes": {k: world_coords(o) for k, o in self.meshes().items()},
            "bones": bone_heads(self.arm),
        }


class MpfbSource:
    """Humans from MPFB and MakeHuman's CC0 asset pack."""

    def __init__(self, recipe):
        from bl_ext.blender_org.mpfb.entities.objectproperties import HumanObjectProperties
        from bl_ext.blender_org.mpfb.services.humanservice import HumanService
        from bl_ext.blender_org.mpfb.services.locationservice import LocationService
        from bl_ext.blender_org.mpfb.services.targetservice import TargetService

        self.recipe = recipe
        self.HumanService = HumanService
        self.TargetService = TargetService
        self.Props = HumanObjectProperties
        self.data = LocationService.get_user_data()
        self.mpfb_targets = LocationService.get_mpfb_data("targets")
        self._target_files = None

    def describe(self):
        return "MPFB"

    def near_recipe(self, p):
        for base in (self.recipe["_dir"], os.path.dirname(self.recipe["_dir"])):
            if os.path.exists(os.path.join(base, p)):
                return os.path.join(base, p)
        raise ValueError(f"can't find {p} beside the recipe")

    def asset(self, kind, name, absolute=False):
        folder = os.path.join(self.data, kind, name)
        ext = ".mhmat" if kind == "skins" else ".mhclo"
        hits = sorted(f for f in os.listdir(folder) if f.endswith(ext)) if os.path.isdir(folder) else []
        if not hits:
            have = sorted(os.listdir(os.path.join(self.data, kind))) if os.path.isdir(os.path.join(self.data, kind)) else []
            raise ValueError(f"no {kind} called {name!r} in MPFB's assets (there are: {', '.join(have)})")
        return os.path.join(folder, hits[0]) if absolute else "%s/%s" % (name, hits[0])

    def start(self, body_id, cfg):
        with open(self.near_recipe(cfg["from"]), encoding="utf-8") as f:
            info = json.load(f)
        info["name"] = body_id
        for k, v in cfg.get("phenotype", {}).items():
            if k == "race":
                info["phenotype"]["race"].update(v)
            else:
                info["phenotype"][k] = v
        info["targets"] = [{"target": t, "value": float(w)} for t, w in cfg.get("targets", {}).items()]
        if cfg.get("skin"):
            info["skin_mhmat"] = self.asset("skins", cfg["skin"])
        for kind in ("eyes", "eyebrows", "eyelashes", "teeth"):
            if cfg.get(kind) == "none":
                info[kind] = ""
            elif cfg.get(kind):
                info[kind] = self.asset(kind, cfg[kind])
        info["hair"] = ""  # every hair goes on as a part, below
        info["clothes"] = []
        info.pop("proxy", None)

        for o in list(bpy.data.objects):
            bpy.data.objects.remove(o, do_unlink=True)
        settings = self.HumanService.get_default_deserialization_settings()
        settings.update({"override_rig": "mixamo", "override_skin_model": "GAMEENGINE", "subdiv_levels": 0})
        body = self.HumanService.deserialize_from_dict(info, settings)
        arm = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
        attachments = {}
        for o in bpy.context.scene.objects:
            if o.type == "MESH" and o is not body:
                attachments[self.kind_of(o)] = o

        parts = {}
        for slot, ids in cfg.get("parts", {}).items():
            kind = self.recipe["slots"][slot].get("kind", "clothes")
            for asset_id in ids:
                path = self.asset(kind, asset_id, absolute=True)
                obj = self.HumanService.add_mhclo_asset(
                    path,
                    body,
                    asset_type=kind,
                    subdiv_levels=0,
                    material_type="MAKESKIN",
                    alternative_materials={},
                )
                obj.name = "%s.%s.%s" % (body_id, slot, asset_id)
                parts[(slot, asset_id)] = obj
        keep = self.body_vertices(body)
        return Human(body, keep, attachments, parts, arm)

    def kind_of(self, obj):
        from bl_ext.blender_org.mpfb.entities.objectproperties import GeneralObjectProperties

        kind = str(GeneralObjectProperties.get_value("object_type", entity_reference=obj) or obj.name)
        return kind.lower()

    def body_vertices(self, body):
        groups = {g.name: g.index for g in body.vertex_groups}
        n = len(body.data.vertices)
        if "body" in groups:
            keep = np.zeros(n, bool)
            gi = groups["body"]
            for v in body.data.vertices:
                if any(g.group == gi and g.weight > 0.5 for g in v.groups):
                    keep[v.index] = True
            return keep
        helper = {groups[g] for g in ("HelperGeometry", "JointCubes") if g in groups}
        keep = np.ones(n, bool)
        for v in body.data.vertices:
            if any(g.group in helper and g.weight > 0.1 for g in v.groups):
                keep[v.index] = False
        return keep

    def target_file(self, name):
        if self._target_files is None:
            self._target_files = {}
            for root, _, files in os.walk(self.mpfb_targets):
                for f in files:
                    if f.endswith(".target.gz"):
                        self._target_files.setdefault(f[: -len(".target.gz")], os.path.join(root, f))
        if name not in self._target_files:
            raise ValueError(f"MPFB has no shape target called {name!r}")
        return self._target_files[name]

    def shape_key(self, body, name):
        keys = body.data.shape_keys
        if not keys:
            return None
        for k in keys.key_blocks:
            if k.name == name or self.TargetService.decode_shapekey_name(k.name) == name:
                return k
        return None

    def get(self, human, slider):
        """The slider's current settings, to put back afterwards."""
        if slider.get("macro"):
            return float(self.Props.get_value(slider["macro"], entity_reference=human.body))
        out = {}
        for t in slider.get("neg", []) + slider.get("pos", []):
            k = self.shape_key(human.body, t)
            out[t] = k.value if k else 0.0
        return out

    def put(self, human, slider, setting):
        if slider.get("macro"):
            self.Props.set_value(slider["macro"], float(setting), entity_reference=human.body)
            self.TargetService.reapply_macro_details(human.body)
        else:
            for t, value in setting.items():
                k = self.shape_key(human.body, t)
                if k is None and value > 0:
                    k = self.TargetService.load_target(human.body, self.target_file(t), weight=value, name=t)
                if k is not None:
                    k.value = value
        self.HumanService.refit(human.body)

    def end_setting(self, human, slider, end):
        """The setting for one end of a slider: ``end`` is "min" or "max"."""
        if slider.get("macro"):
            lo, hi = slider.get("range", (0.0, 1.0))
            return lo if end == "min" else hi
        on, off = ("pos", "neg") if end == "max" else ("neg", "pos")
        setting = {t: 0.0 for t in slider.get(off, [])}
        setting.update({t: 1.0 for t in slider.get(on, [])})
        return setting

    def delete_vertices(self, human, slot, asset_id):
        kind = self.recipe["slots"][slot].get("kind", "clothes")
        return read_delete_verts(self.asset(kind, asset_id, absolute=True))

    def skin_texture(self, name):
        mhmat = self.asset("skins", name, absolute=True)
        with open(mhmat, encoding="utf-8", errors="ignore") as f:
            for line in f:
                words = line.split()
                if len(words) >= 2 and words[0] == "diffuseTexture":
                    path = os.path.join(os.path.dirname(mhmat), " ".join(words[1:]))
                    if not os.path.exists(path):
                        path = os.path.join(os.path.dirname(mhmat), os.path.basename(words[1]))
                    return path
        raise ValueError(f"skin {name!r} has no diffuseTexture")


def read_delete_verts(mhclo):
    """The base-mesh vertices a MakeHuman clothes file hides ("delete_verts")."""
    out = []
    status = False
    with open(mhclo, encoding="utf-8", errors="ignore") as f:
        for line in f:
            words = line.split()
            if not words:
                status = False
                continue
            if words[0] == "delete_verts":
                status = True
                continue
            if not status:
                continue
            if not words[0].isnumeric():
                status = False
                continue
            run = False
            prev = None
            for w in words:
                if w == "-":
                    run = True
                    continue
                v = int(w)
                if run and prev is not None:
                    out.extend(range(prev + 1, v + 1))
                else:
                    out.append(v)
                run = False
                prev = v
    return out


class FakeSource:
    """A small stand-in human for tests: a sphere body on a three-bone rig,
    with brows, a hat and a coat, whose sliders move it in simple ways."""

    def __init__(self, recipe):
        self.recipe = recipe
        self.settings = {}
        self.base = {}

    def describe(self):
        return "a stand-in human (no MPFB)"

    def start(self, body_id, cfg):
        for o in list(bpy.data.objects):
            bpy.data.objects.remove(o, do_unlink=True)
        arm_data = bpy.data.armatures.new(body_id + ".rig")
        arm = bpy.data.objects.new(body_id + ".rig", arm_data)
        bpy.context.scene.collection.objects.link(arm)
        bpy.context.view_layer.objects.active = arm
        bpy.ops.object.mode_set(mode="EDIT")
        bones = {}
        for name, head, tail, parent in (
            ("mixamorig:Hips", (0, 0, 0.9), (0, 0, 1.1), None),
            ("mixamorig:Spine", (0, 0, 1.1), (0, 0, 1.4), "mixamorig:Hips"),
            ("mixamorig:Head", (0, 0, 1.4), (0, 0, 1.7), "mixamorig:Spine"),
        ):
            b = arm_data.edit_bones.new(name)
            b.head, b.tail = head, tail
            if parent:
                b.parent = bones[parent]
            bones[name] = b
        bpy.ops.object.mode_set(mode="OBJECT")

        def mesh_object(name, make, material_rgb):
            bm = bmesh.new()
            make(bm)
            me = bpy.data.meshes.new(name)
            bm.to_mesh(me)
            bm.free()
            me.uv_layers.new(name="UVMap")
            mat = bpy.data.materials.new(name)
            mat.use_nodes = True
            mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*material_rgb, 1)
            me.materials.append(mat)
            obj = bpy.data.objects.new(name, me)
            bpy.context.scene.collection.objects.link(obj)
            obj.parent = arm
            mod = obj.modifiers.new("Armature", "ARMATURE")
            mod.object = arm
            co = np.array([v.co[:] for v in me.vertices])
            for bone in ("mixamorig:Hips", "mixamorig:Spine", "mixamorig:Head"):
                obj.vertex_groups.new(name=bone)
            for i, p in enumerate(co):
                bone = "mixamorig:Hips" if p[2] < 1.1 else "mixamorig:Spine" if p[2] < 1.4 else "mixamorig:Head"
                obj.vertex_groups[bone].add([i], 1.0, "REPLACE")
            return obj

        def body_mesh(bm):
            bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=16, radius=0.4)
            bmesh.ops.translate(bm, verts=bm.verts, vec=(0, 0, 1.3))
            helper = bmesh.ops.create_cube(bm, size=0.05)  # stands in for MPFB's helper geometry
            bmesh.ops.translate(bm, verts=helper["verts"], vec=(0, 0, 0.2))

        body = mesh_object(body_id + ".body", body_mesh, (0.8, 0.6, 0.5))
        n = len(body.data.vertices)
        keep = np.ones(n, bool)
        keep[-8:] = False  # the cube's eight vertices
        g = body.vertex_groups.new(name="body")
        g.add([i for i in range(n) if keep[i]], 1.0, "REPLACE")

        def brows(bm):
            bmesh.ops.create_grid(bm, x_segments=2, y_segments=2, size=0.05)
            bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0), matrix=mathutils.Matrix.Rotation(math.pi / 2, 3, "X"))
            bmesh.ops.translate(bm, verts=bm.verts, vec=(0, -0.41, 1.5))

        attachments = {"eyebrows": mesh_object(body_id + ".eyebrows", brows, (0.2, 0.15, 0.1))}
        parts = {}
        for slot, ids in cfg.get("parts", {}).items():
            for k, asset_id in enumerate(ids):
                if slot == "hair":
                    def make(bm, k=k):
                        bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=6, radius=0.3 + 0.02 * k)
                        bmesh.ops.translate(bm, verts=bm.verts, vec=(0, 0, 1.65))
                else:
                    def make(bm, k=k):
                        bmesh.ops.create_cone(bm, cap_ends=False, segments=16, radius1=0.45, radius2=0.42 + 0.01 * k, depth=0.6)
                        bmesh.ops.translate(bm, verts=bm.verts, vec=(0, 0, 1.15))
                parts[(slot, asset_id)] = mesh_object("%s.%s.%s" % (body_id, slot, asset_id), make, (0.3, 0.3 + 0.1 * k, 0.6))
        human = Human(body, keep, attachments, parts, arm)
        self.base = {k: world_coords(o) for k, o in human.meshes().items()}
        self.base_bones = {b.name: (b.head_local.copy(), b.tail_local.copy()) for b in arm_data.bones}
        self.settings = {s["id"]: s.get("_base", 0.5 if s.get("macro") else 0.0) for s in self.recipe.get("sliders", [])}
        for s in self.recipe.get("sliders", []):
            if s.get("macro"):
                self.settings[s["id"]] = cfg.get("phenotype", {}).get(s["macro"], 0.5)
        self.base_settings = dict(self.settings)
        return human

    def get(self, human, slider):
        return self.settings[slider["id"]]

    def end_setting(self, human, slider, end):
        lo, hi = slider.get("range", (0.0, 1.0) if slider.get("macro") else TARGET_RANGE)
        return lo if end == "min" else hi

    def put(self, human, slider, setting):
        self.settings[slider["id"]] = setting
        self.refit(human)

    def refit(self, human):
        """Moves everything as the fake sliders say: "weight" widens the
        whole figure, "height" lifts the head and its bone, anything else
        pushes the face forward."""
        widen, lift, push = 1.0, 0.0, 0.0
        for s in self.recipe.get("sliders", []):
            d = self.settings[s["id"]] - self.base_settings[s["id"]]
            if s.get("macro") == "weight":
                widen += 0.5 * d
            elif s.get("macro") == "height":
                lift += 0.2 * d
            else:
                push += 0.05 * d
        for key, obj in human.meshes().items():
            co = self.base[key].copy()
            co[:, 0] *= widen
            high = co[:, 2] > 1.4
            co[high, 2] += lift
            face = (co[:, 2] > 1.4) & (co[:, 1] < -0.2)
            co[face, 1] -= push
            obj.shape_key_clear()
            obj.data.vertices.foreach_set("co", to_local(obj, co).astype(np.float32).ravel())
            obj.data.update()
        bpy.context.view_layer.objects.active = human.arm
        bpy.ops.object.mode_set(mode="EDIT")
        for b in human.arm.data.edit_bones:
            head, tail = self.base_bones[b.name]
            if b.name == "mixamorig:Head":
                b.head = (head[0], head[1], head[2] + lift)
                b.tail = (tail[0], tail[1], tail[2] + lift)
        bpy.ops.object.mode_set(mode="OBJECT")

    def delete_vertices(self, human, slot, asset_id):
        if slot == "hair":
            return []
        co = self.base["body"]
        return [i for i in range(len(co)) if 0.95 < co[i, 2] < 1.4]

    def skin_texture(self, name):
        img = bpy.data.images.new("fake_skin_" + name, 64, 64)
        img.pixels.foreach_set(np.tile([0.7, 0.5, 0.4, 1.0], 64 * 64).astype(np.float32))
        path = os.path.join(bpy.app.tempdir or "/tmp", "fake_skin_%s.png" % name)
        img.filepath_raw = path
        img.file_format = "PNG"
        img.save()
        return path


# ── building a kit ────────────────────────────────────────────────────────────


def save_skin(src, dst, cap):
    img = bpy.data.images.load(src, check_existing=False)
    w, h = img.size
    if w > cap or h > cap:
        k = cap / max(w, h)
        img.scale(max(1, int(w * k)), max(1, int(h * k)))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    scene = bpy.context.scene
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 90
    img.save_render(dst, scene=scene)
    bpy.data.images.remove(img)


def build_body(recipe, body_id, out_dir, source, log):
    cfg = recipe["bodies"][body_id]
    sliders = sliders_for(recipe, body_id)
    slots = recipe.get("slots", {})
    t0 = time.time()
    log(f"{body_id}: building the base body")
    human = source.start(body_id, cfg)
    base = human.snapshot()

    # Each end of each slider: set it, refit, record, put it back.
    morphs = {}  # morph name -> {mesh key: world offsets}
    bone_moves = {}  # morph name -> {bone: gltf offset}
    slider_info = {}
    for n, s in enumerate(sliders):
        before = source.get(human, s)
        info = {"neg": None, "pos": None}
        if s.get("macro"):
            lo, hi = s.get("range", (0.0, 1.0))
            info.update(base=float(before), min=float(lo), max=float(hi))
        else:
            lo, hi = s.get("range", TARGET_RANGE)
            info.update(base=0.0, min=float(lo), max=float(hi))
        for end, side in (("min", "neg"), ("max", "pos")):
            setting = source.end_setting(human, s, end)
            if s.get("macro") and abs(float(setting) - float(before)) < 1e-6:
                continue  # the body already sits at this end
            if isinstance(setting, dict) and not s.get(side):
                continue  # no targets at this end
            name = "%s%s" % (s["id"], "+" if side == "pos" else "-")
            log(f"{body_id}: slider {n + 1}/{len(sliders)} {name}")
            source.put(human, s, setting)
            snap = human.snapshot()
            source.put(human, s, before)
            moved = {}
            for key, co in snap["meshes"].items():
                d = co - base["meshes"][key]
                if d.shape == co.shape and np.abs(d).max() > MIN_MOVE:
                    moved[key] = d
            if not moved:
                continue
            morphs[name] = moved
            info[side] = name
            bones = {}
            for b, head in snap["bones"].items():
                d = head - base["bones"][b]
                if np.linalg.norm(d) > MIN_BONE_MOVE:
                    bones[b] = [round(x, 5) for x in to_gltf(d)]
            if bones:
                bone_moves[name] = bones
        slider_info[s["id"]] = info

    # Back at the base: did putting the sliders back leave anything moved?
    drift = 0.0
    final = human.snapshot()
    for key, co in final["meshes"].items():
        drift = max(drift, float(np.abs(co - base["meshes"][key]).max()))

    # Which body vertices each part hides: one bit per part that hides any.
    hide = np.zeros(len(human.body.data.vertices), np.float32)
    part_bits = {}
    bit = 0
    for (slot, asset_id) in human.parts:
        verts = source.delete_vertices(human, slot, asset_id)
        verts = [v for v in verts if v < len(hide)]
        if not verts:
            continue
        if bit >= 23:
            raise ValueError("more than 23 parts hide body vertices; split the kit")
        for v in verts:
            hide[v] = float(int(hide[v]) | (1 << bit))
        part_bits[(slot, asset_id)] = 1 << bit
        bit += 1

    # Bake: basis at the base shape, one key per morph, helpers removed.
    meshes = human.meshes()
    for key, obj in meshes.items():
        drop_modifiers(obj)
        bake_shape_keys(obj, base["meshes"][key], {m: d[key] for m, d in morphs.items() if key in d})
    attr = human.body.data.attributes.get("_hide") or human.body.data.attributes.new("_hide", "FLOAT", "POINT")
    attr.data.foreach_set("value", hide)
    keep_vertices(human.body, human.keep)

    # Textures: hair (and brows) grey so the game tints them; sizes capped.
    tints = {}
    for (slot, asset_id), obj in human.parts.items():
        if slots[slot].get("tint"):
            for img in images_of(obj):
                make_tintable(img)
    for name, obj in human.attachments.items():
        if name in recipe.get("attachment_tints", {}):
            tints[name] = recipe["attachment_tints"][name]
            for img in images_of(obj):
                make_tintable(img)
    skin_imgs = images_of(human.body)
    cap_textures(skin_imgs, int(recipe.get("skin_texture_max", 2048)))
    cap_textures([i for i in bpy.data.images if i not in skin_imgs], int(recipe.get("texture_max", 1024)))

    # Files.
    rel = lambda p: os.path.relpath(p, out_dir).replace(os.sep, "/")
    body_path = os.path.join(out_dir, body_id, "body.glb")
    body_objects = [human.body] + list(human.attachments.values())
    for obj in human.parts.values():
        obj.hide_set(True)
    size = export_glb(body_path, body_objects, human.arm)
    log(f"{body_id}: wrote {rel(body_path)} ({size // 1024} KB)")
    entry = {
        "file": rel(body_path),
        "bytes": size,
        "triangles": sum(triangles(o) for o in body_objects),
        "body_mesh": human.body.name,
        "bones": len(human.arm.data.bones),
        "sliders": slider_info,
        "bone_moves": bone_moves,
        "attachments": {name: {"mesh": o.name, **({"tint": tints[name]} if name in tints else {})} for name, o in human.attachments.items()},
        "skins": [],
        "parts": {},
        "default": cfg.get("defaults", {}),
        "drift_m": drift,
    }
    for skin in [cfg.get("skin")] + [s for s in cfg.get("skins", []) if s != cfg.get("skin")]:
        if not skin:
            continue
        if skin == cfg.get("skin"):
            entry["skins"].append({"id": skin, "file": None})
            continue
        dst = os.path.join(out_dir, body_id, "skins", skin + ".jpg")
        save_skin(source.skin_texture(skin), dst, int(recipe.get("skin_texture_max", 2048)))
        entry["skins"].append({"id": skin, "file": rel(dst)})

    for (slot, asset_id), obj in human.parts.items():
        path = os.path.join(out_dir, body_id, slot, asset_id + ".glb")
        for o in human.parts.values():
            o.hide_set(o is not obj)
        size = export_glb(path, [obj], human.arm)
        part = {"id": asset_id, "file": rel(path), "bytes": size, "triangles": triangles(obj), "mesh": obj.name}
        if (slot, asset_id) in part_bits:
            part["hides"] = part_bits[(slot, asset_id)]
        if slots[slot].get("tint"):
            part["tint"] = slots[slot]["tint"]
        label = recipe.get("labels", {}).get(asset_id)
        if label:
            part["label"] = label
        entry["parts"].setdefault(slot, []).append(part)
    log(f"{body_id}: done in {time.time() - t0:.0f} s, {len(morphs)} morphs, {len(human.parts)} parts")
    return entry


def build_kit(recipe, out_dir, source, bodies=None, log=print):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    wanted = bodies or list(recipe["bodies"])
    for b in wanted:
        if b not in recipe["bodies"]:
            raise ValueError(f"the recipe has no body called {b!r} (it has {', '.join(recipe['bodies'])})")
    kit_path = os.path.join(out_dir, "kit.json")
    kit = {}
    if bodies and os.path.exists(kit_path):  # rebuilding some bodies keeps the others
        with open(kit_path, encoding="utf-8") as f:
            kit = json.load(f)
    kit.update(
        {
            "kit": KIT_FORMAT,
            "name": recipe.get("name", "kit"),
            "units": "metres, Y-up, as glTF",
            "built_with": "Blender %s, %s" % (bpy.app.version_string, source.describe()),
            "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sliders": [
                {k: v for k, v in s.items() if k in ("id", "group", "label", "bodies", "unit")}
                for s in recipe.get("sliders", [])
            ],
            "scale": recipe.get("scale", {"label": "Height", "min": 0.9, "max": 1.1}),
            "slots": recipe.get("slots", {}),
            "colours": recipe.get("colours", {}),
        }
    )
    kit.setdefault("bodies", {})
    for b in wanted:
        kit["bodies"][b] = build_body(recipe, b, out_dir, source, log)
        with open(kit_path, "w", encoding="utf-8") as f:  # after each body, so a long build keeps its work
            json.dump(kit, f, indent=1)
    return {
        "kit": kit_path,
        "seconds": time.time() - t0,
        "bodies": [
            {
                "name": b,
                "triangles": kit["bodies"][b]["triangles"],
                "morphs": sum(1 for s in kit["bodies"][b]["sliders"].values() for side in ("neg", "pos") if s[side]),
                "parts": sum(len(v) for v in kit["bodies"][b]["parts"].values()),
                "drift_mm": kit["bodies"][b]["drift_m"] * 1000,
            }
            for b in wanted
        ],
    }


def main():
    with open(job_path(), encoding="utf-8") as f:
        job = json.load(f)
    report = {"ok": False, "blender": bpy.app.version_string}
    try:
        if job.get("op") == "version":
            report["ok"] = True
        else:
            params = job["params"]
            recipe = load_recipe(params["recipe"])
            source = FakeSource(recipe) if recipe.get("source") == "fake" else MpfbSource(recipe)
            report.update(build_kit(recipe, job["output"], source, params.get("bodies")))
            report["ok"] = True
    except ImportError as e:
        report["error"] = (
            f"MPFB is not available in this Blender ({e}). Install the MPFB extension and MakeHuman's "
            "CC0 system assets pack in desktop Blender, and run the hub with that Blender (HUB_BLENDER)"
        )
        report["trace"] = traceback.format_exc()
    except Exception as e:  # reported to the hub as a readable message
        report["error"] = f"{type(e).__name__}: {e}"
        report["trace"] = traceback.format_exc()
    with open(job["report"], "w", encoding="utf-8") as f:
        json.dump(report, f)
    if not report["ok"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
