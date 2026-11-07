"""Builds a rigged reference character and an unrigged 'generated' model,
for the rigging test in `blender_jobs.rs`.

blender -b --python rig_fixtures.py -- OUTDIR [arm_deg_ref] [arm_deg_model] [turn] [beard]

The reference is a capsule figure on a mixamorig-named skeleton in an A-pose,
with a short idle action. The model is fatter, holds its arms at another
angle, is textured skin on the head and hands and dark elsewhere, and can face
the other way, like a generated character. With ``beard``, its skin is dark
brown, its clothes dark grey, and a darker beard covers the lower face.
"""

import math
import sys

import bpy

# isort: split
# bmesh and mathutils load only after bpy.
import bmesh
from mathutils import Matrix, Vector

args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
out = args[0]
ref_angle = float(args[1]) if len(args) > 1 else 47.0
model_angle = float(args[2]) if len(args) > 2 else 32.0
turn = len(args) > 3 and args[3] == "turn"
beard = len(args) > 4 and args[4] == "beard"

BONES = {
    # name: (head, tail, parent)
    "mixamorig:Hips": ((0, 0, 0.95), (0, 0, 1.05), None),
    "mixamorig:Spine": ((0, 0, 1.05), (0, 0, 1.30), "mixamorig:Hips"),
    "mixamorig:Neck": ((0, 0, 1.45), (0, 0, 1.55), "mixamorig:Spine"),
    "mixamorig:Head": ((0, 0, 1.55), (0, 0, 1.75), "mixamorig:Neck"),
    "mixamorig:LeftUpLeg": ((0.1, 0, 0.95), (0.1, 0, 0.5), "mixamorig:Hips"),
    "mixamorig:LeftLeg": ((0.1, 0, 0.5), (0.1, 0, 0.08), "mixamorig:LeftUpLeg"),
    "mixamorig:LeftFoot": ((0.1, 0, 0.08), (0.1, -0.15, 0.02), "mixamorig:LeftLeg"),
    "mixamorig:RightUpLeg": ((-0.1, 0, 0.95), (-0.1, 0, 0.5), "mixamorig:Hips"),
    "mixamorig:RightLeg": ((-0.1, 0, 0.5), (-0.1, 0, 0.08), "mixamorig:RightUpLeg"),
    "mixamorig:RightFoot": ((-0.1, 0, 0.08), (-0.1, -0.15, 0.02), "mixamorig:RightLeg"),
}


def arm_bones(angle):
    a = math.radians(angle)
    out = {}
    for side, sx in (("Left", 1), ("Right", -1)):
        sh = Vector((sx * 0.2, 0, 1.42))
        d = Vector((sx * math.cos(a), 0, -math.sin(a)))
        el = sh + d * 0.28
        wr = el + d * 0.25
        hd = wr + d * 0.1
        out[f"mixamorig:{side}Arm"] = (sh, el, "mixamorig:Spine")
        out[f"mixamorig:{side}ForeArm"] = (el, wr, f"mixamorig:{side}Arm")
        out[f"mixamorig:{side}Hand"] = (wr, hd, f"mixamorig:{side}ForeArm")
    return out


def capsule_body(bones, scale=1.0):
    """A body made of rounded cylinders along the bones, plus a foot box."""
    bm = bmesh.new()
    radii = {
        "Hips": 0.16,
        "Spine": 0.15,
        "Neck": 0.05,
        "Head": 0.1,
        "UpLeg": 0.075,
        "Leg": 0.055,
        "Foot": 0.045,
        "Arm": 0.045,
        "ForeArm": 0.038,
        "Hand": 0.035,
    }
    for name, (h, t, _p) in bones.items():
        key = name.split(":")[1].replace("Left", "").replace("Right", "")
        r = radii[key] * scale
        h, t = Vector(h), Vector(t)
        length = (t - h).length
        mid = (h + t) / 2
        rot = (t - h).to_track_quat("Z", "Y").to_matrix().to_4x4()
        res = bmesh.ops.create_cone(
            bm, cap_ends=True, segments=16, radius1=r, radius2=r, depth=length + r
        )
        bmesh.ops.transform(bm, matrix=Matrix.Translation(mid) @ rot, verts=res["verts"])
    me = bpy.data.meshes.new("body")
    bm.to_mesh(me)
    bm.free()
    return me


def build_armature(bones):
    arm_data = bpy.data.armatures.new("Armature")
    arm = bpy.data.objects.new("Armature", arm_data)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for name, (h, t, _p) in bones.items():
        eb = arm_data.edit_bones.new(name)
        eb.head, eb.tail = Vector(h), Vector(t)
    for name, (_h, _t, p) in bones.items():
        if p:
            arm_data.edit_bones[name].parent = arm_data.edit_bones[p]
    bpy.ops.object.mode_set(mode="OBJECT")
    return arm


def material(name, colour=None, image=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    if image is not None:
        tex = m.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = image
        m.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    elif colour:
        bsdf.inputs["Base Color"].default_value = (*colour, 1)
    return m


# Reference: rigged, A-pose at ref_angle, with a small idle action.
bpy.ops.wm.read_factory_settings(use_empty=True)
bones = {**BONES, **arm_bones(ref_angle)}
arm = build_armature(bones)
body = bpy.data.objects.new("surv_test_body", capsule_body(bones))
bpy.context.scene.collection.objects.link(body)
body.data.materials.append(material("surv_test.body", (0.8, 0.6, 0.5)))
body.select_set(True)
arm.select_set(True)
bpy.context.view_layer.objects.active = arm
with bpy.context.temp_override(
    active_object=arm, selected_objects=[body, arm], selected_editable_objects=[body, arm]
):
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
arm.animation_data_create()
action = bpy.data.actions.new("idle")
arm.animation_data.action = action
pb = arm.pose.bones["mixamorig:Head"]
pb.rotation_mode = "QUATERNION"
for frame, angle in ((1, 0.0), (20, 0.2)):
    pb.rotation_quaternion = Matrix.Rotation(angle, 4, "X").to_quaternion()
    pb.keyframe_insert("rotation_quaternion", frame=frame)
bpy.ops.export_scene.gltf(
    filepath=f"{out}/reference.glb", export_format="GLB", export_animations=True, export_skins=True
)

# Model: unrigged, thicker, arms at model_angle, textured: skin tone on head
# and hands, dark clothes elsewhere; optionally facing the other way.
bpy.ops.wm.read_factory_settings(use_empty=True)
bones = {**BONES, **arm_bones(model_angle)}
me = capsule_body(bones, scale=1.15)
# A dark lock of hair hanging down the middle of the face, in front of it, as
# generated characters often have: skin detection must not take its colour.
strand = bmesh.new()
strand.from_mesh(me)
bmesh.ops.create_cube(
    strand,
    size=1.0,
    matrix=Matrix.Translation((0.0, -0.132, 1.65)) @ Matrix.Diagonal((0.024, 0.01, 0.3, 1.0)),
)
strand.to_mesh(me)
strand.free()
obj = bpy.data.objects.new("generated", me)
bpy.context.scene.collection.objects.link(obj)
img = bpy.data.images.new("tex", 256, 256)
SKIN = (0.35, 0.26, 0.22) if beard else (0.80, 0.60, 0.50)
CLOTHES = (0.25, 0.245, 0.245) if beard else (0.12, 0.12, 0.14)
BEARD = (0.16, 0.11, 0.09)
px = []
for y in range(256):
    for x in range(256):
        # The beard: the front of the head (u up to 0.23), chin to nose
        # (v 0.23 to 0.44; see the UVs below).
        if beard and x < 60 and 60 <= y <= 112:
            px += [*BEARD, 1.0]
        else:
            px += [*SKIN, 1.0] if x < 128 else [*CLOTHES, 1.0]
img.pixels = px
img.pack()
obj.data.materials.append(material("generated", image=img))
uv = me.uv_layers.new()


def clip(t):
    return min(max(t, 0.0), 1.0)


for poly in me.polygons:
    z = sum(me.vertices[v].co.z for v in poly.vertices) / len(poly.vertices)
    xs = sum(me.vertices[v].co.x for v in poly.vertices) / len(poly.vertices)
    for li, v in zip(poly.loop_indices, poly.vertices, strict=True):
        co = me.vertices[v].co
        if min(me.vertices[w].co.y for w in poly.vertices) < -0.125:
            uv.data[li].uv = (0.75, 0.5)  # the lock of hair
        elif z > 1.5:
            # The head gets its own texels, the front and back side by side,
            # so a face can be painted onto it.
            u = (0.0 if poly.normal.y < 0 else 0.25) + 0.02 + 0.21 * clip((co.x + 0.15) / 0.3)
            uv.data[li].uv = (u, 0.02 + 0.96 * clip((co.z - 1.4) / 0.45))
        else:
            uv.data[li].uv = (0.24 if abs(xs) > 0.55 else 0.75, 0.5)
if turn:
    me.transform(Matrix.Rotation(math.pi, 4, "Z"))
# Generators make models of arbitrary size; shrink it to a unit-ish box.
me.transform(Matrix.Scale(1.1, 4))
bpy.ops.export_scene.gltf(filepath=f"{out}/model.glb", export_format="GLB")
print("fixtures written", out)
