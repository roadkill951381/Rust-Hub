"""Builds a rigged reference character and an unrigged 'generated' model,
for the rigging test in `blender_jobs.rs`.

blender -b --python rig_fixtures.py -- OUTDIR [arm_deg_ref] [arm_deg_model] [turn]

The reference is a capsule figure on a mixamorig-named skeleton in an A-pose,
with a short idle action. The model is fatter, holds its arms at another
angle, is textured skin on the head and hands and dark elsewhere, and can face
the other way, like a generated character.
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
obj = bpy.data.objects.new("generated", me)
bpy.context.scene.collection.objects.link(obj)
img = bpy.data.images.new("tex", 64, 64)
px = []
for _y in range(64):
    for x in range(64):
        px += [0.80, 0.60, 0.50, 1.0] if x < 32 else [0.12, 0.12, 0.14, 1.0]
img.pixels = px
img.pack()
obj.data.materials.append(material("generated", image=img))
uv = me.uv_layers.new()
for poly in me.polygons:
    z = sum(me.vertices[v].co.z for v in poly.vertices) / len(poly.vertices)
    xs = sum(me.vertices[v].co.x for v in poly.vertices) / len(poly.vertices)
    skin = z > 1.5 or abs(xs) > 0.55
    for li in poly.loop_indices:
        uv.data[li].uv = (0.25 if skin else 0.75, 0.5)
if turn:
    me.transform(Matrix.Rotation(math.pi, 4, "Z"))
# Generators make models of arbitrary size; shrink it to a unit-ish box.
me.transform(Matrix.Scale(1.1, 4))
bpy.ops.export_scene.gltf(filepath=f"{out}/model.glb", export_format="GLB")
print("fixtures written", out)
