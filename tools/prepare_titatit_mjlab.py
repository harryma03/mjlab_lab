#!/usr/bin/env python3

from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]

SRC = (
    ROOT
    / "assets"
    / "robots"
    / "titatit"
    / "mujoco"
    / "wheeled_titatit_rl.xml"
)

DST = (
    ROOT
    / "assets"
    / "robots"
    / "titatit"
    / "mujoco"
    / "wheeled_titatit_mjlab.xml"
)


# ============================================================
# 1. Official TITATIT joint naming
# ============================================================

JOINT_MAP = {
    # Front Right
    "F_joint_right_leg_1": "FR_hip_joint",
    "F_joint_right_leg_2": "FR_thigh_joint",
    "F_joint_right_leg_3": "FR_calf_joint",
    "F_joint_right_leg_4": "FR_foot_joint",

    # Front Left
    "F_joint_left_leg_1": "FL_hip_joint",
    "F_joint_left_leg_2": "FL_thigh_joint",
    "F_joint_left_leg_3": "FL_calf_joint",
    "F_joint_left_leg_4": "FL_foot_joint",

    # Rear Right
    "R_joint_right_leg_1": "RR_hip_joint",
    "R_joint_right_leg_2": "RR_thigh_joint",
    "R_joint_right_leg_3": "RR_calf_joint",
    "R_joint_right_leg_4": "RR_foot_joint",

    # Rear Left
    "R_joint_left_leg_1": "RL_hip_joint",
    "R_joint_left_leg_2": "RL_thigh_joint",
    "R_joint_left_leg_3": "RL_calf_joint",
    "R_joint_left_leg_4": "RL_foot_joint",
}


# actuator name -> official actuator name
ACTUATOR_MAP = {
    "FR_hip": "FR_hip_joint",
    "FR_thigh": "FR_thigh_joint",
    "FR_calf": "FR_calf_joint",
    "FR_foot": "FR_foot_joint",

    "FL_hip": "FL_hip_joint",
    "FL_thigh": "FL_thigh_joint",
    "FL_calf": "FL_calf_joint",
    "FL_foot": "FL_foot_joint",

    "RR_hip": "RR_hip_joint",
    "RR_thigh": "RR_thigh_joint",
    "RR_calf": "RR_calf_joint",
    "RR_foot": "RR_foot_joint",

    "RL_hip": "RL_hip_joint",
    "RL_thigh": "RL_thigh_joint",
    "RL_calf": "RL_calf_joint",
    "RL_foot": "RL_foot_joint",
}


# ============================================================
# 2. Make a fresh copy every time
# ============================================================

shutil.copy2(SRC, DST)

tree = ET.parse(DST)
root = tree.getroot()


# ============================================================
# 3. Fix base/trunk inertial parameters
#
# Official quadruped-wheel-titatit-rl URDF:
#
# mass = 26.085 kg
# COM  = (9.657E-05, 4.946E-06, 0.018311)
#
# inertia:
#   ixx = 0.13241
#   ixy = -1.5238E-05
#   ixz = 2.9159E-07
#   iyy = 0.21427
#   iyz = 3.83E-07
#   izz = 0.29638
#
# ============================================================

worldbody = root.find("worldbody")

if worldbody is None:
    raise RuntimeError("No <worldbody> found")

base = None

for body in worldbody.iter("body"):
    if body.attrib.get("name") == "base_link":
        base = body
        break

if base is None:
    raise RuntimeError('Could not find body name="base_link"')


# Remove existing inertial if this script is rerun on an edited file.
old_inertial = base.find("inertial")

if old_inertial is not None:
    base.remove(old_inertial)


official_inertial = ET.Element(
    "inertial",
    {
        "pos": "9.657e-05 4.946e-06 0.018311",
        "mass": "26.085",
        "fullinertia": (
            "0.13241 "
            "0.21427 "
            "0.29638 "
            "-1.5238e-05 "
            "2.9159e-07 "
            "3.83e-07"
        ),
    },
)


# Insert immediately after freejoint.
insert_index = 0

for i, child in enumerate(list(base)):
    if child.tag == "freejoint":
        insert_index = i + 1
        break

base.insert(insert_index, official_inertial)


# ============================================================
# 4. Rename all physical joints
# ============================================================

for joint in root.iter("joint"):

    old_name = joint.attrib.get("name")

    if old_name in JOINT_MAP:
        joint.attrib["name"] = JOINT_MAP[old_name]


# ============================================================
# 5. Update EVERY joint reference
#
# This covers:
#   actuator motor joint=
#   jointpos
#   jointvel
#   jointactuatorfrc
# etc.
# ============================================================

for elem in root.iter():

    if "joint" in elem.attrib:

        old_joint = elem.attrib["joint"]

        if old_joint in JOINT_MAP:
            elem.attrib["joint"] = JOINT_MAP[old_joint]


# ============================================================
# 6. Rename actuator names to match joint names
#
# This makes MjLab D1-style regex work:
#
# .*_(hip|thigh|calf)_joint
# .*_foot_joint
#
# ============================================================

# Original XML actuators are removed below; MjLab creates them dynamically.


# ============================================================
# 7. Remove original XML actuators
#
# MjLab DcMotorActuatorCfg / IdealPdActuatorCfg dynamically
# create their own motor actuators. Keeping the original
# <actuator> section would create duplicate actuators.
# ============================================================

xml_actuator = root.find("actuator")

if xml_actuator is not None:
    root.remove(xml_actuator)


# ============================================================
# 7. Remove original XML actuators
#
# MjLab DcMotorActuatorCfg / IdealPdActuatorCfg dynamically
# create their own motor actuators. Keeping the original
# <actuator> section would create duplicate actuators.
# ============================================================

xml_actuator = root.find("actuator")

if xml_actuator is not None:
    root.remove(xml_actuator)


# ============================================================
# 7. Pretty print and save
# ============================================================

ET.indent(tree, space="  ")

tree.write(
    DST,
    encoding="utf-8",
    xml_declaration=False,
)

print()
print("Generated MjLab TITATIT model:")
print(DST)
print()
print("Applied:")
print("  [1] official trunk mass/inertia")
print("  [2] official 16 joint names")
print("  [3] actuator joint references")
print("  [4] sensor joint references")
print("  [5] actuator names aligned with joint names")
print()
