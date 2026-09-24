from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]

xml_path = (
    ROOT
    / "assets"
    / "robots"
    / "titatit"
    / "robot.xml"
)

tree = ET.parse(xml_path)
root = tree.getroot()

# --------------------------------------------------
# Remove existing actuator section if script reruns.
# --------------------------------------------------
old_actuator = root.find("actuator")

if old_actuator is not None:
    root.remove(old_actuator)

# --------------------------------------------------
# Add 16 passthrough motors.
# --------------------------------------------------
actuator = ET.SubElement(root, "actuator")

legs = ("FL", "FR", "RL", "RR")
parts = ("hip", "thigh", "calf", "foot")

for leg in legs:
    for part in parts:

        joint_name = f"{leg}_{part}_joint"

        ET.SubElement(
            actuator,
            "motor",
            {
                "name": joint_name,
                "joint": joint_name,
                "gear": "1",
            },
        )

ET.indent(tree, space="  ")

tree.write(
    xml_path,
    encoding="utf-8",
    xml_declaration=True,
)

print("Added 16 actuators to:")
print(xml_path)
