from pathlib import Path
import mujoco

ROOT = Path(__file__).resolve().parents[1]

src = (
    ROOT
    / "assets"
    / "robots"
    / "titatit"
    / "urdf"
    / "wheeled_titatit_mujoco.urdf"
)

dst = (
    ROOT
    / "assets"
    / "robots"
    / "titatit"
    / "robot.xml"
)

print("Input :", src)
print("Output:", dst)

model = mujoco.MjModel.from_xml_path(str(src))

mujoco.mj_saveLastXML(
    str(dst),
    model,
)

print("\nSaved.")

print("nbody =", model.nbody)
print("njnt  =", model.njnt)
print("nq    =", model.nq)
print("nv    =", model.nv)
print("nu    =", model.nu)
