"""Convert ZG-M1 URDF to a floating-base MJCF while preserving visual meshes."""

from __future__ import annotations

import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import mujoco

import trimesh

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/robots/zg_m1/urdf/zg_m1_urdf.urdf"
OUT = ROOT / "assets/robots/zg_m1/mujoco/zg_m1.xml"
VISUAL_MESH_DIR = OUT.parent / "meshes"


def _quat_from_rpy(rpy: str) -> str:
    roll, pitch, yaw = map(float, rpy.split())
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return f"{cr * cp * cy + sr * sp * sy:.12g} {sr * cp * cy - cr * sp * sy:.12g} {cr * sp * cy + sr * cp * sy:.12g} {cr * cp * sy - sr * sp * cy:.12g}"


def _visual_mesh_paths(filename: str) -> list[str]:
    source = (URDF.parent / filename).resolve()
    raw = source.read_bytes()
    is_ascii_stl = raw.lstrip().lower().startswith(b"solid")
    is_binary_stl = not is_ascii_stl and len(raw) >= 84 and 84 + 50 * struct.unpack_from("<I", raw, 80)[0] == len(raw)
    if is_binary_stl and struct.unpack_from("<I", raw, 80)[0] <= 190_000:
        return [filename]

    VISUAL_MESH_DIR.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load_mesh(source, force="mesh", process=False)
    pieces = [mesh] if len(mesh.faces) <= 190_000 else [
        trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[start:start + 190_000], process=False)
        for start in range(0, len(mesh.faces), 190_000)
    ]
    paths = []
    for index, piece in enumerate(pieces):
        suffix = "" if len(pieces) == 1 else f"_{index}"
        converted = VISUAL_MESH_DIR / f"{source.stem}{suffix}{source.suffix}"
        converted.write_bytes(trimesh.exchange.stl.export_stl(piece))
        paths.append(f"meshes/{converted.name}")
    return paths

def main() -> None:
    # MuJoCo imports the URDF's collision model and folds fixed rotor links into
    # their parent bodies.  Add the required floating base before serializing.
    spec = mujoco.MjSpec.from_file(str(URDF))
    base = spec.body("BASE_LINK")
    base.add_freejoint(name="floating_base_joint")
    # Sensor meshes are authored as collision tags in the supplied URDF; they
    # should be visible-only and must not create phantom contacts.
    for geom in base.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
            geom.contype = geom.conaffinity = 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    physics_tmp = OUT.with_name(f"{OUT.stem}.physics.tmp.xml")
    spec.compile()
    spec.to_file(str(physics_tmp))
    mjcf = ET.parse(physics_tmp).getroot()
    physics_tmp.unlink()
    for geom in mjcf.findall(".//geom"):
        geom.set("rgba", "0 0 0 0")
        geom.set("group", "3")

    assets = mjcf.find("asset")
    urdf = ET.parse(URDF).getroot()
    source_visual_count = visual_geom_count = 0
    for link in urdf.findall("link"):
        visual = link.find("visual")
        mesh = None if visual is None else visual.find("geometry/mesh")
        if mesh is None:
            continue
        name = link.attrib["name"]
        body = mjcf.find(f".//body[@name='{name}']")
        if body is None:
            raise ValueError(f"URDF visual body missing after conversion: {name}")
        source_visual_count += 1
        origin = visual.find("origin")
        color = visual.find("material/color")
        for index, mesh_path in enumerate(_visual_mesh_paths(mesh.attrib["filename"])):
            mesh_name = f"visual_{name}_{index}"
            mesh_attrs = {"name": mesh_name, "file": mesh_path}
            if "scale" in mesh.attrib:
                mesh_attrs["scale"] = mesh.attrib["scale"]
            ET.SubElement(assets, "mesh", mesh_attrs)
            geom_attrs = {
                "name": mesh_name, "type": "mesh", "mesh": mesh_name,
                "contype": "0", "conaffinity": "0", "group": "2",
                "rgba": color.attrib["rgba"] if color is not None else "0.7 0.7 0.7 1",
                "pos": origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0",
                "quat": _quat_from_rpy(origin.attrib.get("rpy", "0 0 0")) if origin is not None else "1 0 0 0",
            }
            ET.SubElement(body, "geom", geom_attrs)
            visual_geom_count += 1

    ET.indent(mjcf, space="  ")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_name(f"{OUT.stem}.tmp.xml")
    tmp.write_text(ET.tostring(mjcf, encoding="unicode") + "\n", encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(tmp))
    assert (model.nq, model.nv, model.nu) == (23, 22, 0)
    assert sum((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("visual_") for i in range(model.ngeom)) == visual_geom_count
    tmp.replace(OUT)
    print(f"wrote {OUT}: {source_visual_count} URDF visual meshes as {visual_geom_count} MuJoCo geoms, {model.ngeom} geoms total")


if __name__ == "__main__":
    main()
