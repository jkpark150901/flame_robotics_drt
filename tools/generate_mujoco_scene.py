#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
import sys
import xml.etree.ElementTree as ET


ROOT_PATH = pathlib.Path(__file__).resolve().parents[1]
PYTHON_PATH = ROOT_PATH / "python"
sys.path.append(str(PYTHON_PATH))

from viewermujoco.simulator import MujocoSimulator


def _rewrite_mesh_paths(root: ET.Element, output_dir: pathlib.Path):
    for mesh in root.findall(".//mesh"):
        mesh_file = mesh.get("file")
        if mesh_file:
            mesh.set("file", os.path.relpath(mesh_file, output_dir))


def _strip_scene_nodes(root: ET.Element):
    worldbody = root.find("worldbody")
    if worldbody is None:
        return

    for child in list(worldbody):
        if child.tag != "body":
            worldbody.remove(child)


def _build_scene_root(config: dict, robot_files: list[pathlib.Path], scene_dir: pathlib.Path) -> ET.Element:
    physics = config.get("physics", {})
    root = ET.Element("mujoco", {"model": "drt_scene"})
    ET.SubElement(root, "compiler", {"angle": "radian"})

    for robot_file in robot_files:
        ET.SubElement(root, "include", {"file": os.path.relpath(robot_file, scene_dir)})

    ET.SubElement(root, "option", {
        "timestep": str(physics.get("timestep", 0.001)),
        "integrator": physics.get("integrator", "implicitfast"),
        "iterations": str(physics.get("iterations", 80)),
        "ls_iterations": str(physics.get("ls_iterations", 20)),
        "gravity": " ".join(str(float(v)) for v in physics.get("gravity", [0, 0, -9.81]))
    })

    ET.SubElement(root, "statistic", {"center": "4 2 1", "extent": "6", "meansize": "0.2"})

    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", {
        "diffuse": "0.6 0.6 0.6",
        "ambient": "0.3 0.3 0.3",
        "specular": "0 0 0"
    })
    ET.SubElement(visual, "global", {"azimuth": "135", "elevation": "-25"})
    ET.SubElement(visual, "quality", {"shadowsize": "4096"})

    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", {
        "type": "2d",
        "name": "groundplane",
        "builtin": "checker",
        "mark": "edge",
        "rgb1": "0.22 0.24 0.26",
        "rgb2": "0.14 0.16 0.18",
        "markrgb": "0.8 0.8 0.8",
        "width": "300",
        "height": "300"
    })
    ET.SubElement(asset, "material", {
        "name": "groundplane",
        "texture": "groundplane",
        "texuniform": "true",
        "texrepeat": "10 10",
        "reflectance": "0.15"
    })

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", {"name": "key_light", "pos": "4 -5 7", "dir": "-0.5 0.6 -1"})
    ET.SubElement(worldbody, "light", {"name": "fill_light", "pos": "-4 4 5", "dir": "0.5 -0.4 -1"})
    ET.SubElement(worldbody, "camera", {"name": "overview", "pos": "6 -9 5", "xyaxes": "1 0 0 0 0.5 1"})
    ET.SubElement(worldbody, "geom", {
        "name": "ground",
        "type": "plane",
        "size": "10 10 0.01",
        "material": "groundplane",
        "condim": "3",
        "friction": "1 0.005 0.0001"
    })
    return root


def _vec(text: str) -> list[float]:
    return [float(v) for v in text.split()]


def _fmt(values: list[float]) -> str:
    clean = [0.0 if abs(v) < 1e-12 else v for v in values]
    return " ".join(f"{v:.10g}" for v in clean)


def _sub_vec(a: str, b: list[float]) -> str:
    values = _vec(a)
    return _fmt([values[i] - b[i] for i in range(3)])


def _apply_positioner_body_offsets(root: ET.Element):
    """Keep positioner moving-column visual frames on their MJCF body frames.

    The URDF stores some moving-column offsets on visual/joint origins. For
    MuJoCo editing this is awkward because the body stays at the slide joint
    origin while the rendered column appears elsewhere. This absorbs those
    offsets into the moving body and compensates child/collision offsets.
    """
    offsets = {
        "f_column_z": {
            "offset": [0.26, 1.375, 0.629],
            "visual": "positioner_f_column_z_visual_0",
            "collision": "positioner_f_column_z_collision_0",
            "children": ["f_column_r"],
        },
        "m_column_z": {
            "offset": [0.0, 0.0, 0.885],
            "visual": "positioner_m_column_z_visual_0",
            "collision": "positioner_m_column_z_collision_0",
            "children": ["m_column_passive_r"],
        },
    }

    for body in root.findall(".//body"):
        body_name = body.get("name")
        cfg = offsets.get(body_name)
        if cfg is None:
            continue

        offset = cfg["offset"]
        body.set("pos", _fmt(offset))

        for geom in body.findall("geom"):
            geom_name = geom.get("name")
            if geom_name == cfg["visual"]:
                geom.set("pos", "0 0 0")
            elif geom_name == cfg["collision"]:
                geom.set("pos", _fmt([-v for v in offset]))

        for child in body.findall("body"):
            if child.get("name") in cfg["children"]:
                child.set("pos", _sub_vec(child.get("pos", "0 0 0"), offset))


def main():
    parser = argparse.ArgumentParser(description="Generate MuJoCo scene XML from viewermujoco.cfg")
    parser.add_argument("--config", default=str(ROOT_PATH / "python" / "viewermujoco.cfg"))
    parser.add_argument("--output", default=str(ROOT_PATH / "mjcf" / "scene.xml"))
    parser.add_argument("--robot-dir", default=str(ROOT_PATH / "mjcf" / "robots"))
    args = parser.parse_args()

    with open(args.config, "r") as fp:
        config = json.load(fp)

    config["root_path"] = ROOT_PATH
    config.setdefault("generated_mesh_dir", "mjcf/assets/generated")
    config.setdefault("enable_collision", True)

    simulator = MujocoSimulator(config)
    if not simulator._import_mujoco():
        raise SystemExit(1)

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    robot_dir = pathlib.Path(args.robot_dir)
    robot_dir.mkdir(parents=True, exist_ok=True)

    specs = simulator._normalize_urdf_entries(config.get("urdf", []))
    robot_files = []
    for spec in specs:
        robot_xml = simulator._build_urdf_workspace_xml([spec])
        robot_root = ET.fromstring(robot_xml)
        robot_root.set("model", spec["name"])
        if spec["name"] == "positioner":
            _apply_positioner_body_offsets(robot_root)
        for node_name in ("compiler", "option"):
            node = robot_root.find(node_name)
            if node is not None:
                robot_root.remove(node)
        _strip_scene_nodes(robot_root)
        robot_path = robot_dir / f"{spec['name']}.xml"
        # MuJoCo resolves mesh paths in included XMLs from the main scene file,
        # so robot MJCF mesh paths are written relative to scene.xml.
        _rewrite_mesh_paths(robot_root, output_path.parent)
        ET.indent(robot_root, space="  ")
        robot_path.write_text(ET.tostring(robot_root, encoding="unicode"), encoding="utf-8")
        robot_files.append(robot_path)
        print(f"Wrote {robot_path}")

    scene_root = _build_scene_root(config, robot_files, output_path.parent)
    ET.indent(scene_root, space="  ")
    output_path.write_text(ET.tostring(scene_root, encoding="unicode"), encoding="utf-8")
    print(f"Wrote {output_path}")

    model = simulator.mujoco.MjModel.from_xml_path(str(output_path))
    print(f"Validated scene: nq={model.nq}, nu={model.nu}, ngeom={model.ngeom}, nmesh={model.nmesh}")


if __name__ == "__main__":
    main()
