import pathlib
import re
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque

from util.logger.console import ConsoleLogger


class MujocoSimulator:
    """Small MuJoCo runtime that accepts ZAPI commands from simtool."""

    def __init__(self, config: dict = None):
        self._config = config or {}
        self.__console = ConsoleLogger.get_logger()
        self._request_queue = deque(maxlen=100)
        self._queue_lock = threading.Lock()
        self._should_close = False
        self._current_mode = self._config.get("operation_mode", "simulation")

        self.mujoco = None
        self.viewer_module = None
        self.model = None
        self.data = None
        self.model_path = None
        mesh_dir = self._config.get("generated_mesh_dir", "")
        self._generated_mesh_dir = self._resolve_path(mesh_dir) if mesh_dir else pathlib.Path(tempfile.gettempdir()) / "viewermujoco_meshes"
        self._mimic_joints = []

    def _import_mujoco(self):
        if self.mujoco is not None:
            return True

        try:
            import mujoco
            import mujoco.viewer

            self.mujoco = mujoco
            self.viewer_module = mujoco.viewer
            return True
        except ImportError as exc:
            self.__console.error(f"MuJoCo Python package is required: {exc}")
            return False

    def _resolve_path(self, path: str) -> pathlib.Path:
        path_obj = pathlib.Path(path)
        if path_obj.is_absolute():
            return path_obj
        return pathlib.Path(self._config.get("root_path", ".")).resolve() / path_obj

    def _resolve_urdf_mesh_path(self, urdf_path: pathlib.Path, mesh_filename: str) -> pathlib.Path:
        mesh_path = pathlib.Path(mesh_filename)
        if mesh_path.is_absolute():
            return mesh_path
        if mesh_filename.startswith("package://"):
            mesh_path = pathlib.Path(mesh_filename.replace("package://", "", 1))
            return pathlib.Path(self._config.get("root_path", ".")).resolve() / mesh_path
        return urdf_path.parent / mesh_path

    def load_model(self, path: str):
        if not self._import_mujoco():
            return False

        model_path = self._resolve_path(path)
        if not model_path.is_file():
            self.__console.error(f"MuJoCo model not found: {model_path}")
            return False

        self.model = self.mujoco.MjModel.from_xml_path(str(model_path))
        self.data = self.mujoco.MjData(self.model)
        self.model_path = model_path
        self.__console.info(f"Loaded MuJoCo model: {model_path}")
        return True

    def load_models(self, model_entries: list):
        if not self._import_mujoco():
            return False

        model_specs = self._normalize_model_entries(model_entries)
        model_paths = [spec["path"] for spec in model_specs]
        missing_paths = [path for path in model_paths if not path.is_file()]
        if missing_paths:
            self.__console.error(f"MuJoCo model not found: {missing_paths[0]}")
            return False

        scene_xml = self._build_workspace_xml(model_specs)
        self.model = self.mujoco.MjModel.from_xml_string(scene_xml)
        self.data = self.mujoco.MjData(self.model)
        self.model_path = pathlib.Path("workspace")
        self.__console.info(f"Loaded MuJoCo workspace: {len(model_paths)} models")
        return True

    def load_urdf_workspace(self, urdf_entries: list):
        if not self._import_mujoco():
            return False

        specs = self._normalize_urdf_entries(urdf_entries)
        missing_paths = [spec["path"] for spec in specs if not spec["path"].is_file()]
        if missing_paths:
            self.__console.error(f"URDF file not found: {missing_paths[0]}")
            return False

        scene_xml = self._build_urdf_workspace_xml(specs)
        self.model = self.mujoco.MjModel.from_xml_string(scene_xml)
        self.data = self.mujoco.MjData(self.model)
        self.model_path = pathlib.Path("urdf_workspace")
        self._sync_ctrl_to_qpos()
        self.__console.info(f"Loaded MuJoCo URDF workspace: {len(specs)} robots")
        return True

    def _normalize_urdf_entries(self, urdf_entries: list) -> list:
        specs = []
        for index, entry in enumerate(urdf_entries):
            if isinstance(entry, str):
                path = entry
                name = pathlib.Path(entry).stem
                base = [0, 0, 0, 0, 0, 0]
            else:
                path = entry.get("path", "")
                name = entry.get("name", pathlib.Path(path).stem or f"robot_{index}")
                base = entry.get("base", [0, 0, 0, 0, 0, 0])

            base = list(base) + [0, 0, 0, 0, 0, 0]
            specs.append({
                "name": self._safe_name(name),
                "path": self._resolve_path(path),
                "base": base[:6]
            })
        return specs

    def _build_urdf_workspace_xml(self, specs: list) -> str:
        self._generated_mesh_dir.mkdir(parents=True, exist_ok=True)
        self._mimic_joints = []
        physics = self._config.get("physics", {})

        root = ET.Element("mujoco", {"model": "drt_urdf_workspace"})
        ET.SubElement(root, "compiler", {"angle": "radian"})
        ET.SubElement(root, "option", {
            "timestep": str(physics.get("timestep", 0.001)),
            "integrator": physics.get("integrator", "implicitfast"),
            "iterations": str(physics.get("iterations", 80)),
            "ls_iterations": str(physics.get("ls_iterations", 20)),
            "gravity": self._vec_to_str(physics.get("gravity", [0, 0, -9.81]))
        })

        asset = ET.SubElement(root, "asset")
        worldbody = ET.SubElement(root, "worldbody")
        ET.SubElement(worldbody, "light", {"name": "key_light", "pos": "4 -5 7", "dir": "-0.5 0.6 -1"})
        ET.SubElement(worldbody, "camera", {"name": "overview", "pos": "6 -9 5", "xyaxes": "1 0 0 0 0.5 1"})
        ET.SubElement(worldbody, "geom", {
            "name": "ground",
            "type": "plane",
            "size": "10 10 0.01",
            "rgba": "0.9 0.9 0.9 1"
        })

        actuator = ET.SubElement(root, "actuator")

        for spec in specs:
            urdf_root = ET.parse(spec["path"]).getroot()
            context = self._parse_urdf_context(spec, urdf_root, asset)
            self._append_urdf_robot(worldbody, actuator, context)

        return ET.tostring(root, encoding="unicode")

    def _parse_urdf_context(self, spec: dict, urdf_root: ET.Element, asset: ET.Element) -> dict:
        materials = self._parse_urdf_materials(urdf_root)
        links = {link.get("name"): link for link in urdf_root.findall("link")}
        joints = [joint for joint in urdf_root.findall("joint")]

        child_links = set()
        children_by_parent = {}
        for joint in joints:
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue
            parent_name = parent.get("link")
            child_name = child.get("link")
            child_links.add(child_name)
            children_by_parent.setdefault(parent_name, []).append(joint)

        root_links = [name for name in links.keys() if name not in child_links]
        return {
            "spec": spec,
            "asset": asset,
            "materials": materials,
            "links": links,
            "children_by_parent": children_by_parent,
            "root_links": root_links
        }

    def _append_urdf_robot(self, worldbody: ET.Element, actuator: ET.Element, context: dict):
        spec = context["spec"]
        base = spec["base"]
        robot_body = ET.SubElement(worldbody, "body", {
            "name": f"{spec['name']}_base",
            "pos": self._vec_to_str(base[:3]),
            "euler": self._vec_to_str(base[3:6])
        })

        for root_link in context["root_links"]:
            link_body = ET.SubElement(robot_body, "body", {"name": self._safe_name(root_link)})
            self._append_link_inertial(link_body, context["links"].get(root_link), default=True)
            self._append_link_visuals(link_body, root_link, context)
            self._append_link_collisions(link_body, root_link, context)
            self._append_child_joints(link_body, root_link, actuator, context)

    def _append_child_joints(self, parent_body: ET.Element, parent_link: str, actuator: ET.Element, context: dict):
        for joint in context["children_by_parent"].get(parent_link, []):
            child_link = joint.find("child").get("link")
            origin = self._parse_origin(joint.find("origin"))

            body_attrs = {
                "name": self._safe_name(child_link),
                "pos": self._vec_to_str(origin["xyz"]),
                "euler": self._vec_to_str(origin["rpy"])
            }
            child_body = ET.SubElement(parent_body, "body", body_attrs)
            self._append_link_inertial(child_body, context["links"].get(child_link), default=True)

            joint_type = joint.get("type", "fixed")
            if joint_type not in ("fixed", "floating", "planar"):
                self._append_mujoco_joint(child_body, actuator, joint)

            self._append_link_visuals(child_body, child_link, context)
            self._append_link_collisions(child_body, child_link, context)
            self._append_child_joints(child_body, child_link, actuator, context)

    def _append_mujoco_joint(self, body: ET.Element, actuator: ET.Element, joint: ET.Element):
        joint_name = joint.get("name", "joint")
        joint_type = joint.get("type", "revolute")
        
        joint_cfg = self._control_config_for_joint(joint_name, joint_type)
        axis_node = joint.find("axis")
        axis = self._parse_vec(axis_node.get("xyz") if axis_node is not None else "0 0 1")
        limit = joint.find("limit")
        lower = "-3.14159"
        upper = "3.14159"
        effort = "100"
        if limit is not None:
            lower = limit.get("lower", lower)
            upper = limit.get("upper", upper)
            effort = limit.get("effort", effort)

        mj_type = "slide" if joint_type == "prismatic" else "hinge"
        attrs = {
            "name": joint_name,
            "type": mj_type,
            "axis": self._vec_to_str(axis),
            "range": f"{lower} {upper}",
            "actuatorfrcrange": f"-{effort} {effort}",
            "damping": str(joint_cfg["damping"]),
            "armature": str(joint_cfg["armature"])
        }
        ET.SubElement(body, "joint", attrs)

        kp = joint_cfg["kp"]
        kv = joint_cfg["kv"]
        ET.SubElement(actuator, "position", {
            "name": f"{joint_name}_pos",
            "joint": joint_name,
            "kp": str(kp),
            "kv": str(kv),
            "ctrlrange": f"{lower} {upper}",
            "forcerange": f"-{effort} {effort}"
        })

        mimic = joint.find("mimic")
        if mimic is not None and mimic.get("joint"):
            self._mimic_joints.append({
                "joint": joint_name,
                "source": mimic.get("joint"),
                "multiplier": float(mimic.get("multiplier", "1.0")),
                "offset": float(mimic.get("offset", "0.0"))
            })

    def _control_config_for_joint(self, joint_name: str, joint_type: str) -> dict:
        control = self._config.get("control", {})
        default = control.get("prismatic" if joint_type == "prismatic" else "revolute", {})
        joint_overrides = control.get("joints", {}).get(joint_name, {})

        base = {
            "kp": 300 if joint_type == "prismatic" else 40,
            "kv": 80 if joint_type == "prismatic" else 12,
            "damping": 80 if joint_type == "prismatic" else 8,
            "armature": 0.1 if joint_type == "prismatic" else 0.05
        }
        base.update(default)
        base.update(joint_overrides)
        return base

    def _append_link_inertial(self, body: ET.Element, link: ET.Element, default: bool = False):
        inertial = link.find("inertial") if link is not None else None
        if inertial is None:
            if default:
                ET.SubElement(body, "inertial", {
                    "pos": "0 0 0",
                    "mass": "1",
                    "diaginertia": "0.01 0.01 0.01"
                })
            return

        origin = self._parse_origin(inertial.find("origin"))
        mass_node = inertial.find("mass")
        inertia_node = inertial.find("inertia")
        mass = mass_node.get("value", "1") if mass_node is not None else "1"
        ixx = inertia_node.get("ixx", "0.01") if inertia_node is not None else "0.01"
        iyy = inertia_node.get("iyy", "0.01") if inertia_node is not None else "0.01"
        izz = inertia_node.get("izz", "0.01") if inertia_node is not None else "0.01"

        attrs = {
            "pos": self._vec_to_str(origin["xyz"]),
            "mass": mass,
            "diaginertia": f"{ixx} {iyy} {izz}"
        }
        if any(abs(v) > 1e-12 for v in origin["rpy"]):
            attrs["euler"] = self._vec_to_str(origin["rpy"])
        ET.SubElement(body, "inertial", attrs)

    def _append_link_visuals(self, body: ET.Element, link_name: str, context: dict):
        link = context["links"].get(link_name)
        if link is None:
            return

        for index, visual in enumerate(link.findall("visual")):
            geom_node = visual.find("geometry")
            if geom_node is None:
                continue

            origin = self._parse_origin(visual.find("origin"))
            rgba = self._visual_rgba(visual, context["materials"])
            geom_attrs = {
                "name": self._safe_name(f"{context['spec']['name']}_{link_name}_visual_{index}"),
                "pos": self._vec_to_str(origin["xyz"]),
                "euler": self._vec_to_str(origin["rpy"]),
                "rgba": rgba,
                "contype": "0",
                "conaffinity": "0",
                "group": "1"
            }

            box_node = geom_node.find("box")
            mesh_node = geom_node.find("mesh")
            if box_node is not None:
                size = [float(v) * 0.5 for v in self._parse_vec(box_node.get("size", "0.1 0.1 0.1"))]
                geom_attrs.update({"type": "box", "size": self._vec_to_str(size)})
                ET.SubElement(body, "geom", geom_attrs)
            elif mesh_node is not None:
                mesh_name = self._ensure_mesh_asset(mesh_node, context, link_name, index)
                if mesh_name:
                    geom_attrs.update({"type": "mesh", "mesh": mesh_name})
                    ET.SubElement(body, "geom", geom_attrs)

    def _append_link_collisions(self, body: ET.Element, link_name: str, context: dict):
        if not self._config.get("enable_collision", True):
            return

        link = context["links"].get(link_name)
        if link is None:
            return

        collision_cfg = self._config.get("collision", {})
        for index, collision in enumerate(link.findall("collision")):
            geom_node = collision.find("geometry")
            if geom_node is None:
                continue

            origin = self._parse_origin(collision.find("origin"))
            geom_attrs = {
                "name": self._safe_name(f"{context['spec']['name']}_{link_name}_collision_{index}"),
                "pos": self._vec_to_str(origin["xyz"]),
                "euler": self._vec_to_str(origin["rpy"]),
                "rgba": collision_cfg.get("rgba", "1 0 0 0.18"),
                "contype": str(collision_cfg.get("contype", 1)),
                "conaffinity": str(collision_cfg.get("conaffinity", 1)),
                "group": str(collision_cfg.get("group", 3)),
                "condim": str(collision_cfg.get("condim", 3)),
                "friction": self._vec_to_str(collision_cfg.get("friction", [1, 0.005, 0.0001])),
                "solref": self._vec_to_str(collision_cfg.get("solref", [0.01, 1])),
                "solimp": self._vec_to_str(collision_cfg.get("solimp", [0.9, 0.95, 0.001]))
            }

            box_node = geom_node.find("box")
            mesh_node = geom_node.find("mesh")
            if box_node is not None:
                size = [float(v) * 0.5 for v in self._parse_vec(box_node.get("size", "0.1 0.1 0.1"))]
                geom_attrs.update({"type": "box", "size": self._vec_to_str(size)})
                ET.SubElement(body, "geom", geom_attrs)
            elif mesh_node is not None:
                mesh_name = self._ensure_mesh_asset(mesh_node, context, link_name, index, role="collision")
                if mesh_name:
                    geom_attrs.update({"type": "mesh", "mesh": mesh_name})
                    ET.SubElement(body, "geom", geom_attrs)

    def _ensure_mesh_asset(self, mesh_node: ET.Element, context: dict, link_name: str, index: int, role: str = "visual") -> str:
        filename = mesh_node.get("filename", "")
        if not filename:
            return ""

        mesh_path = self._resolve_urdf_mesh_path(context["spec"]["path"], filename)
        if not mesh_path.is_file():
            self.__console.warning(f"Mesh file not found: {mesh_path}")
            return ""

        scale = mesh_node.get("scale", "")
        mesh_name = self._safe_name(f"{context['spec']['name']}_{link_name}_{role}_{index}")
        asset_file = mesh_path
        asset_attrs = {"name": mesh_name}

        if mesh_path.suffix.lower() != ".obj":
            asset_file = self._convert_mesh_to_obj(mesh_path, mesh_name)
        if not asset_file:
            return ""

        asset_attrs["file"] = str(asset_file)
        if scale:
            asset_attrs["scale"] = scale
        ET.SubElement(context["asset"], "mesh", asset_attrs)
        return mesh_name

    def _convert_mesh_to_obj(self, mesh_path: pathlib.Path, mesh_name: str) -> pathlib.Path:
        try:
            import trimesh

            loaded = trimesh.load(str(mesh_path), force="scene")
            if isinstance(loaded, trimesh.Scene):
                mesh = loaded.to_geometry()
            else:
                mesh = loaded

            output_path = self._generated_mesh_dir / f"{mesh_name}.obj"
            mesh.export(output_path)
            return output_path
        except Exception as exc:
            self.__console.warning(f"Failed to convert mesh to OBJ: {mesh_path} ({exc})")
            return None

    def _parse_urdf_materials(self, urdf_root: ET.Element) -> dict:
        materials = {}
        for material in urdf_root.findall("material"):
            color = material.find("color")
            if material.get("name") and color is not None:
                materials[material.get("name")] = color.get("rgba", "0.6 0.6 0.6 1")
        return materials

    def _visual_rgba(self, visual: ET.Element, materials: dict) -> str:
        material = visual.find("material")
        if material is None:
            return "0.25 0.45 0.6 0.9"
        color = material.find("color")
        if color is not None:
            return color.get("rgba", "0.25 0.45 0.6 0.9")
        return materials.get(material.get("name"), "0.25 0.45 0.6 0.9")

    def _parse_origin(self, origin: ET.Element) -> dict:
        if origin is None:
            return {"xyz": [0, 0, 0], "rpy": [0, 0, 0]}
        return {
            "xyz": self._parse_vec(origin.get("xyz", "0 0 0")),
            "rpy": self._parse_vec(origin.get("rpy", "0 0 0"))
        }

    def _parse_vec(self, text: str) -> list:
        return [float(v) for v in text.split()]

    def _vec_to_str(self, values: list) -> str:
        return " ".join(f"{float(v):.10g}" for v in values)

    def _safe_name(self, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]", "_", value)

    def _normalize_model_entries(self, model_entries: list) -> list:
        model_specs = []
        for index, entry in enumerate(model_entries):
            if isinstance(entry, str):
                path = entry
                name = pathlib.Path(entry).stem
                base = [0, 0, 0, 0, 0, 0]
            else:
                path = entry.get("path", "")
                name = entry.get("name", pathlib.Path(path).stem or f"model_{index}")
                base = entry.get("base", [0, 0, 0, 0, 0, 0])

            base = list(base) + [0, 0, 0, 0, 0, 0]
            model_specs.append({
                "name": name,
                "path": self._resolve_path(path),
                "base": base[:6]
            })
        return model_specs

    def _build_workspace_xml(self, model_specs: list) -> str:
        root = ET.Element("mujoco", {"model": "drt_workspace"})
        ET.SubElement(root, "compiler", {"angle": "radian"})
        ET.SubElement(root, "option", {"timestep": "0.002"})

        worldbody = ET.SubElement(root, "worldbody")
        ET.SubElement(worldbody, "light", {"name": "key_light", "pos": "4 -5 7", "dir": "-0.5 0.6 -1"})
        ET.SubElement(worldbody, "camera", {"name": "overview", "pos": "6 -9 5", "xyaxes": "1 0 0 0 0.5 1"})
        ET.SubElement(worldbody, "geom", {
            "name": "ground",
            "type": "plane",
            "size": "10 10 0.01",
            "rgba": "0.15 0.15 0.15 1"
        })

        actuator = ET.SubElement(root, "actuator")

        for spec in model_specs:
            source_root = ET.parse(spec["path"]).getroot()
            base = spec["base"]
            base_body = ET.SubElement(worldbody, "body", {
                "name": f"{spec['name']}_base",
                "pos": f"{base[0]} {base[1]} {base[2]}",
                "euler": f"{base[3]} {base[4]} {base[5]}"
            })

            source_worldbody = source_root.find("worldbody")
            if source_worldbody is not None:
                for child in list(source_worldbody):
                    if child.tag == "body":
                        base_body.append(child)

            source_actuator = source_root.find("actuator")
            if source_actuator is not None:
                for child in list(source_actuator):
                    actuator.append(child)

        return ET.tostring(root, encoding="unicode")

    def push_request(self, data: dict):
        with self._queue_lock:
            self._request_queue.append(data)

    def _pop_request(self):
        with self._queue_lock:
            if self._request_queue:
                return self._request_queue.popleft()
        return None

    def _process_requests(self):
        while True:
            request = self._pop_request()
            if request is None:
                break

            command = request.get("command")
            if command == "load_model":
                self.load_model(request.get("path", ""))
            elif command == "load_models":
                self.load_models(request.get("paths", []))
            elif command == "load_urdf_workspace":
                self.load_urdf_workspace(request.get("urdf", []))
            elif command == "set_mode":
                self._current_mode = request.get("mode", self._current_mode)
                self.__console.info(f"MuJoCo mode set to: {self._current_mode}")
            elif command == "reset":
                self.reset()
            elif command == "set_joint_positions":
                self.set_joint_positions(request.get("positions", {}))
            elif command == "set_joint_targets":
                self.set_joint_targets(request.get("targets", {}))
            elif command == "terminate":
                self._should_close = True
            else:
                self.__console.warning(f"Unknown MuJoCo command: {command}")

    def reset(self):
        if self.model is None or self.data is None:
            return
        self.mujoco.mj_resetData(self.model, self.data)
        self._sync_ctrl_to_qpos()
        self.mujoco.mj_forward(self.model, self.data)
        self.__console.info("MuJoCo simulation reset")

    def set_joint_positions(self, positions: dict):
        """Set qpos directly. Useful for MJCFs that do not define actuators yet."""
        if self.model is None or self.data is None:
            return

        for joint_name, value in positions.items():
            joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                self.__console.warning(f"Unknown MuJoCo joint: {joint_name}")
                continue

            qpos_adr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_adr] = float(value)
            actuator_id = self._find_actuator_id(joint_name)
            if actuator_id >= 0:
                self.data.ctrl[actuator_id] = float(value)

        self._apply_mimic_joints()
        self.mujoco.mj_forward(self.model, self.data)

    def set_joint_targets(self, targets: dict):
        """Set actuator controls by actuator name or joint name."""
        if self.model is None or self.data is None:
            return

        if self.model.nu == 0:
            self.__console.warning("MuJoCo model has no actuators; use set_joint_positions instead")
            return

        for name, value in targets.items():
            actuator_id = self._find_actuator_id(name)
            if actuator_id < 0:
                self.__console.warning(f"Unknown MuJoCo actuator or joint target: {name}")
                continue
            self.data.ctrl[actuator_id] = float(value)

    def _sync_ctrl_to_qpos(self):
        if self.model is None or self.data is None:
            return
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id][0])
            if joint_id < 0:
                continue
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            self.data.ctrl[actuator_id] = self.data.qpos[qpos_adr]

    def _apply_actuator_controls_to_qpos(self):
        if self.model is None or self.data is None:
            return
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id][0])
            if joint_id < 0:
                continue
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            ctrl = float(self.data.ctrl[actuator_id])
            if self.model.actuator_ctrllimited[actuator_id]:
                low, high = self.model.actuator_ctrlrange[actuator_id]
                ctrl = min(max(ctrl, float(low)), float(high))
                self.data.ctrl[actuator_id] = ctrl
            self.data.qpos[qpos_adr] = ctrl

        self._apply_mimic_joints()

    def _apply_mimic_joints(self):
        if self.model is None or self.data is None:
            return
        for mimic in self._mimic_joints:
            source_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, mimic["source"])
            joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, mimic["joint"])
            if source_id < 0 or joint_id < 0:
                continue

            source_value = self.data.qpos[int(self.model.jnt_qposadr[source_id])]
            value = source_value * mimic["multiplier"] + mimic["offset"]
            self.data.qpos[int(self.model.jnt_qposadr[joint_id])] = value

            actuator_id = self._find_actuator_id(mimic["joint"])
            if actuator_id >= 0:
                self.data.ctrl[actuator_id] = value

    def _find_actuator_id(self, name: str) -> int:
        actuator_id = self.mujoco.mj_name2id(
            self.model,
            self.mujoco.mjtObj.mjOBJ_ACTUATOR,
            name
        )
        if actuator_id >= 0:
            return actuator_id

        return self.mujoco.mj_name2id(
            self.model,
            self.mujoco.mjtObj.mjOBJ_ACTUATOR,
            f"{name}_pos"
        )

    def run(self, fps: int = 60):
        if not self._import_mujoco():
            return

        default_urdf = self._config.get("urdf", [])
        default_models = self._config.get("models", [])
        default_model = self._config.get("model", "")
        if default_urdf and self.model is None:
            self.load_urdf_workspace(default_urdf)
        elif default_models and self.model is None:
            self.load_models(default_models)
        elif default_model and self.model is None:
            self.load_model(default_model)

        if self.model is None or self.data is None:
            self.__console.error("No MuJoCo model loaded; exiting viewer")
            return

        frame_dt = 1.0 / max(fps, 1)
        simulate_physics = self._config.get("simulate_physics", False)
        with self.viewer_module.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running() and not self._should_close:
                start = time.time()
                self._process_requests()

                if self._current_mode == "simulation" and simulate_physics:
                    self.mujoco.mj_step(self.model, self.data)
                else:
                    self._apply_actuator_controls_to_qpos()
                    self.mujoco.mj_forward(self.model, self.data)

                viewer.sync()
                elapsed = time.time() - start
                if elapsed < frame_dt:
                    time.sleep(frame_dt - elapsed)
