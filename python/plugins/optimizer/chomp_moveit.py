"""CHOMP (MoveIt 2 chomp_motion_planner, the reference implementation) as an optimizer plugin.

The planner's path is the seed ("fillTrajectory" initialisation) of MoveIt's ChompPlanner, run by the standalone C++
runner ~/ws_chomp/install/chomp_runner (built against the system MoveIt, with ~/ws_chomp's patched
chomp_motion_planner). The obstacle potential is CHOMP's own (potential = "chomp") or the APF used by the project's
other optimizers, 0.5*eta*(1/d - 1/d0)^2 (potential = "apf" -> environment variable CHOMP_POTENTIAL=apf for the patched
library; d0 = collision_clearance, eta = apf_eta).

Inputs prepared per call (cached in a temp dir):
- URDF copy with absolute file:// mesh paths (MoveIt cannot resolve the project's relative ones),
- SRDF: group "arm" = every non-fixed joint of the robot; collision checking disabled for exactly the link pairs that
  the project's PyBullet scene does NOT check, so CHOMP's self-collision set equals the project's,
- the obstacle meshes of the robot's current collision scene (backend._configured_meshes: pipe, positioner, virtual wall,
  already in the robot root frame) as STL,
- distance-field volume = the C-space box (viewervedo.cfg c_space_bound) in the robot root frame.
The CHOMP trajectory (100 points) is returned as the optimized path; the caller verifies it with the project's own
collision check and falls back to the planner's path on a regression.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

import numpy as np

from plugins.pluginbase.optimizerbase import OptimizerBase

WS = os.path.expanduser("~/ws_chomp")
RUNNER = os.path.join(WS, "install", "chomp_runner", "lib", "chomp_runner", "chomp_runner")
_CACHE_DIR = os.path.join(tempfile.gettempdir(), "flame_chomp")


def _robot_files(handle, joint_names: List[str]) -> Dict[str, str]:
    """URDF (absolute mesh paths) + SRDF for a robot handle, cached by URDF path + self pair set."""
    urdf_path = os.path.abspath(handle.description.urdf_path)
    scene = handle.pb_scene
    pairs = sorted(tuple(sorted((a, b))) for a, b in (scene.self_pairs if scene is not None else []))
    key = hashlib.md5((urdf_path + repr(pairs) + repr(joint_names)).encode()).hexdigest()[:12]
    out = os.path.join(_CACHE_DIR, key)
    urdf_out, srdf_out = os.path.join(out, "robot.urdf"), os.path.join(out, "robot.srdf")
    if os.path.exists(urdf_out) and os.path.exists(srdf_out):
        return {"urdf": urdf_out, "srdf": srdf_out}
    os.makedirs(out, exist_ok=True)
    base_dir = os.path.dirname(urdf_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename", "")
        if fn.startswith("package://") or fn.startswith("file://"):
            continue
        mesh.set("filename", "file://" + os.path.normpath(os.path.join(base_dir, fn)))
    tree.write(urdf_out)
    robot_name = root.get("name")
    links = [l.get("name") for l in root.findall("link")]
    colliding = {l.get("name") for l in root.findall("link") if l.find("collision") is not None}
    checked = set(pairs)
    srdf = ET.Element("robot", name=robot_name)
    grp = ET.SubElement(srdf, "group", name="arm")
    for jn in joint_names:
        ET.SubElement(grp, "joint", name=jn)
    for i, a in enumerate(links):
        for b in links[i + 1:]:
            if a in colliding and b in colliding and tuple(sorted((a, b))) not in checked:
                ET.SubElement(srdf, "disable_collisions", link1=a, link2=b, reason="Never")
    ET.ElementTree(srdf).write(srdf_out)
    return {"urdf": urdf_out, "srdf": srdf_out}


def _mesh_files(meshes) -> List[str]:
    """STL files of the configured static meshes (robot root frame), cached by object identity + vertex count."""
    import open3d as o3d
    os.makedirs(_CACHE_DIR, exist_ok=True)
    out = []
    for m in meshes or []:
        v = np.asarray(m.vertices)
        key = hashlib.md5(v[:: max(1, len(v) // 2000)].tobytes() + str(len(v)).encode()).hexdigest()[:16]
        path = os.path.join(_CACHE_DIR, f"mesh_{key}.stl")
        if not os.path.exists(path):
            mm = o3d.geometry.TriangleMesh(m)
            mm.compute_triangle_normals()
            o3d.io.write_triangle_mesh(path, mm)
        out.append(path)
    return out


class ChompMoveIt(OptimizerBase):
    def __init__(self, config_path: str = None):
        super().__init__(config_path)
        config_path = config_path or os.path.splitext(__file__)[0] + ".json"
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.potential = str(self.config.get("potential", "apf"))
        self.apf_eta = float(self.config.get("apf_eta", 1.0))
        self.collision_clearance = float(self.config.get("collision_clearance", 0.15))
        self.chomp_params = dict(self.config.get("chomp", {}))
        self.resolution = float(self.config.get("resolution", 0.03))
        self.max_propagation = float(self.config.get("max_propagation", 0.25))
        self.c_space_bound = list(self.config.get("c_space_bound", [8.5, 5.0, 3.7]))
        self.timeout_s = float(self.config.get("timeout_s", 120.0))
        self.last_optimization_status = None
        self.last_result: Dict[str, Any] = {}

    def optimize(self, path: list, planner) -> list:
        self.last_result = {}
        if not path or len(path) < 2 or not os.path.exists(RUNNER):
            self.last_optimization_status = "chomp_unavailable" if path else None
            return path
        backend, robot_name = planner._robotics_collision_backend()
        handle = backend._handle(robot_name)
        joint_names = list(backend.joint_names(robot_name))
        files = _robot_files(handle, joint_names)
        meshes = _mesh_files(handle._configured_meshes)
        base_T = np.asarray(handle.description.base_T, dtype=float)
        # distance-field volume: the C-space box (world, origin 0) expressed in the robot root frame
        gmin = -base_T[:3, 3]
        work = tempfile.mkdtemp(prefix="run_", dir=_CACHE_DIR)
        seed_csv, out_csv, cfg_path = (os.path.join(work, n) for n in ("seed.csv", "out.csv", "chomp.cfg"))
        arr = np.asarray(path, dtype=float)
        np.savetxt(seed_csv, arr, delimiter=",", fmt="%.10f")
        lines = [f"urdf={files['urdf']}", f"srdf={files['srdf']}", "group=arm",
                 "start=" + ",".join(f"{v:.10f}" for v in arr[0]), "goal=" + ",".join(f"{v:.10f}" for v in arr[-1]),
                 f"seed={seed_csv}", f"out={out_csv}",
                 "grid_min=" + ",".join(f"{v:.4f}" for v in gmin),
                 "grid_size=" + ",".join(f"{v:.4f}" for v in self.c_space_bound),
                 f"resolution={self.resolution}", f"max_propagation={self.max_propagation}",
                 f"collision_clearance={self.collision_clearance}"]
        lines += [f"mesh={m}" for m in meshes]
        lines += [f"{k}={v}" for k, v in self.chomp_params.items()]
        with open(cfg_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        env_prefix = (f"export CHOMP_POTENTIAL={'apf' if self.potential == 'apf' else 'chomp'} "
                      f"CHOMP_APF_ETA={self.apf_eta}; ")
        cmd = (f"source /opt/ros/humble/setup.bash && source {WS}/install/setup.bash && {env_prefix}"
               f"{RUNNER} {cfg_path}")
        try:
            proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=self.timeout_s,
                                  env={"HOME": os.path.expanduser("~"), "PATH": "/usr/local/bin:/usr/bin:/bin"})
        except subprocess.TimeoutExpired:
            self.last_optimization_status = "chomp_timeout"
            return path
        m = re.search(r"RESULT success=(\d) collision_free=(\d) points=(\d+) time_s=([\d.eE+-]+)", proc.stdout)
        if not m:
            self.last_optimization_status = "chomp_error"
            self.last_result = {"stderr": proc.stderr[-2000:], "stdout": proc.stdout[-2000:]}
            print(f"[CHOMP] runner failed: {proc.stderr[-500:]}")
            return path
        ok, free, n, secs = int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))
        self.last_result = {"success": bool(ok), "collision_free": bool(free), "points": n, "time_s": secs,
                            "potential": self.potential}
        print(f"[CHOMP] potential={self.potential} success={ok} collision_free(MoveIt)={free} points={n} "
              f"time={secs:.2f}s")
        if not ok or n < 2 or not os.path.exists(out_csv):
            self.last_optimization_status = "chomp_failed"
            return path
        result = np.loadtxt(out_csv, delimiter=",", ndmin=2)
        self.last_optimization_status = "success" if free else "chomp_not_collision_free"
        return [np.asarray(q, dtype=float) for q in result]
