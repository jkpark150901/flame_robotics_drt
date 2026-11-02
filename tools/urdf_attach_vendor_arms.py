#!/usr/bin/env python3
"""Replace the CAD-exported arms of a combined URDF with the vendor (Rainbow Robotics) arm URDFs.

The CAD export (urdf/model.urdf) bakes an assembly pose into every arm joint and its link dimensions are
simplified, so its q = 0 pose, joint directions and wrist offsets differ from the real robots. This tool keeps
everything else of the CAD model (rails, carriages, scan bodies, tools, positioner, tcp frames) and swaps each
arm for the vendor URDF the project already uses for that robot:

    DDA  RB10-1300E   <- rb10_1300e.urdf           (matches rbpodo_ros2 robots/rb10_1300e/joint.yaml)
    RT   RB20-1900ES  <- arm part of rb20_1900es_RT.urdf

Mounting (all computed, nothing hand-tuned):
  * the vendor base joint is placed at the CAD base joint (zeroed pose, see urdf_zero_offsets.py), upright;
  * yaw: the shoulder axis points along the rail's -x at q = 0, the convention of the original per-robot URDFs;
  * the CAD tool keeps its absolute orientation at q = 0 and its offset vector from the flange, re-expressed in
    the vendor flange frame (fixed joint link6 -> tool).

Usage (the vendor files come from git history because they were removed from the working tree):
    python tools/urdf_attach_vendor_arms.py --out-dir urdf/model_vendor --tcp-z-outward dda,rt --tool-frame-rotate dda:y:180
    # input: urdf/model.cad_original.urdf (the CAD export as received); the result is copied to urdf/model.urdf
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
from urdf_zero_offsets import Urdf, make_T, rpy_to_R  # noqa: E402

# CAD arm -> what to drop / where to mount / what the vendor arm looks like
ARMS = {
    "dda": dict(
        cad_arm_links=["link_4_arm_dda_base", "link_5_arm_dda_link1", "link_6_arm_dda_link2", "link_7_arm_dda_link3",
                       "link_8_arm_dda_link4", "link_9_arm_dda_link5", "link_11_arm_dda_link6"],
        cad_mount_joint="joint_4_arm_dda_base", cad_base_joint="joint_5_arm_dda_link1",
        cad_last_link="link_11_arm_dda_link6", cad_tool_joint="joint_0_DDA", rail="link_21_rail_DDA",
        vendor_urdf="urdf/rb10_1300e.urdf", vendor_root="link0", vendor_flange="link6",
        link_map={f"link{i}": f"dda_link{i}" for i in range(7)},
        joint_map={"base": "dda_joint_base", "shoulder": "dda_joint_shoulder", "elbow": "dda_joint_elbow",
                   "wrist1": "dda_joint_wrist1", "wrist2": "dda_joint_wrist2", "wrist3": "dda_joint_wrist3"},
        mesh_dir="meshes/rb10_1300e", mount_joint_name="dda_joint_mount",
        flange_normal_local=(0.0, -1.0, 0.0),   # vendor tcp_joint offset (0, -0.1153, 0): tool side of link6
        tcp_joint="tcp_DDA_joint", tcp_link="tcp_DDA",
    ),
    "rt": dict(
        cad_arm_links=["link_10_arm_rt_base", "link_12_arm_rt_link1", "link_13_arm_rt_link2", "link_14_arm_rt_link3",
                       "link_15_arm_rt_link5", "link_16_arm_rt_link6", "link_17_arm_rt_link4"],
        cad_mount_joint="joint_10_arm_rt_base", cad_base_joint="joint_12_arm_rt_link1",
        cad_last_link="link_16_arm_rt_link6", cad_tool_joint="joint_2_RT", rail="link_20_rail_RT",
        vendor_urdf="urdf/rb20_1900es_RT.urdf", vendor_root="rt_link0", vendor_flange="rt_link6",
        link_map={f"rt_link{i}": f"rt_link{i}" for i in range(7)},
        joint_map={n: n for n in ("rt_base", "rt_shoulder", "rt_elbow", "rt_wrist1", "rt_wrist2", "rt_wrist3")},
        mesh_dir="meshes/rb20_1900es", mount_joint_name="rt_joint_mount",
        flange_normal_local=(0.0, 0.0, 1.0),    # vendor rt_joint_end offset (0, 0, 0.13): tool side of link6
        tcp_joint="tcp_RT_joint", tcp_link="tcp_RT",
    ),
}
ARM_JOINT_TYPES = ("revolute", "continuous")


def git_show(ref, path):
    return subprocess.run(["git", "show", f"{ref}:{path}"], check=True, capture_output=True).stdout


def fmt(v):
    return " ".join(f"{float(x):.10g}" for x in v)


def vendor_subtree(vendor_root_el, root_link, link_map, joint_map):
    """Arm links/joints of a vendor URDF (no track, carriage, tcp or end frames), renamed."""
    links = [copy.deepcopy(l) for l in vendor_root_el.findall("link") if l.get("name") in link_map]
    joints = [copy.deepcopy(j) for j in vendor_root_el.findall("joint") if j.get("name") in joint_map]
    for l in links:
        l.set("name", link_map[l.get("name")])
    for j in joints:
        j.set("name", joint_map[j.get("name")])
        j.find("parent").set("link", link_map[j.find("parent").get("link")])
        j.find("child").set("link", link_map[j.find("child").get("link")])
    return links, joints


# ---------------------------------------------------------------------------------------------------------------------
# mesh-contact refinement: put the arm base and the tool exactly on the mesh boundaries they mount to
# ---------------------------------------------------------------------------------------------------------------------
def _link_meshes(urdf, link, which="visual"):
    import trimesh
    T_link = urdf.link_T(link, {}, None)
    el = next(l for l in urdf.root.findall("link") if l.get("name") == link)
    out = []
    for v in el.findall(which):
        o, g = v.find("origin"), v.find("geometry")[0]
        parse = lambda key: [float(t) for t in (o.get(key, "0 0 0") if o is not None else "0 0 0").split()]
        T = T_link @ make_T(parse("xyz"), rpy_to_R(parse("rpy")))
        if g.tag == "box":
            m = trimesh.creation.box(extents=[float(t) for t in g.get("size").split()])
        elif g.tag == "cylinder":
            m = trimesh.creation.cylinder(radius=float(g.get("radius")), height=float(g.get("length")))
        elif g.tag == "mesh":
            m = trimesh.load(str(urdf.path_dir / g.get("filename")), force="mesh")
            if g.get("scale"):
                m.apply_scale([float(t) for t in g.get("scale").split()])
        else:
            continue
        m = m.copy()
        m.apply_transform(T)
        out.append(m)
    return out


class _Scene:
    """Open3D raycasting scene over trimesh meshes: inside tests and surface distances."""

    def __init__(self, meshes):
        import open3d as o3d
        self.o3d = o3d
        self.meshes = meshes
        self.watertight = all(m.is_watertight for m in meshes)
        import trimesh
        areas = np.array([max(m.area, 1e-9) for m in meshes])
        counts = np.maximum(300, (60000 * areas / areas.sum()).astype(int))
        # vertices + dense surface samples: flat faces that touch face-to-face have no vertex inside the other body
        samples = [trimesh.sample.sample_surface(m, int(c), seed=0)[0] for m, c in zip(meshes, counts)]
        self.vertices = np.vstack([np.asarray(m.vertices, float) for m in meshes] + samples)
        self.scene = o3d.t.geometry.RaycastingScene()
        for m in meshes:
            self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(np.asarray(m.vertices, float)),
                o3d.utility.Vector3iVector(np.asarray(m.faces, np.int32)))))

    def inside(self, pts):
        return self.scene.compute_occupancy(self.o3d.core.Tensor(np.asarray(pts, np.float32))).numpy() > 0.5

    def distance(self, pts):
        return self.scene.compute_distance(self.o3d.core.Tensor(np.asarray(pts, np.float32))).numpy()


PENETRATION_NOISE = 5      # isolated inside-hits are ray-parity noise at coplanar edges; real overlaps flag hundreds


def _penetration(moving: _Scene, fixed: _Scene, n, delta):
    """Number of vertices of one body inside the other when `moving` is shifted by delta along n.
    Parity (inside) tests are only trusted on closed meshes; the vendor DAE meshes are not always closed."""
    count = 0
    if fixed.watertight:
        count += int(fixed.inside(moving.vertices + delta * n).sum())
    if moving.watertight:
        count += int(moving.inside(fixed.vertices - delta * n).sum())
    return count if count > PENETRATION_NOISE else 0


def _contact_shift(moving, fixed, n, limit=0.6, step=0.002, clearance=5e-5):
    """Shift of `moving` along n that makes the two bodies just touch (no penetration), nearest to the current
    placement: out of an overlap, or across a gap until the first contact. `clearance` (0.05 mm) keeps the result on
    the free side so coplanar faces are not flagged as penetrating by float round-off."""
    pen0 = _penetration(moving, fixed, n, 0.0)
    direction = 1.0 if pen0 > 0 else -1.0     # overlapping: move out along +n; apart: close the gap along -n
    prev, d = 0.0, 0.0
    while abs(d) <= limit:
        d += direction * step
        free = _penetration(moving, fixed, n, d) == 0
        if (pen0 > 0 and free) or (pen0 == 0 and not free):
            a, b = prev, d                                         # state at prev / state at d
            for _ in range(40):
                mid = 0.5 * (a + b)
                is_free = _penetration(moving, fixed, n, mid) == 0
                if is_free == (pen0 > 0):
                    b = mid
                else:
                    a = mid
            boundary = b if pen0 > 0 else a                        # the free-side end of the contact interval
            return boundary + clearance, pen0
        prev = d
    raise RuntimeError("no contact found within %.2f m" % limit)


def refine_contacts(out_urdf: Path, tree_root_loader):
    """Re-seat each arm base on its slide body and each tool on its link6 flange face (mesh contact)."""
    report = {}
    for name, arm in tree_root_loader["arms"].items():
        root = tree_root_loader["root"]
        text = ET.tostring(root, encoding="unicode")
        tmp = out_urdf.with_name(".refine_tmp.urdf")
        tmp.write_text(text, encoding="utf-8")
        u = Urdf(tmp)
        u.path_dir = out_urdf.parent
        tmp.unlink()
        link0 = arm["link_map"][arm["vendor_root"]]
        link6 = arm["link_map"][arm["vendor_flange"]]
        mount = next(j for j in root.findall("joint") if j.get("name") == arm["mount_joint_name"])
        parent = mount.find("parent").get("link")
        # ---- arm base on the slide body ----
        body = _Scene(_link_meshes(u, parent))
        base = _Scene(_link_meshes(u, link0))
        T_link0 = u.link_T(link0, {})
        n_base = T_link0[:3, :3] @ np.array([0.0, 0.0, 1.0])                  # arm base axis (up)
        shift, pen0 = _contact_shift(base, body, n_base)
        R_parent = u.link_T(parent, {})[:3, :3]
        o = mount.find("origin")
        xyz = np.array([float(t) for t in o.get("xyz").split()]) + R_parent.T @ (n_base * shift)
        o.set("xyz", fmt(xyz))
        # ---- tool on the flange face ----
        u = Urdf_from_root(root, out_urdf.parent, out_urdf)
        L6 = _Scene(_link_meshes(u, link6))
        tool_joint = next(j for j in root.findall("joint") if j.get("name") == arm["cad_tool_joint"])
        tool_link = tool_joint.find("child").get("link")
        tool = _Scene(_link_meshes(u, tool_link))
        n_loc = np.asarray(arm["flange_normal_local"], float)
        n_world = u.link_T(link6, {})[:3, :3] @ n_loc
        s6_max = float((L6.vertices @ n_world).max())
        st_min = float((tool.vertices @ n_world).min())
        shift_plane = s6_max - st_min                                         # tool's lowest point on the flange end plane
        pen_plane = _penetration(tool, L6, n_world, shift_plane)
        # Approach from the free side: start clear of the flange and move the tool toward it along the normal until
        # the first surface contact (the tool body, not just a bracket that flanks the flange, decides where it stops).
        hi = shift_plane + 0.02
        while _penetration(tool, L6, n_world, hi) > 0:
            hi += 0.02
        step, d = 0.001, hi
        while _penetration(tool, L6, n_world, d - step) == 0 and hi - d < 0.25:
            d -= step
        a, b = d - step, d                                                    # a penetrates, b is free
        for _ in range(40):
            mid = 0.5 * (a + b)
            if _penetration(tool, L6, n_world, mid) == 0:
                b = mid
            else:
                a = mid
        shift_t = b + 5e-5
        o = tool_joint.find("origin")
        txyz = np.array([float(t) for t in o.get("xyz").split()]) + n_loc * shift_t
        o.set("xyz", fmt(txyz))
        report[name] = dict(base_shift_m=float(shift), base_was_overlapping=bool(pen0 > 0), tool_shift_m=float(shift_t),
                            tool_plane_penetration_vertices=int(pen_plane))
        print(f"   {name.upper()}: arm base moved {shift * 1000:+.1f} mm ({'was overlapping' if pen0 > 0 else 'was floating'}), "
              f"tool moved {shift_t * 1000:+.1f} mm along the flange normal")
    return report


def Urdf_from_root(root, path_dir, out_urdf):
    tmp = out_urdf.with_name(".refine_tmp.urdf")
    tmp.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    u = Urdf(tmp)
    u.path_dir = path_dir
    tmp.unlink()
    return u


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="urdf/model.cad_original.urdf", help="the CAD export as received (urdf/model.urdf is the generated result)")
    ap.add_argument("--offsets", default="urdf/model_zeroed.offsets.json", help="zero-pose offsets from urdf_zero_offsets.py")
    ap.add_argument("--vendor-ref", default="HEAD", help="git ref holding the vendor URDFs/meshes")
    ap.add_argument("--out-dir", default="urdf/model_vendor")
    ap.add_argument("--robots", default="dda,rt")
    ap.add_argument("--flip", default="", help="comma list of arms to turn 180 deg about their base axis (tool turns with the arm)")
    ap.add_argument("--tcp-z-outward", default="", help="comma list of arms whose TCP frame z+ must point away from the robot "
                                                       "(along the flange normal); minimal rotation, TCP position unchanged")
    ap.add_argument("--tool-roll", default="", help="e.g. dda:180 - turn the tool (and its TCP) about the flange axis (link6 wrist3 axis, "
                                                    "through the link6 origin) by this many degrees; the tool offset rotates with it")
    ap.add_argument("--tool-frame-rotate", default="", help="e.g. dda:y:180 - rotate the tool LINK FRAME about its local axis by this "
                                                            "angle while the geometry stays where it is (visual/collision/child-joint origins are compensated)")
    ap.add_argument("--tool-rotate", default="", help="e.g. dda:y:180 - turn the whole tool link (frame AND visual/collision meshes) about the tool's "
                                                      "local axis by this angle, pivoting at the flange centre (link6 origin) so the mount offset "
                                                      "is kept; the tool is then re-seated on the flange face")
    args = ap.parse_args()
    tool_rotate = {}
    for item in [i for i in args.tool_rotate.split(",") if i.strip()]:
        nm, ax, deg = item.split(":")
        tool_rotate[nm.strip().lower()] = ("xyz".index(ax.strip().lower()), float(deg))
    frame_rot = {}
    for item in [i for i in args.tool_frame_rotate.split(",") if i.strip()]:
        nm, ax, deg = item.split(":")
        frame_rot[nm.strip().lower()] = R.from_rotvec(np.eye(3)["xyz".index(ax.strip().lower())] * np.radians(float(deg))).as_matrix()
    tool_roll = {k.strip().lower(): float(v) for k, v in (item.split(":") for item in args.tool_roll.split(",") if item.strip())}
    tcp_outward = {r.strip().lower() for r in args.tcp_z_outward.split(",") if r.strip()}
    flips = {r.strip().lower() for r in args.flip.split(",") if r.strip()}

    out_dir = Path(args.out_dir)
    (out_dir / "meshes").mkdir(parents=True, exist_ok=True)
    cad = Urdf(args.model)
    theta = {j: v["offset_rad"] for r in json.load(open(args.offsets))["robots"].values() for j, v in r["joints"].items()}
    tree = ET.parse(args.model)
    root = tree.getroot()
    report = {}

    for name in [r.strip() for r in args.robots.split(",") if r.strip()]:
        arm = ARMS[name]
        vend_text = git_show(args.vendor_ref, arm["vendor_urdf"])
        vend_path = out_dir / f".vendor_{name}.urdf"
        vend_path.write_bytes(vend_text)
        vend = Urdf(vend_path)
        vend_root = ET.fromstring(vend_text)
        vend_path.unlink()

        # --- mounting pose -----------------------------------------------------------------------------------
        cad_mount = cad.joints[arm["cad_mount_joint"]]
        parent = cad_mount["parent"]
        T_world_parent = cad.link_T(parent, {})
        p_base_cad, _ = cad.joint_world(arm["cad_base_joint"], theta)               # zeroed CAD base joint position
        # vendor geometry at q = 0 in its own link0 frame
        base_j = next(j for j in vend.joints.values() if j["name"] in ("base", "rt_base"))
        sh_j = next(j for j in vend.joints.values() if j["name"] in ("shoulder", "rt_shoulder"))
        p_base_v, _ = vend.joint_world(base_j["name"], {}, arm["vendor_root"])
        _, a_sh_v = vend.joint_world(sh_j["name"], {}, arm["vendor_root"])
        T_rail = cad.link_T(arm["rail"], {})
        target_sh = T_rail[:3, :3] @ np.array([-1.0, 0.0, 0.0])                   # shoulder axis = rail -x at q = 0
        yaw = np.arctan2(target_sh[1], target_sh[0]) - np.arctan2(a_sh_v[1], a_sh_v[0])
        yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
        R_ref = R.from_euler("z", yaw).as_matrix()                                # reference placement (tool relation is derived here)
        yaw_mount = yaw + (np.pi if name in flips else 0.0)                       # --flip: arm turned 180 deg about its base axis
        R_m = R.from_euler("z", yaw_mount).as_matrix()
        t_link0 = p_base_cad - R_m @ p_base_v                                     # base joint stays on the CAD base axis
        T_world_link0 = make_T(t_link0, R_m)
        T_parent_link0 = np.linalg.inv(T_world_parent) @ T_world_link0
        mount_rpy = R.from_matrix(T_parent_link0[:3, :3]).as_euler("xyz")

        # --- tool: flange-relative pose from the CAD arrangement (tool keeps its bolted relation when the arm is flipped) ---
        T_ref_link0 = make_T(p_base_cad - R_ref @ p_base_v, R_ref)
        T_v6 = T_ref_link0 @ vend.link_T(arm["vendor_flange"], {}, arm["vendor_root"])
        R_11 = cad.link_T(arm["cad_last_link"], theta)[:3, :3]
        R_rel = T_v6[:3, :3].T @ R_11
        T_11_tool = cad.origin_T(arm["cad_tool_joint"])
        T_6_tool = make_T(np.zeros(3), R_rel) @ T_11_tool
        if name in tool_roll:
            # rotate about the flange axis through the link6 origin (not about the tool origin): offsets turn with the tool
            R_roll = R.from_rotvec(np.asarray(arm["flange_normal_local"], float) * np.radians(tool_roll[name])).as_matrix()
            T_6_tool = make_T(np.zeros(3), R_roll) @ T_6_tool
        tool_rpy = R.from_matrix(T_6_tool[:3, :3]).as_euler("xyz")
        tool_link = cad.joints[arm["cad_tool_joint"]]["child"]

        # --- edit the XML ---------------------------------------------------------------------------------------
        cad_joint_names = {j.get("name") for j in root.findall("joint")}
        drop_links = set(arm["cad_arm_links"])
        drop_joints = {j.get("name") for j in root.findall("joint")
                       if j.find("child").get("link") in drop_links or j.get("name") == arm["cad_tool_joint"]}
        for j in list(root.findall("joint")):
            if j.get("name") in drop_joints:
                root.remove(j)
        for l in list(root.findall("link")):
            if l.get("name") in drop_links:
                root.remove(l)
        links, joints = vendor_subtree(vend_root, arm["vendor_root"], arm["link_map"], arm["joint_map"])
        for l in links:
            root.append(l)
        mount = ET.SubElement(root, "joint", name=arm["mount_joint_name"], type="fixed")
        ET.SubElement(mount, "origin", xyz=fmt(T_parent_link0[:3, 3]), rpy=fmt(mount_rpy))
        ET.SubElement(mount, "parent", link=parent)
        ET.SubElement(mount, "child", link=arm["link_map"][arm["vendor_root"]])
        for j in joints:
            root.append(j)
        if name in tool_rotate:
            # rotate the whole link (frame + visuals + collisions follow because they are expressed in the link frame):
            # axis = the tool frame's own axis, pivot = flange centre (link6 origin), so the lateral offset is kept
            axis_idx, deg = tool_rotate[name]
            axis_in_link6 = T_6_tool[:3, axis_idx]
            R_tr = R.from_rotvec(axis_in_link6 / np.linalg.norm(axis_in_link6) * np.radians(deg)).as_matrix()
            T_6_tool = make_T(np.zeros(3), R_tr) @ T_6_tool
            tool_rpy = R.from_matrix(T_6_tool[:3, :3]).as_euler("xyz")
        if name in frame_rot:
            # new frame = old frame * Q (geometry untouched): joint origin gets Q, visuals/collisions get Q^-1 = Q^T
            Q = frame_rot[name]
            T_6_tool = T_6_tool @ make_T(np.zeros(3), Q)
            tool_rpy = R.from_matrix(T_6_tool[:3, :3]).as_euler("xyz")
            tool_link_el = next(l for l in root.findall("link") if l.get("name") == tool_link)
            for el in list(tool_link_el.findall("visual")) + list(tool_link_el.findall("collision")):
                o = el.find("origin")
                if o is None:
                    o = ET.Element("origin", xyz="0 0 0", rpy="0 0 0")
                    el.insert(0, o)
                T_el = make_T([float(t) for t in o.get("xyz", "0 0 0").split()], rpy_to_R([float(t) for t in o.get("rpy", "0 0 0").split()]))
                T_new = make_T(np.zeros(3), Q.T) @ T_el
                o.set("xyz", fmt(T_new[:3, 3]))
                o.set("rpy", fmt(R.from_matrix(T_new[:3, :3]).as_euler("xyz")))
        tool = ET.SubElement(root, "joint", name=arm["cad_tool_joint"], type="fixed")
        ET.SubElement(tool, "origin", xyz=fmt(T_6_tool[:3, 3]), rpy=fmt(tool_rpy))
        ET.SubElement(tool, "parent", link=arm["link_map"][arm["vendor_flange"]])
        ET.SubElement(tool, "child", link=tool_link)

        # --- vendor meshes ----------------------------------------------------------------------------------------
        copied = 0
        for l in links:
            for el in l.iter("mesh"):
                rel = el.get("filename")
                data = git_show(args.vendor_ref, "urdf/" + rel)
                dest = out_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                copied += 1
        report[name] = dict(
            mount_parent=parent, mount_xyz=T_parent_link0[:3, 3].tolist(), mount_rpy_deg=np.degrees(mount_rpy).tolist(),
            yaw_deg=float(np.degrees(yaw_mount)), flipped=bool(name in flips), tool_roll_deg=float(tool_roll.get(name, 0.0)), tool_frame_rotated=bool(name in frame_rot), tool_rotate=(list(tool_rotate[name]) if name in tool_rotate else None), tool_joint=arm["cad_tool_joint"], tool_xyz=T_6_tool[:3, 3].tolist(),
            tool_rpy_deg=np.degrees(tool_rpy).tolist(), vendor_meshes=copied, dropped_cad_links=sorted(drop_links))
        print(f"== {name.upper()}: mounted on {parent}, yaw {np.degrees(yaw_mount):+.1f} deg{' (flipped)' if name in flips else ''}, {copied} vendor mesh files, "
              f"dropped {len(drop_links)} CAD links")

    # CAD meshes referenced by the remaining links
    for el in root.iter("mesh"):
        rel = el.get("filename")
        dest = out_dir / rel
        if not dest.exists():
            src = Path(args.model).parent / rel
            if src.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(src.read_bytes())
    out_urdf = out_dir / "model_vendor.urdf"
    print("-- mesh-contact refinement")
    contact = refine_contacts(out_urdf, {"arms": {n: ARMS[n] for n in report}, "root": root})
    for n, c in contact.items():
        report[n].update(c)
    # TCP frames were fixed to base_link at the CAD pose; make them follow the tool like they did in the CAD design
    orig = Urdf(args.model)
    final = Urdf_from_root(root, out_dir, out_urdf)
    for n in report:
        arm = ARMS[n]
        tool_link = orig.joints[arm["cad_tool_joint"]]["child"]
        T_tool_tcp = np.linalg.inv(orig.link_T(tool_link, {})) @ orig.link_T(arm["tcp_link"], {})
        if n in frame_rot:
            T_tool_tcp = make_T(np.zeros(3), frame_rot[n].T) @ T_tool_tcp          # tcp world pose unchanged by the frame change
        if n in tcp_outward:
            # z+ of the TCP frame points away from the robot: along the flange normal (link6 -> tool side)
            T_w_tool = final.link_T(tool_link, {})
            T_w_tcp = T_w_tool @ T_tool_tcp
            n_world = final.link_T(arm["link_map"][arm["vendor_flange"]], {})[:3, :3] @ np.asarray(arm["flange_normal_local"], float)
            z = T_w_tcp[:3, 2]
            axis, c = np.cross(z, n_world), float(z @ n_world)
            s_ = float(np.linalg.norm(axis))
            if s_ > 1e-9:
                R_fix = R.from_rotvec(axis / s_ * np.arctan2(s_, c)).as_matrix()
            elif c < 0:
                R_fix = R.from_rotvec(T_w_tcp[:3, 0] * np.pi).as_matrix()      # antiparallel: half turn about the tcp x axis
            else:
                R_fix = np.eye(3)
            T_tool_tcp[:3, :3] = T_w_tool[:3, :3].T @ (R_fix @ T_w_tcp[:3, :3])
            report[n]["tcp_z_outward"] = True
        j = next(x for x in root.findall("joint") if x.get("name") == arm["tcp_joint"])
        j.find("parent").set("link", tool_link)
        o = j.find("origin")
        o.set("xyz", fmt(T_tool_tcp[:3, 3]))
        o.set("rpy", fmt(R.from_matrix(T_tool_tcp[:3, :3]).as_euler("xyz")))
        report[n]["tcp_in_tool_xyz"] = T_tool_tcp[:3, 3].tolist()
    ET.indent(tree, space="  ")
    tree.write(out_urdf, encoding="utf-8", xml_declaration=True)
    (out_dir / "model_vendor.report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_urdf}")


if __name__ == "__main__":
    sys.exit(main())
