#!/usr/bin/env python
"""Export the combined CAD model (urdf/model.urdf) as the three per-robot URDFs the simulation tool loads.

The Viewer / SimTool / Robot Core address joints and links by name and assume a few conventions that the combined
model does not share, so this adapter re-expresses the model instead of changing the application:

positioner.urdf        base_link, f_column_z, f_column_r, f_column_passive_clamp, m_column, m_column_z,
                       m_column_passive_r with the old joint names. Every link frame is world-aligned at the zero pose and
                       every axis points along +x/+y/+z, so UI x -> joint -x, z -> both z joints, clamp -> joint -clamp
                       keep working. Zero poses follow the old URDF: z=0 is the lowest position, clamp/x=0 the retracted end.
rb10_1300e_DDA.urdf    dda_link_linear_track -> dda_joint_linear_track (0 at the -x end, +x) -> dda_link_carriage ->
                       dda_joint_carriage (+ = towards the cell centre) -> dda_link_slide -> arm -> dda_link_end (= TCP
                       frame, z+ outward, one merged tool mesh)
rb20_1900es_RT.urdf    same with the rt_ prefix

All three share the model's world frame (base_link of the CAD), so the cfg base pose of each is the same cell offset.
Usage: python tools/urdf_export_app_models.py [--model urdf/model.urdf] [--out-dir urdf]
"""
import argparse
import copy
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from urdf_zero_offsets import Urdf, make_T, rpy_to_R  # noqa: E402

EFFORT, VELOCITY = "1000.0", "1.0"
Z_LIFT_RANGE = 0.8      # CAD lift joints run -0.8..0 (0 = top); the app expects 0 = lowest


# ---------------------------------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------------------------------
def fmt(v):
    v = 0.0 if abs(v) < 1e-9 else float(v)
    return f"{v:.9g}"


def vec_str(v):
    return " ".join(fmt(x) for x in v)


def T_of(el):
    """4x4 of an element's <origin>."""
    o = el.find("origin") if el is not None else None
    if o is None:
        return np.eye(4)
    xyz = [float(x) for x in o.get("xyz", "0 0 0").split()]
    rpy = [float(x) for x in o.get("rpy", "0 0 0").split()]
    return make_T(xyz, rpy_to_R(rpy))


def rpy_of(R):
    sy = -R[2, 0]
    pitch = math.asin(max(-1.0, min(1.0, sy)))
    if abs(abs(sy) - 1.0) < 1e-9:
        return np.array([0.0, pitch, math.atan2(-R[0, 1], R[1, 1])])
    return np.array([math.atan2(R[2, 1], R[2, 2]), pitch, math.atan2(R[1, 0], R[0, 0])])


def set_origin(parent, T):
    o = parent.find("origin")
    if o is None:
        o = ET.SubElement(parent, "origin")
        parent.remove(o)
        parent.insert(0, o)
    o.set("xyz", vec_str(T[:3, 3]))
    o.set("rpy", vec_str(rpy_of(T[:3, :3])))


def clean_axis(a):
    a = np.array(a.split() if isinstance(a, str) else a, float)
    a[np.abs(a) < 1e-6] = 0.0
    return a / np.linalg.norm(a)


def trans(v):
    T = np.eye(4)
    T[:3, 3] = v
    return T


def new_joint(name, jtype, parent, child, T, axis=None, lower=None, upper=None):
    j = ET.Element("joint", name=name, type=jtype)
    ET.SubElement(j, "parent", link=parent)
    ET.SubElement(j, "child", link=child)
    set_origin(j, T)
    if axis is not None:
        ET.SubElement(j, "axis", xyz=vec_str(axis))
    if lower is not None:
        ET.SubElement(j, "limit", lower=fmt(lower), upper=fmt(upper), effort=EFFORT, velocity=VELOCITY)
    return j


def geometry_pieces(link_el, kind):
    """(geometry element, origin T, material element) of every <visual>/<collision>, capsules expanded."""
    out = []
    for g in link_el.findall(kind):
        geom = g.find("geometry")[0]
        T = T_of(g)
        mat = g.find("material")
        if geom.tag == "capsule":
            r, length = float(geom.get("radius")), float(geom.get("length"))
            cyl = ET.Element("cylinder", radius=geom.get("radius"), length=geom.get("length"))
            out.append((cyl, T, mat))
            for s in (-1.0, 1.0):
                out.append((ET.Element("sphere", radius=geom.get("radius")), T @ trans([0, 0, s * length / 2]), mat))
        else:
            out.append((geom, T, mat))
    return out


def add_geometry(new_link, src_link, kind, to_new):
    """Copy src_link's <visual>/<collision> into new_link, pre-multiplying each origin with `to_new` (4x4)."""
    for geom, T, mat in geometry_pieces(src_link, kind):
        e = ET.SubElement(new_link, kind)
        set_origin(e, to_new @ T)
        ET.SubElement(e, "geometry").append(copy.deepcopy(geom))
        if mat is not None:
            e.append(copy.deepcopy(mat))


def write_urdf(root, path):
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


class Model:
    def __init__(self, path):
        self.path = path
        self.root = ET.parse(path).getroot()
        self.links = {l.get("name"): l for l in self.root.findall("link")}
        self.joints = {j.get("name"): j for j in self.root.findall("joint")}
        self.kin = Urdf(path)


# ---------------------------------------------------------------------------------------------------------------------
# positioner
# ---------------------------------------------------------------------------------------------------------------------
def export_positioner(m, out_dir):
    """World-aligned positioner with the old joint names/axes; returns the report entries."""
    q0 = {"joint_24_positioner_lift_link1": -Z_LIFT_RANGE, "joint_25_positioner_lift_idler_link2": -Z_LIFT_RANGE}
    W = lambda link: m.kin.link_T(link, q0)
    pos = lambda link: W(link)[:3, 3].copy()

    L22, L23, L24, L25 = ("link_22_positioner_base", "link_23_positioner_lift_idler_link1",
                          "link_24_positioner_lift_link1", "link_25_positioner_lift_idler_link2")
    L26, L27, L28, L29 = ("link_26_positioner_lift_link2", "link_27_positioner_lift_link3",
                          "link_28_positioner_lift_link4", "link_29_positioner_lift_link5")
    L30, L31 = "link_30_positioner_lift_idler_link4", "link_31_positioner_lift_idler_link3"

    # the chuck frames sit on the chuck axis (y, z) like the old clamp/passive_r frames, so cfg center_offset is x only
    f_clamp = np.array([pos(L27)[0], pos(L28)[1], pos(L28)[2]])
    # name, parent, joint, type, axis, (lo, hi), source links, frame position
    spec = [
        ("base_link", None, None, None, None, None, [L22], np.zeros(3)),
        ("f_column_z", "base_link", "base_to_f_column_z", "prismatic", [0, 0, 1], (0.0, Z_LIFT_RANGE), [L24], pos(L24)),
        ("f_column_r", "f_column_z", "f_column_z_to_f_column_r", "revolute", [1, 0, 0], (-math.pi, math.pi), [L26], pos(L26)),
        ("f_column_passive_clamp", "f_column_r", "f_column_r_to_f_column_passive_clamp", "prismatic", [0, 1, 0],
         (-float(m.joints["joint_27_positioner_lift_link3"].find("limit").get("upper")), 0.0), [L27, L28, L29], f_clamp),
        ("m_column", "base_link", "base_to_m_column", "prismatic", [1, 0, 0],
         (-float(m.joints["joint_23_positioner_lift_idler_link1"].find("limit").get("upper")), 0.0), [L23], pos(L23)),
        ("m_column_z", "m_column", "m_column_to_m_column_z", "prismatic", [0, 0, 1], (0.0, Z_LIFT_RANGE), [L25], pos(L25)),
        ("m_column_passive_r", "m_column_z", "m_column_z_to_m_column_passive_r", "revolute", [1, 0, 0],
         (-math.pi, math.pi), [L31, L30], pos(L31)),
    ]
    frames = {s[0]: s[7] for s in spec}
    robot = ET.Element("robot", name="positioner")
    ET.SubElement(robot, "link", name="__placeholder__")
    robot.remove(robot[0])
    joints = []
    for name, parent, jname, jtype, axis, lim, srcs, fpos in spec:
        link = ET.SubElement(robot, "link", name=name)
        F_inv = np.linalg.inv(trans(fpos))
        for src in srcs:
            for kind in ("visual", "collision"):
                add_geometry(link, m.links[src], kind, F_inv @ W(src))
        if parent is not None:
            j = new_joint(jname, jtype, parent, name, trans(fpos - frames[parent]), axis, lim[0], lim[1])
            if jname == "m_column_to_m_column_z":
                ET.SubElement(j, "mimic", joint="base_to_f_column_z", multiplier="1.0", offset="0.0")
            joints.append(j)
    for j in joints:
        robot.append(j)
    write_urdf(robot, os.path.join(out_dir, "positioner.urdf"))

    # chuck faces: where the pipe end seats, as an offset in the chuck link frame (see cfg chuck_mount)
    def face_x(links, pick):
        xs = []
        for l in links:
            for geom, T, _ in geometry_pieces(m.links[l], "collision"):
                pts = piece_mesh(geom, m).copy()
                pts.apply_transform(W(l) @ T)
                xs.extend(pts.bounds[:, 0])
        return pick(xs)

    f_face = face_x([L28, L29], max)
    m_face = face_x([L30, L31], min)
    return {
        "f_column": {"center_offset": [round(f_face - f_clamp[0], 4), 0.0, 0.0], "axis": [1.0, 0.0, 0.0]},
        "m_column": {"center_offset": [round(m_face - pos(L31)[0], 4), 0.0, 0.0], "axis": [-1.0, 0.0, 0.0]},
        "chuck_axis_world_at_zero": {"f_clamp_q0": [round(float(v), 4) for v in f_clamp],
                                     "m": [round(float(v), 4) for v in pos(L31)]},
        "limits": {s[2]: [round(s[5][0], 4), round(s[5][1], 4)] for s in spec if s[2]},
    }


# ---------------------------------------------------------------------------------------------------------------------
# tools / robots
# ---------------------------------------------------------------------------------------------------------------------
def piece_mesh(geom, m=None):
    t = geom.tag
    if t == "box":
        return trimesh.creation.box(extents=[float(x) for x in geom.get("size").split()])
    if t == "cylinder":
        return trimesh.creation.cylinder(radius=float(geom.get("radius")), height=float(geom.get("length")), sections=32)
    if t == "sphere":
        return trimesh.creation.icosphere(subdivisions=2, radius=float(geom.get("radius")))
    if t == "capsule":
        return trimesh.creation.capsule(radius=float(geom.get("radius")), height=float(geom.get("length")))
    if t == "mesh":
        path = os.path.join(os.path.dirname(m.path), geom.get("filename"))
        mesh = trimesh.load(path, force="mesh")
        if geom.get("scale"):
            mesh.apply_scale([float(s) for s in geom.get("scale").split()])
        return mesh
    raise ValueError(f"unsupported geometry {t}")


def merged_tool_mesh(m, tool_link, tcp_frame, kind):
    """Every <kind> geometry of the tool link merged into one mesh expressed in the TCP frame."""
    T_to_tcp = np.linalg.inv(m.kin.link_T(tcp_frame)) @ m.kin.link_T(tool_link)
    parts = []
    for geom, T, _ in geometry_pieces(m.links[tool_link], kind):
        mesh = piece_mesh(geom, m).copy()
        mesh.apply_transform(T_to_tcp @ T)
        parts.append(mesh)
    return trimesh.util.concatenate(parts)


ROBOTS = {
    "dda": dict(
        name="dda_rb10_1300e", file="rb10_1300e_DDA.urdf", rail="link_21_rail_DDA", rail_joint="joint_21_rail_DDA",
        track_joint="joint_1_scan_body_DDA", body="link_1_scan_body_DDA",
        slide_joint="joint_18_scan_body_slide_DDA", slide="link_18_scan_body_slide_DDA",
        mount_joint="dda_joint_mount", arm_root="dda_link0", link6="dda_link6", tool="link_0_DDA", tcp="tcp_DDA",
        old_end=dict(xyz=[0.0, -0.1163, 0.0], rpy=[0.0, 0.0, 0.0], mesh="link_dda.stl"),
        arm_links=[f"dda_link{i}" for i in range(7)],
        arm_joints=["dda_joint_base", "dda_joint_shoulder", "dda_joint_elbow", "dda_joint_wrist1", "dda_joint_wrist2",
                    "dda_joint_wrist3"],
    ),
    "rt": dict(
        name="rb20_1900es", file="rb20_1900es_RT.urdf", rail="link_20_rail_RT", rail_joint="joint_20_rail_RT",
        track_joint="joint_3_scan_body_RT", body="link_3_scan_body_RT",
        slide_joint="joint_19_scan_body_slide_RT", slide="link_19_scan_body_slide_RT",
        mount_joint="rt_joint_mount", arm_root="rt_link0", link6="rt_link6", tool="link_2_RT", tcp="tcp_RT",
        old_end=dict(xyz=[0.0, 0.0, 0.13], rpy=[-math.pi / 2, 0.0, 0.0], mesh="link_rt.stl"),
        arm_links=[f"rt_link{i}" for i in range(7)],
        arm_joints=["rt_base", "rt_shoulder", "rt_elbow", "rt_wrist1", "rt_wrist2", "rt_wrist3"],
    ),
}


def copy_link(m, src, new_name):
    l = copy.deepcopy(m.links[src])
    l.set("name", new_name)
    for tag in ("inertial",):
        for e in l.findall(tag):
            l.remove(e)
    return l


def export_robot(m, key, out_dir):
    c = ROBOTS[key]
    p = key
    robot = ET.Element("robot", name=c["name"])
    ET.SubElement(robot, "link", name=f"{p}_link_base")
    track_name, body_name, slide_name = f"{p}_link_linear_track", f"{p}_link_carriage", f"{p}_link_slide"

    robot.append(copy_link(m, c["rail"], track_name))
    robot.append(copy_link(m, c["body"], body_name))
    robot.append(copy_link(m, c["slide"], slide_name))
    for l in c["arm_links"]:
        robot.append(copy_link(m, l, l))

    # rail mount from the model's base_link
    mount = m.joints[c["rail_joint"]]
    robot.append(new_joint(f"{p}_joint_rail_mount", "fixed", f"{p}_link_base", track_name, T_of(mount)))

    # linear track: +q runs along world +x, zero at the -x end. The direction is read from the model: the CAD export
    # has +q towards -x (zero moved to the upper limit, axis flipped); a model converted by
    # urdf_separate_linear_tracks.py already has +q towards +x and lower=0 (kept as is).
    tj = m.joints[c["track_joint"]]
    ax = clean_axis(tj.find("axis").get("xyz"))
    lo, hi = float(tj.find("limit").get("lower")), float(tj.find("limit").get("upper"))
    _, ax_world = m.kin.joint_world(c["track_joint"])
    if ax_world[0] >= 0.0:
        T_track, ax_track = T_of(tj) @ trans(ax * lo), ax
    else:
        T_track, ax_track = T_of(tj) @ trans(ax * hi), -ax
    robot.append(new_joint(f"{p}_joint_linear_track", "prismatic", track_name, body_name,
                           T_track, ax_track, 0.0, hi - lo))

    # carriage slide: positive = towards the cell centre
    sj = m.joints[c["slide_joint"]]
    sax = clean_axis(sj.find("axis").get("xyz"))
    slo, shi = float(sj.find("limit").get("lower")), float(sj.find("limit").get("upper"))
    robot.append(new_joint(f"{p}_joint_carriage", "prismatic", body_name, slide_name, T_of(sj), sax, slo, shi))

    mj = m.joints[c["mount_joint"]]
    robot.append(new_joint(f"{p}_joint_mount", "fixed", slide_name, c["arm_root"], T_of(mj)))

    # arm joints verbatim (parent/child names are already the app names)
    for jn in c["arm_joints"]:
        j = copy.deepcopy(m.joints[jn])
        for e in j.findall("mimic"):
            j.remove(e)
        robot.append(j)

    # TCP / tool: one link whose frame is the TCP (z+ outward), one merged mesh (visual and collision)
    T_tcp = np.linalg.inv(m.kin.link_T(c["link6"])) @ m.kin.link_T(c["tcp"])
    if OLD_MESH_DIR:
        # previous end effector: its mesh and its end-frame position on link6; the frame orientation stays the
        # TCP convention (z+ outward) so the mesh is re-expressed in that frame
        oe = c["old_end"]
        T_old_end = make_T(oe["xyz"], rpy_to_R(oe["rpy"]))
        T_tcp = make_T(oe["xyz"], T_tcp[:3, :3])
    end_link = ET.SubElement(robot, "link", name=f"{p}_link_end")
    mesh_rel = f"meshes/tool_{key}.stl"
    if OLD_MESH_DIR:
        mesh = trimesh.load(os.path.join(OLD_MESH_DIR, oe["mesh"]), force="mesh")
        mesh.apply_scale(0.001)
        mesh.apply_transform(np.linalg.inv(T_tcp) @ T_old_end @ trans([0.0, 0.1163, 0.0]))
    else:
        mesh = merged_tool_mesh(m, c["tool"], c["tcp"], "collision")
    mesh.export(os.path.join(out_dir, mesh_rel))
    visual_mesh_rel = mesh_rel
    if not OLD_MESH_DIR:
        visual_mesh_rel = f"meshes/tool_{key}_visual.stl"
        merged_tool_mesh(m, c["tool"], c["tcp"], "visual").export(os.path.join(out_dir, visual_mesh_rel))
    for kind in ("visual", "collision"):
        e = ET.SubElement(end_link, kind)
        set_origin(e, np.eye(4))
        ET.SubElement(ET.SubElement(e, "geometry"), "mesh",
                      filename=visual_mesh_rel if kind == "visual" else mesh_rel)
        if kind == "visual":
            mat = ET.SubElement(e, "material", name="tool_black")
            ET.SubElement(mat, "color", rgba="0.1 0.1 0.1 1")
    robot.append(new_joint(f"{p}_joint_end", "fixed", c["link6"], f"{p}_link_end", T_tcp))

    # links first, then joints (the order the old files used)
    links = [e for e in robot if e.tag == "link"]
    joints = [e for e in robot if e.tag == "joint"]
    for e in list(robot):
        robot.remove(e)
    for e in links + joints:
        robot.append(e)
    write_urdf(robot, os.path.join(out_dir, c["file"]))
    return {
        "linear_track": [0.0, round(hi - lo, 4)], "carriage": [slo, shi],
        "tcp_in_link6_xyz": [round(float(v), 5) for v in T_tcp[:3, 3]],
        "tool_mesh": mesh_rel, "tool_mesh_bounds": np.round(mesh.bounds, 4).tolist(),
    }


OLD_MESH_DIR = None


def main():
    global OLD_MESH_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="urdf/model.urdf")
    ap.add_argument("--out-dir", default="urdf")
    ap.add_argument("--old-tool-meshes", default="",
                    help="optional legacy tool mesh directory; by default use the model's tools and TCP")
    a = ap.parse_args()
    OLD_MESH_DIR = a.old_tool_meshes or None
    m = Model(a.model)
    os.makedirs(os.path.join(a.out_dir, "meshes"), exist_ok=True)
    report = {"positioner": export_positioner(m, a.out_dir)}
    for key in ROBOTS:
        report[key] = export_robot(m, key, a.out_dir)
    path = os.path.join(a.out_dir, "app_models.report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
