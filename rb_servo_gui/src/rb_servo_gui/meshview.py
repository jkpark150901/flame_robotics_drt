"""URDF 로봇 메시를 두 자세(측정 / 원본)로 배치해 표면 차이를 화면에 그린다. matplotlib 만 사용(추가 설치 없음).

- 메시는 numpy 격자 군집(vertex clustering)으로 링크당 `max_faces` 개 안팎으로 줄여서 그린다 (그림용이며 변위 계산 정밀도에는 영향이 작음).
- 정사영 + 깊이 정렬(painter's algorithm) + 면 방향 음영으로 그린다. OpenGL 이 필요 없다.
- 색은 같은 메시 점의 **실제 변위(mm)** 이고, 형상도 실제 크기(×1) 그대로 그린다. 최악 변위도 ×1 기준이다.
"""
import os

import numpy as np
from matplotlib import colormaps
from matplotlib.collections import PolyCollection


def decimate(V, F, max_faces):
    """격자 군집으로 삼각형 수를 max_faces 이하로 줄인다 (정점을 격자 셀 평균으로 합치고 퇴화/중복 면 제거)."""
    V, F = np.asarray(V, float), np.asarray(F, int)
    if len(F) <= max_faces:
        return V, F
    lo_v = V.min(0)
    diag = float(np.linalg.norm(V.max(0) - lo_v))

    def cluster(cell):
        key = np.floor((V - lo_v) / cell).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        n = inv.max() + 1
        cnt = np.bincount(inv, minlength=n).astype(float)
        Vn = np.column_stack([np.bincount(inv, weights=V[:, k], minlength=n) / cnt for k in range(3)])
        Fn = inv[F]
        ok = (Fn[:, 0] != Fn[:, 1]) & (Fn[:, 1] != Fn[:, 2]) & (Fn[:, 0] != Fn[:, 2])
        Fn = np.unique(np.sort(Fn[ok], axis=1), axis=0)
        return Vn, Fn

    lo, hi = diag / 3000.0, diag / 3.0
    best = cluster(hi)
    for _ in range(18):
        mid = float(np.sqrt(lo * hi))
        cand = cluster(mid)
        if len(cand[1]) > max_faces:
            lo = mid
        else:
            hi, best = mid, cand
    return best


class MeshScene:
    def __init__(self, urdf_path, max_faces=1800, kind="collision"):
        import pinocchio as pin
        import trimesh

        self.pin = pin
        urdf_path = os.path.abspath(os.path.expanduser(urdf_path))
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        gtype = pin.GeometryType.COLLISION if kind == "collision" else pin.GeometryType.VISUAL
        geom = pin.buildGeomFromUrdf(self.model, urdf_path, gtype, package_dirs=[os.path.dirname(urdf_path)])
        from .surface import geom_mesh

        self.nq = self.model.nq
        first_arm = self.nq - 5
        self.links = []  # (이름, 부모 관절, 배치 SE3, 정점(로컬), 면)
        self.moving, self.struct = set(), set()
        for go in geom.geometryObjects:
            mesh, prim = geom_mesh(go)
            V, F = decimate(mesh.vertices, mesh.faces, max_faces)
            self.links.append((go.name, go.parentJoint, go.placement, V, F))
            if go.parentJoint >= first_arm:
                self.moving.add(go.name)
            if prim:
                self.struct.add(go.name)
        self.names = [l[0] for l in self.links if l[0] not in self.struct]

    def world(self, arm_rad):
        """{링크: 세계 좌표 정점 [m]}. 모델에 외부축 등 앞쪽 관절이 더 있으면 0 으로 둔다."""
        q = np.zeros(self.nq)
        arm = np.asarray(arm_rad, float)
        q[self.nq - len(arm):] = arm
        self.pin.forwardKinematics(self.model, self.data, q)
        out = {}
        for name, jid, pl, V, F in self.links:
            X = self.data.oMi[jid] * pl
            out[name] = V @ X.rotation.T + X.translation
        return out

    def displacement(self, arm_meas, arm_ref):
        """같은 메시 점의 변위 [mm]. 반환: ({링크: (n,)}, 링크별 최대, 전체 최대, 최악 링크)"""
        A, B = self.world(arm_meas), self.world(arm_ref)
        d = {n: np.linalg.norm(A[n] - B[n], axis=1) * 1000.0 for n in A}
        lmax = {n: float(v.max()) for n, v in d.items() if n not in self.struct}
        worst = max((n for n in lmax if n in self.moving), key=lmax.get)
        return d, lmax, lmax[worst], worst


def _view_basis(azim, elev):
    az, el = np.deg2rad(azim), np.deg2rad(elev)
    c = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])  # 로봇 -> 카메라 방향
    right = np.cross([0.0, 0.0, 1.0], c)
    if np.linalg.norm(right) < 1e-6:  # 정확히 위에서 볼 때
        right = np.array([1.0, 0.0, 0.0]) if el > 0 else np.array([-1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    up = np.cross(c, right)
    return c, right, up


def render(ax, scene, arm_meas, arm_ref, azim=45, elev=20, focus=None, vmax=None, show_ref=True, cmap="RdYlGn_r", show_struct=False):
    """ax 에 두 자세의 메시를 그린다. 측정 자세는 변위로 색칠, 기준 자세는 회색 반투명.

    반환: {"collections": [...], "vmax": 사용한 색 상한 [mm], "per_link": {링크: 최대 변위 mm}, "worst": (링크, mm), "scale": cmap 용 Normalize 인자}
    """
    d, lmax, dmax, worst = scene.displacement(arm_meas, arm_ref)
    A, B = scene.world(arm_meas), scene.world(arm_ref)
    c, right, up = _view_basis(azim, elev)
    vmax = vmax if vmax and vmax > 0 else max(dmax, 1e-6)
    cm = colormaps[cmap]
    names = [n for n in (scene.names + (sorted(scene.struct) if show_struct else [])) if (focus in (None, "전체") or n == focus)]

    def draw(polys3d, colors, z, alpha=1.0):
        pc = PolyCollection(polys3d, facecolors=colors, edgecolors="none", linewidths=0, zorder=z, alpha=alpha)
        ax.add_collection(pc)
        return pc

    cols, bounds = [], []
    ref_p, ref_c, mea_p, mea_c = [], [], [], []
    for name, jid, pl, V, F in scene.links:
        if name not in names:
            continue
        Vr = B[name]
        Vd = A[name]                           # 측정 자세 (실제 크기)
        for P, bucket_p, bucket_c, is_meas in ((Vr, ref_p, ref_c, False), (Vd, mea_p, mea_c, True)):
            if not is_meas and not show_ref:
                continue
            tri = P[F]                                       # (nf,3,3)
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            shade = 0.55 + 0.45 * np.abs(n @ c)               # 두 면 모두 보이게 |n·c|
            depth = tri.mean(1) @ c
            scr = np.stack([tri @ right, tri @ up], axis=-1)  # (nf,3,2)
            if is_meas:
                fd = d[name][F].mean(1)                       # 면 색 = 세 정점 변위의 평균 (실제 변위)
                base = cm(np.clip(fd / vmax, 0, 1))[:, :3]
            else:
                base = np.tile(np.array([0.62, 0.64, 0.68]), (len(F), 1))
            col = np.clip(base * shade[:, None], 0, 1)
            bucket_p.append((depth, scr)); bucket_c.append(col)
            bounds.append(scr.reshape(-1, 2))
    ax.clear()
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    colls = []
    for bucket_p, bucket_c, z, alpha in ((ref_p, ref_c, 1, 0.38), (mea_p, mea_c, 2, 1.0)):
        if not bucket_p:
            continue
        depth = np.concatenate([b[0] for b in bucket_p]); scr = np.concatenate([b[1] for b in bucket_p]); col = np.concatenate(bucket_c)
        order = np.argsort(depth)                              # 먼 면부터 그린다
        colls.append(draw(scr[order], col[order], z, alpha))
    if bounds:
        B_ = np.vstack(bounds)
        lo, hi = B_.min(0), B_.max(0)
        pad = 0.08 * float(np.max(hi - lo))
        ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(lo[1] - pad, hi[1] + pad)
    return {"collections": colls, "vmax": vmax, "per_link": lmax, "worst": (worst, dmax), "cmap": cmap}
