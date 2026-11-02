"""URDF 로봇의 링크 표면(메시)을 두 자세에서 비교한다: 관절각 오차가 실제 형상에서 표면 위치 차이로 얼마나 나타나는가.

- 측정 자세 q 와 DTW 로 짝지은 원본 자세 p 에서 각각 FK 로 링크 메시(표면 샘플점)를 배치하고 비교한다.
- 대응점 변위: 같은 메시 점이 두 자세에서 얼마나 떨어졌나 (링크별 최대/평균/p95). 최악의 점을 보여준다.
- Chamfer / Hausdorff: 대응점 없이 두 표면이 서로 얼마나 가까운가 (최근접 거리의 평균 / 최대). 형상 차이 관점.
  대칭 형상이 자기 축으로 도는 오차(예: 원통형 말단의 손목 회전)는 표면이 거의 겹쳐서 Chamfer 가 작지만 대응점 변위는 크다.
단위: URDF 길이 단위(m) -> mm 로 보고한다. 외부축 등 앞쪽 관절은 0 으로 두며, 두 자세에 같이 적용되므로 상대 차이에는 영향이 없다.
"""
import os

import numpy as np


def geom_mesh(go):
    """pinocchio GeometryObject -> (trimesh, 기본도형 여부). 메시 파일이 있으면 그것을, 박스/원기둥/구/캡슐은 직접 만든다."""
    import trimesh

    g = go.geometry
    kind = type(g).__name__
    if os.path.exists(go.meshPath):
        mesh = trimesh.load(go.meshPath, force="mesh")
        mesh.apply_scale(np.asarray(go.meshScale, float))
        return mesh, False
    if kind == "Box":
        return trimesh.creation.box(extents=2.0 * np.asarray(g.halfSide, float)), True
    if kind == "Cylinder":
        return trimesh.creation.cylinder(radius=float(g.radius), height=2.0 * float(g.halfLength)), True
    if kind == "Sphere":
        return trimesh.creation.icosphere(radius=float(g.radius)), True
    if kind == "Capsule":
        return trimesh.creation.capsule(radius=float(g.radius), height=2.0 * float(g.halfLength)), True
    raise ValueError(f"지원하지 않는 도형: {kind} ({go.name})")


class RobotSurface:
    def __init__(self, urdf_path, n_points=600, kind="collision", seed=0):
        import pinocchio as pin
        import trimesh

        self.pin = pin
        urdf_path = os.path.abspath(os.path.expanduser(urdf_path))
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        gtype = pin.GeometryType.COLLISION if kind == "collision" else pin.GeometryType.VISUAL
        geom = pin.buildGeomFromUrdf(self.model, urdf_path, gtype, package_dirs=[os.path.dirname(urdf_path)])
        rng_seed = int(seed)
        self.nq = self.model.nq
        first_arm = self.nq - 5                  # 팔 6관절은 모델의 마지막 6개 (관절 id 는 1부터)
        self.links = []  # (이름, 부모 관절 인덱스, 배치 SE3, 로컬 표면 샘플점 (n,3))
        self.moving, self.struct = set(), set()  # moving: 팔 관절과 함께 움직이는 링크 / struct: 레일 등 기본도형 구조물
        for go in geom.geometryObjects:
            mesh, prim = geom_mesh(go)
            pts, _ = trimesh.sample.sample_surface(mesh, int(n_points), seed=rng_seed)
            self.links.append((go.name, go.parentJoint, go.placement, np.asarray(pts, float)))
            if go.parentJoint >= first_arm:
                self.moving.add(go.name)
            if prim:
                self.struct.add(go.name)
        self.joint_names = list(self.model.names)[1:]

    def full_q(self, arm_rad):
        """팔 6관절(rad)을 모델 q 로. 모델에 외부축 등 앞쪽 관절이 더 있으면 0 으로 둔다."""
        q = np.zeros(self.nq)
        arm = np.asarray(arm_rad, float)
        q[self.nq - len(arm):] = arm
        return q

    def points(self, arm_rad):
        """각 링크 표면 샘플점의 세계 좌표 [m]. 반환: {링크 이름: (n,3)}"""
        pin = self.pin
        pin.forwardKinematics(self.model, self.data, self.full_q(arm_rad))
        out = {}
        for name, jid, placement, pts in self.links:
            X = self.data.oMi[jid] * placement
            out[name] = pts @ X.rotation.T + X.translation
        return out


def compare_poses(robot, arm_a, arm_b):
    """두 자세(측정 a, 원본 b)의 링크 표면 비교. 반환 dict (모두 mm)."""
    from scipy.spatial import cKDTree

    A, B = robot.points(arm_a), robot.points(arm_b)
    res = {"link_max": {}, "link_mean": {}}
    allA, allB = [], []
    for name in A:
        if name in robot.struct:                      # 레일/캐리지 같은 구조물은 팔 오차와 무관해서 제외
            continue
        d = np.linalg.norm(A[name] - B[name], axis=1) * 1000.0  # 같은 메시 점의 변위
        res["link_max"][name] = float(d.max())
        res["link_mean"][name] = float(d.mean())
        if name in robot.moving:                      # 전체 로봇 지표(Chamfer 등)는 움직이는 링크만: 고정 링크는 값을 희석한다
            allA.append(A[name]); allB.append(B[name])
    PA, PB = np.vstack(allA), np.vstack(allB)
    dd = np.linalg.norm(PA - PB, axis=1) * 1000.0
    res["disp_max"], res["disp_mean"], res["disp_p95"] = float(dd.max()), float(dd.mean()), float(np.percentile(dd, 95))
    dab = cKDTree(PB).query(PA)[0] * 1000.0   # a 의 각 점 -> b 표면
    dba = cKDTree(PA).query(PB)[0] * 1000.0   # b 의 각 점 -> a 표면
    res["chamfer"] = float(0.5 * (dab.mean() + dba.mean()))
    res["hausdorff"] = float(max(dab.max(), dba.max()))
    return res
