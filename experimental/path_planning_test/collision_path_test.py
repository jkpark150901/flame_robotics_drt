#!/usr/bin/env python3
"""
충돌 검증 + 경로계획 최소 테스트 (standalone)

목적:
  - 메시(워크피스 STL/PLY)를 정적 장애물로 놓고
  - 수기 입력한 start/goal 자세 사이를 플래너(기본 RRT)로 계획
  - 결과 경로가 정말 충돌이 없는지 독립적으로 재검증(광선 + 점유 occupancy)

simtool/viewervedo는 이미 메시 자세를 들고 있으므로, 검증되면 이 로직을 그대로
UI(zapi 명령)로 옮겨 목표 자세만 받아 호출하면 됨. 지금은 수기 입력 단계.

사용 예:
  python collision_path_test.py --mesh ../../sample/"PIPE NO.1_fill.stl"
  python collision_path_test.py --start 700 -200 100 0 0 0 --goal -300 -1700 100 0 0 0 --vis
"""

import argparse
import pathlib
import sys

import numpy as np
import open3d as o3d

# 플래너 플러그인 import 경로 (repo/python)
_PYROOT = pathlib.Path(__file__).resolve().parents[2] / "python"
sys.path.insert(0, str(_PYROOT))


def load_planner(name):
    """이름으로 플래너 플러그인 인스턴스 반환."""
    import importlib
    mod = importlib.import_module(f"plugins.pathplanner.{name}")
    # 모듈에서 PlannerBase 하위 클래스 찾기
    from plugins.pluginbase.plannerbase import PlannerBase
    import inspect
    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if issubclass(obj, PlannerBase) and obj is not PlannerBase:
            return obj()
    raise RuntimeError(f"플래너 클래스 못 찾음: {name}")


def build_scene(mesh):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def verify_path(scene, path, n_dense=2000):
    """경로를 독립 재검증.
      1) 세그먼트 광선 검사(플래너와 동일): p1→p2 사이에 메시가 있으면 충돌
      2) 경로를 촘촘히 샘플해 점유(occupancy) 검사: 솔리드 내부면 관통
    반환: dict(ray_hits, penetrations, n_samples)
    """
    path = np.asarray(path, dtype=float)[:, :3]
    # 1) 세그먼트 광선
    ray_hits = 0
    for a, b in zip(path[:-1], path[1:]):
        d = b - a
        L = np.linalg.norm(d)
        if L < 1e-9:
            continue
        d = d / L
        rays = o3d.core.Tensor([[*a, *d]], dtype=o3d.core.Dtype.Float32)
        t_hit = scene.cast_rays(rays)['t_hit'][0].item()
        if np.isfinite(t_hit) and t_hit < L:
            ray_hits += 1
    # 2) 촘촘한 점유 검사
    dense = []
    for a, b in zip(path[:-1], path[1:]):
        seg = np.linalg.norm(b - a)
        k = max(2, int(seg / max(np.ptp(path, axis=0).max(), 1) * n_dense / max(len(path), 1)) + 2)
        dense.append(np.linspace(a, b, k))
    dense = np.vstack(dense) if dense else path
    q = o3d.core.Tensor(dense, dtype=o3d.core.Dtype.Float32)
    occ = scene.compute_occupancy(q).numpy()
    penetrations = int(np.count_nonzero(occ > 0.5))
    return {"ray_hits": ray_hits, "penetrations": penetrations, "n_samples": len(dense)}


def main():
    ap = argparse.ArgumentParser(description="충돌 검증 + 경로계획 최소 테스트")
    ap.add_argument("--mesh", default=str(_PYROOT.parent / "sample" / "PIPE NO.1_fill.stl"))
    ap.add_argument("--planner", default="rrt", help="플래너 플러그인 모듈명 (rrt, rrt_star, ...)")
    ap.add_argument("--start", nargs=6, type=float, default=None, metavar="V",
                    help="시작 자세 x y z r p y (미지정 시 메시 bbox에서 자동)")
    ap.add_argument("--goal", nargs=6, type=float, default=None, metavar="V",
                    help="목표 자세 x y z r p y (미지정 시 메시 bbox에서 자동)")
    ap.add_argument("--margin", type=float, default=0.15, help="작업공간 bbox 여유 비율")
    ap.add_argument("--max-iter", type=int, default=5000)
    ap.add_argument("--step-frac", type=float, default=0.02, help="step_size = bbox대각 * 이 비율")
    ap.add_argument("--vis", action="store_true", help="open3d 시각화")
    args = ap.parse_args()

    if not pathlib.Path(args.mesh).exists():
        print(f"[!] 메시 파일 없음: {args.mesh}")
        return
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    mesh.compute_vertex_normals()
    mn, mx = mesh.get_min_bound(), mesh.get_max_bound()
    ext = mx - mn
    diag = float(np.linalg.norm(ext))
    print(f"메시: {args.mesh}")
    print(f"  bbox min={np.round(mn,1)} max={np.round(mx,1)} (대각 {diag:.1f})")

    # start/goal 자동 기본값: bbox 바깥 양 끝 (충돌 없는 경로가 메시를 돌아가야 함)
    pad = ext * args.margin
    if args.start is None:
        args.start = [float(mx[0] + pad[0]), float(mn[1] - pad[1]), float((mn[2]+mx[2])/2), 0, 0, 0]
    if args.goal is None:
        args.goal = [float(mn[0] - pad[0]), float(mx[1] + pad[1]), float((mn[2]+mx[2])/2), 0, 0, 0]
    print(f"  start={np.round(args.start,1)}")
    print(f"  goal ={np.round(args.goal,1)}")

    # 플래너 구성: bbox 기준 bounds/step 설정
    planner = load_planner(args.planner)
    planner.bounds = {
        "x_min": float(mn[0]-pad[0]), "x_max": float(mx[0]+pad[0]),
        "y_min": float(mn[1]-pad[1]), "y_max": float(mx[1]+pad[1]),
        "z_min": float(mn[2]-pad[2]), "z_max": float(mx[2]+pad[2]),
    }
    if hasattr(planner, "step_size"):
        planner.step_size = diag * args.step_frac
    if hasattr(planner, "max_iter"):
        planner.max_iter = args.max_iter
    planner.add_collision_object(mesh)
    print(f"플래너: {type(planner).__name__}  step_size={getattr(planner,'step_size','?'):.2f}  max_iter={getattr(planner,'max_iter','?')}")

    # 계획
    import time
    t0 = time.time()
    path = planner.generate(np.array(args.start, float), np.array(args.goal, float))
    dt = time.time() - t0
    if not path:
        print(f"[실패] 경로를 찾지 못함 ({dt:.2f}s). max-iter/step 조정 필요.")
        return
    path = [np.asarray(p, float) for p in path]
    print(f"[성공] 웨이포인트 {len(path)}개 ({dt:.2f}s)")

    # 독립 충돌 재검증
    scene = build_scene(mesh)
    res = verify_path(scene, path)
    print("── 충돌 재검증 ──")
    print(f"  세그먼트 광선 충돌: {res['ray_hits']} (경로 세그먼트 {len(path)-1}개 중)")
    print(f"  점유(관통) 샘플   : {res['penetrations']} / {res['n_samples']}")
    ok = res['ray_hits'] == 0 and res['penetrations'] == 0
    print(f"  결과: {'충돌 없음 ✓' if ok else '⚠ 충돌 검출됨'}")

    if args.vis:
        geoms = [mesh]
        mesh.paint_uniform_color([0.6, 0.6, 0.6])
        pts = np.array([p[:3] for p in path])
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector([[i, i+1] for i in range(len(pts)-1)]))
        ls.paint_uniform_color([0.1, 0.8, 0.1])
        geoms.append(ls)
        for p, c in [(args.start, [1, 0, 0]), (args.goal, [0, 0, 1])]:
            s = o3d.geometry.TriangleMesh.create_sphere(radius=diag*0.01)
            s.translate(np.asarray(p[:3])); s.paint_uniform_color(c)
            geoms.append(s)
        o3d.visualization.draw_geometries(geoms, window_name="path planning collision test")


if __name__ == "__main__":
    main()
