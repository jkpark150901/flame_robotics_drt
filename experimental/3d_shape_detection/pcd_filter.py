#!/usr/bin/env python3
"""
PCD 옥트리(복셀) CCL 필터링  (시각화 없음)

octree_ccl 노트북에서 시각화를 걷어내고 필터링 처리만 남긴 CLI.
핵심 함수는 util.pcd_tool 의 것을 재사용한다.
(평가는 pcd_eval.py 로 분리되어 있음)

사용 예:
  python pcd_filter.py IN_DIR OUT_DIR --level 7 --min-points 30 --scale 1e-3
"""

import argparse
import pathlib
import sys
import time

import numpy as np

# 프로젝트 python/ 디렉토리를 import 경로에 추가 (util.pcd_tool 사용)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from util.pcd_tool import load_pcd, voxel_ccl, save_pcd

PCD_EXTS = {".pcd", ".ply"}


def level_to_voxel(pcd, level, size_expand=0.01):
    """옥트리 레벨 → 복셀 한 변 길이. (root 큐브 크기 / 2**level)

    옥트리를 실제로 만들지 않고 바운딩 큐브로부터 직접 계산한다.
    """
    pts = np.asarray(pcd.points)
    root = float((pts.max(axis=0) - pts.min(axis=0)).max()) * (1 + size_expand)
    return root / (2 ** level)


def filter_largest_component(pcd, voxel_size, min_points=30, connectivity=26):
    """복셀 CCL로 가장 큰 연결요소만 남긴 PointCloud 반환."""
    _, labels = voxel_ccl(np.asarray(pcd.points), voxel_size,
                          min_points=min_points, connectivity=connectivity)
    valid = labels[labels >= 0]
    if len(valid) == 0:
        raise ValueError("연결요소를 찾지 못했습니다. level/min_points를 조정하세요.")
    uniq, cnts = np.unique(valid, return_counts=True)
    target = uniq[np.argmax(cnts)]
    sel = np.where(labels == target)[0]
    return pcd.select_by_index(sel)


def filter_pcd(pcd, level, min_points=30, connectivity=26):
    """레벨로 복셀 크기를 정하고 CCL 필터링. (clean_pcd, voxel_size) 반환."""
    voxel = level_to_voxel(pcd, level)
    clean = filter_largest_component(pcd, voxel, min_points, connectivity)
    return clean, voxel


def list_pcds(folder):
    folder = pathlib.Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"폴더가 아닙니다: {folder}")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in PCD_EXTS)


def main():
    parser = argparse.ArgumentParser(
        description="PCD 옥트리(복셀) CCL 필터링 (시각화 없음)")
    parser.add_argument("input_dir", help="입력 PCD 폴더")
    parser.add_argument("output_dir", help="출력 폴더")
    parser.add_argument("--scale", type=float, default=1.0, help="단위 변환 (mm→m면 1e-3)")
    parser.add_argument("--level", type=int, default=7, help="옥트리 레벨 (복셀 크기 결정)")
    parser.add_argument("--min-points", type=int, default=30, help="유지할 컴포넌트 최소 점 수")
    parser.add_argument("--connectivity", type=int, choices=[6, 26], default=26, help="복셀 인접성")
    parser.add_argument("--suffix", default="_clean", help="출력 파일명 접미사")
    args = parser.parse_args()

    in_dir = pathlib.Path(args.input_dir)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_pcds(in_dir)
    if not files:
        print(f"PCD/PLY 파일이 없습니다: {in_dir}")
        return

    print(f"필터링: {len(files)}개 파일  (level={args.level}, min_points={args.min_points}, conn={args.connectivity})")
    total_t = 0.0
    for f in files:
        t0 = time.perf_counter()
        pcd = load_pcd(f, scale=args.scale)
        t_load = time.perf_counter() - t0

        t1 = time.perf_counter()
        clean, voxel = filter_pcd(pcd, args.level, args.min_points, args.connectivity)
        t_filter = time.perf_counter() - t1

        t2 = time.perf_counter()
        out_path = out_dir / (f.stem + args.suffix + ".ply")
        save_pcd(clean, out_path)
        t_save = time.perf_counter() - t2

        elapsed = t_load + t_filter + t_save
        total_t += elapsed
        print(f"  {f.name}: {len(pcd.points):,} → {len(clean.points):,} "
              f"점  (voxel={voxel:.4f} m)  → {out_path.name}")
        print(f"    시간 {elapsed:.2f}s  (로드 {t_load:.2f}s / 필터 {t_filter:.2f}s / 저장 {t_save:.2f}s)")

    print(f"전체 소요 시간: {total_t:.2f}s  ({len(files)}개 파일)")


if __name__ == "__main__":
    main()
