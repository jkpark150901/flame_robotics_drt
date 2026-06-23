#!/usr/bin/env python3
"""
두 PCD 폴더 비교 평가  (시각화 없음)

두 폴더의 PCD를 파일명으로 매칭한 뒤, ICP 정합 후
점 개수 차이와 평균 최근접 점거리만 출력한다.
(필터링은 pcd_filter.py 로 분리되어 있음)

사용 예:
  python pcd_eval.py MY_DIR REF_DIR --scale-a 1.0 --scale-b 1e-3
"""

import argparse
import pathlib
import sys

import numpy as np

# 프로젝트 python/ 디렉토리를 import 경로에 추가 (util.pcd_tool 사용)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from util.pcd_tool import load_pcd, compare_with_reference

PCD_EXTS = {".pcd", ".ply"}


def list_pcds(folder):
    folder = pathlib.Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"폴더가 아닙니다: {folder}")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in PCD_EXTS)


def match_files(files_a, files_b):
    """두 폴더 파일을 파일명(stem)으로 매칭. 공통이 없으면 정렬 순서로 짝짓기."""
    a = {p.stem: p for p in files_a}
    b = {p.stem: p for p in files_b}
    common = sorted(set(a) & set(b))
    if common:
        return [(s, a[s], b[s]) for s in common]
    if len(files_a) == len(files_b) and files_a:
        print("[!] 파일명이 일치하지 않아 정렬 순서로 짝짓습니다.")
        return [(pa.stem, pa, pb) for pa, pb in zip(files_a, files_b)]
    return []


def main():
    parser = argparse.ArgumentParser(
        description="두 PCD 폴더를 ICP 정합 후 비교 평가 (시각화 없음)")
    parser.add_argument("dir_a", help="결과 폴더 A (내 필터 결과 등)")
    parser.add_argument("dir_b", help="기준 폴더 B (상용툴 결과 등)")
    parser.add_argument("--scale-a", type=float, default=1.0, help="A 단위 변환")
    parser.add_argument("--scale-b", type=float, default=1.0, help="B 단위 변환")
    parser.add_argument("--icp-threshold", type=float, default=0.05, help="ICP 대응점 최대 거리 (m)")
    args = parser.parse_args()

    files_a = list_pcds(args.dir_a)
    files_b = list_pcds(args.dir_b)
    pairs = match_files(files_a, files_b)
    if not pairs:
        print("매칭되는 파일 쌍이 없습니다. 파일명 또는 개수를 확인하세요.")
        return

    print(f"평가: {len(pairs)}쌍  (A={args.dir_a}  vs  B={args.dir_b})")
    print(f"{'file':<28}{'nA':>10}{'nB':>10}{'diff':>9}{'diff%':>8}{'mean_d(m)':>11}{'icp_rmse':>10}")
    print("-" * 86)

    agg = {"count_diff": [], "diff_ratio": [], "mean_sym": [], "rmse": []}
    for stem, pa, pb in pairs:
        ra = load_pcd(pa, scale=args.scale_a)
        rb = load_pcd(pb, scale=1.0)   # B의 단위 변환은 compare 내부(ref_scale)에서
        cmp = compare_with_reference(ra, rb, icp_threshold=args.icp_threshold,
                                     ref_scale=args.scale_b)
        agg["count_diff"].append(cmp["count_diff"])
        agg["diff_ratio"].append(cmp["count_diff_ratio"])
        agg["mean_sym"].append(cmp["mean_dist_symmetric"])
        agg["rmse"].append(cmp["icp_rmse"])
        name = stem if len(stem) <= 26 else stem[:25] + "…"
        print(f"{name:<28}{cmp['n_result']:>10,}{cmp['n_reference']:>10,}"
              f"{cmp['count_diff']:>+9,}{cmp['count_diff_ratio']*100:>+7.1f}%"
              f"{cmp['mean_dist_symmetric']:>11.5f}{cmp['icp_rmse']:>10.5f}")

    print("-" * 86)
    print(f"{'평균':<28}{'':>10}{'':>10}{np.mean(agg['count_diff']):>+9,.0f}"
          f"{np.mean(agg['diff_ratio'])*100:>+7.1f}%"
          f"{np.mean(agg['mean_sym']):>11.5f}{np.mean(agg['rmse']):>10.5f}")
    print()
    print("해석: 대칭 평균거리(mean_d)가 작을수록 두 필터 결과의 형상이 일치.")
    print("      diff(+)는 A가 B보다 점을 더 많이 남겼음을 의미.")


if __name__ == "__main__":
    main()
