#!/usr/bin/env python3
"""
PCD 필터 결과 평가  (시각화 없음)

원본 점군 대비 **줄어든 비율(reduction ratio)** 을 본다.
결과 폴더를 두 개 주면(내 결과 vs 상용툴) 감소율을 나란히 비교하고,
ICP 정합 후 형상 차이(평균 최근접 거리)도 함께 출력한다.

사용 예:
  python pcd_eval.py ORIG_DIR MY_DIR                    # 원본 대비 감소율
  python pcd_eval.py ORIG_DIR MY_DIR REF_DIR            # 두 결과 감소율 + 형상 비교
  python pcd_eval.py ORIG_DIR MY_DIR REF_DIR --scale-orig 1e-3
"""

import argparse
import pathlib
import sys

import numpy as np

# 프로젝트 python/ 디렉토리를 import 경로에 추가 (util.pcd_tool 사용)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from util.pcd_tool import load_pcd, compare_with_reference

PCD_EXTS = {".pcd", ".ply"}
# 결과 파일명에 흔히 붙는 접미사 — 원본과 매칭할 때 제거
SUFFIXES = ("_ccl", "_sor", "_clean", "_mesh", "_filtered")


def list_pcds(folder):
    folder = pathlib.Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"폴더가 아닙니다: {folder}")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in PCD_EXTS)


def norm_stem(stem):
    """결과 접미사를 제거해 원본 stem과 매칭되도록 정규화."""
    for s in SUFFIXES:
        if stem.endswith(s):
            return stem[: -len(s)]
    return stem


def point_count(path, scale=1.0):
    return len(load_pcd(path, scale=scale).points)


def match_to_orig(orig_files, result_files):
    """원본 파일 → 결과 파일 매핑 (정규화 stem 기준, 실패 시 정렬 순서)."""
    by_norm = {}
    for p in result_files:
        by_norm.setdefault(norm_stem(p.stem), p)
    mapping = {}
    for o in orig_files:
        mapping[o] = by_norm.get(o.stem)
    if any(v is None for v in mapping.values()) and len(orig_files) == len(result_files):
        # 정규화 매칭 실패 → 정렬 순서로 짝짓기
        mapping = dict(zip(orig_files, result_files))
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="원본 대비 감소율 평가 (시각화 없음)")
    parser.add_argument("orig_dir", help="원본 점군 폴더")
    parser.add_argument("dir_a", help="결과 폴더 A (내 필터 결과)")
    parser.add_argument("dir_b", nargs="?", default=None, help="결과 폴더 B (상용툴 결과, 선택)")
    parser.add_argument("--scale-orig", type=float, default=1.0, help="원본 단위 변환")
    parser.add_argument("--scale-a", type=float, default=1.0, help="A 단위 변환")
    parser.add_argument("--scale-b", type=float, default=1.0, help="B 단위 변환")
    parser.add_argument("--icp-threshold", type=float, default=0.05,
                        help="A vs B ICP 대응점 최대 거리 (m)")
    args = parser.parse_args()

    orig_files = list_pcds(args.orig_dir)
    a_files = list_pcds(args.dir_a)
    b_files = list_pcds(args.dir_b) if args.dir_b else []
    if not orig_files:
        print("원본 폴더에 PCD/PLY가 없습니다.")
        return

    map_a = match_to_orig(orig_files, a_files)
    map_b = match_to_orig(orig_files, b_files) if b_files else {}

    has_b = bool(b_files)
    print(f"평가: 원본 {len(orig_files)}개 기준  (A={args.dir_a}"
          + (f"  B={args.dir_b}" if has_b else "") + ")")
    if has_b:
        header = (f"{'file':<24}{'n_orig':>10}{'n_A':>10}{'A감소%':>9}"
                  f"{'n_B':>10}{'B감소%':>9}{'mean_d(m)':>11}")
    else:
        header = f"{'file':<24}{'n_orig':>10}{'n_A':>10}{'A감소%':>9}"
    print(header)
    print("-" * len(header))

    agg = {"red_a": [], "red_b": [], "mean_d": []}
    for o in orig_files:
        pa = map_a.get(o)
        if pa is None:
            continue
        n_orig = point_count(o, scale=args.scale_orig)
        n_a = point_count(pa, scale=args.scale_a)
        red_a = (1 - n_a / n_orig) * 100 if n_orig else float("nan")
        agg["red_a"].append(red_a)
        name = o.stem if len(o.stem) <= 22 else o.stem[:21] + "…"

        if has_b:
            pb = map_b.get(o)
            if pb is not None:
                n_b = point_count(pb, scale=args.scale_b)
                red_b = (1 - n_b / n_orig) * 100 if n_orig else float("nan")
                ra = load_pcd(pa, scale=args.scale_a)
                rb = load_pcd(pb, scale=1.0)
                cmp = compare_with_reference(ra, rb, icp_threshold=args.icp_threshold,
                                             ref_scale=args.scale_b)
                mean_d = cmp["mean_dist_symmetric"]
                agg["red_b"].append(red_b)
                agg["mean_d"].append(mean_d)
                print(f"{name:<24}{n_orig:>10,}{n_a:>10,}{red_a:>8.1f}%"
                      f"{n_b:>10,}{red_b:>8.1f}%{mean_d:>11.5f}")
            else:
                print(f"{name:<24}{n_orig:>10,}{n_a:>10,}{red_a:>8.1f}%"
                      f"{'-':>10}{'-':>9}{'-':>11}")
        else:
            print(f"{name:<24}{n_orig:>10,}{n_a:>10,}{red_a:>8.1f}%")

    print("-" * len(header))
    if has_b:
        print(f"{'평균':<24}{'':>10}{'':>10}{np.mean(agg['red_a']):>8.1f}%"
              f"{'':>10}{np.mean(agg['red_b']):>8.1f}%"
              f"{np.mean(agg['mean_d']):>11.5f}")
    else:
        print(f"{'평균':<24}{'':>10}{'':>10}{np.mean(agg['red_a']):>8.1f}%")
    print()
    print("해석: 감소% = 원본 대비 제거된 점 비율 (클수록 많이 걸러냄).")
    if has_b:
        print("      mean_d 작을수록 A·B 형상이 일치. A/B 감소%를 비교해 과/소필터 판단.")


if __name__ == "__main__":
    main()
