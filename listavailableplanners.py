#!/usr/bin/env python3
"""
OMPL(Open Motion Planning Library) Python 바인딩에서
사용 가능한 path planning 알고리즘(플래너) 목록을 출력하는 스크립트.

요구사항:
    pip install ompl
    (또는 소스 빌드 시 PYTHONPATH에 ompl 파이썬 바인딩 경로 추가)
"""

from ompl import base as ob
from ompl import geometric as og
from ompl import control as oc


def list_planners(module, base_class):
    """module 안에서 base_class를 상속하는 플래너 클래스 이름들을 반환"""
    planners = []
    for name in dir(module):
        obj = getattr(module, name)
        try:
            if isinstance(obj, type) and issubclass(obj, base_class) and obj is not base_class:
                planners.append(name)
        except TypeError:
            continue
    return sorted(planners)


def main():
    print("=" * 50)
    print(" Geometric Planners (ompl.geometric)")
    print("=" * 50)
    for name in list_planners(og, ob.Planner):
        print(f" - {name}")

    print()
    print("=" * 50)
    print(" Control-based Planners (ompl.control)")
    print("=" * 50)
    for name in list_planners(oc, oc.Planner):
        print(f" - {name}")


if __name__ == "__main__":
    main()