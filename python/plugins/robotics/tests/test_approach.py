import numpy as np
import pytest

from plugins.robotics.inspection_planning_base import (
    ApproachPlanningError,
    ApproachSpec,
    InspectionPlanningBase,
)


class _FakeBackend:
    """set_link_clearance만 흉내 낸다 - link_clearance()의 적용/해제 규약을 검증하기 위한 것."""

    def __init__(self, applied=3):
        self.applied = applied
        self.calls = []

    def set_link_clearance(self, robot_name, link_names, margin):
        self.calls.append((robot_name, list(link_names), float(margin)))
        return self.applied if (link_names and margin > 0) else 0


def test_spec_normalizes_facing_axis():
    spec = ApproachSpec(facing_axis_local=[0.0, -2.0, 0.0], clearance_links=["dda_link_end"])
    assert spec.facing_axis_local == [0.0, -1.0, 0.0]


@pytest.mark.parametrize("kwargs", [
    {"min_clearance": 0.2, "standoff": 0.15},   # margin이 standoff 이상이면 pre/retreat 끝점이 스스로 margin을 어긴다
    {"min_clearance": 0.0},
    {"clearance_links": []},
    {"facing_axis_local": [0.0, 0.0, 0.0]},
])
def test_spec_rejects_inconsistent_settings(kwargs):
    base = {"facing_axis_local": [0.0, -1.0, 0.0], "clearance_links": ["dda_link_end"]}
    base.update(kwargs)
    with pytest.raises(ValueError):
        ApproachSpec(**base)


def test_link_clearance_applies_then_always_clears():
    backend = _FakeBackend()
    base = InspectionPlanningBase(backend)
    with base.link_clearance("dda", ["dda_link_end"], 0.1):
        assert backend.calls == [("dda", ["dda_link_end"], 0.1)]
    assert backend.calls[-1] == ("dda", [], 0.0)


def test_link_clearance_clears_even_when_body_raises():
    backend = _FakeBackend()
    base = InspectionPlanningBase(backend)
    with pytest.raises(RuntimeError):
        with base.link_clearance("dda", ["dda_link_end"], 0.1):
            raise RuntimeError("planner blew up")
    assert backend.calls[-1] == ("dda", [], 0.0)


def test_link_clearance_refuses_to_silently_skip_an_unenforced_margin():
    # backend가 margin을 하나도 못 걸었으면(미지원/장애물 없음) 제약이 안 걸린 채 계획이 "성공"하면 안 된다.
    backend = _FakeBackend(applied=0)
    base = InspectionPlanningBase(backend)
    with pytest.raises(ApproachPlanningError) as info:
        with base.link_clearance("dda", ["dda_link_end"], 0.1):
            pass
    assert info.value.reason == "clearance_not_enforced"


def test_link_clearance_noop_when_disabled():
    backend = _FakeBackend()
    base = InspectionPlanningBase(backend)
    with base.link_clearance("dda", [], 0.1):
        pass
    with base.link_clearance("dda", ["dda_link_end"], 0.0):
        pass
    assert backend.calls == []


def test_link_min_distance_matches_suffixed_geometry_names():
    class Backend(_FakeBackend):
        def link_obstacle_distances(self, robot_name, q):
            return [
                {"link": "dda_link_end_0", "obstacle": "pipe", "distance": 0.4},
                {"link": "dda_link_end_0", "obstacle": "positioner", "distance": 0.9},
                {"link": "dda_link6_0", "obstacle": "pipe", "distance": 0.05},
            ]

    base = InspectionPlanningBase(Backend())
    assert base.link_min_distance("dda", np.zeros(3), ["dda_link_end"]) == pytest.approx(0.4)
    assert base.link_min_distance("dda", np.zeros(3), ["not_a_link"]) is None
