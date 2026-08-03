import types
import importlib.util

import numpy as np
import pytest

from plugins.pathplanner.ompl.path_extractor import (
    ensure_exact_endpoints,
    extract_full_q_path,
)
from plugins.pathplanner.ompl.planner_factory import create_ompl_planner
from plugins.pathplanner.ompl.state_codec import JointStateCodec
from plugins.pathplanner.ompl.validity_adapter import StateValidityAdapter
from plugins.pathplanner.rrt_connect import RRTConnect
from plugins.pathplanner.rrt_star import RRTStar
from plugins.pathplanner.informed_rrt_star import InformedRRTStar
from plugins.pathplanner.bit_star import BITStar
from plugins.pathplanner.rrt import RRT


def test_joint_state_codec_round_trip_and_fixed_joint():
    codec = JointStateCodec(
        full_dof=4,
        lower=[-2.0, -1.0, 0.0, -4.0],
        upper=[2.0, 3.0, 10.0, 4.0],
        active_indices=[0, 2, 3],
        fixed_values={1: 0.75},
        reference_q=[0.0, 0.75, 0.0, 0.0],
    )
    q = np.asarray([1.0, -0.5, 2.5, -2.0])

    values = codec.full_q_to_state_values(q)
    restored = codec.state_to_full_q(values)

    assert np.allclose(values, [0.75, 0.25, 0.25])
    assert np.allclose(restored, [1.0, 0.75, 2.5, -2.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lower": [0.0], "upper": [0.0], "active_indices": [0]},
        {"lower": [0.0], "upper": [1.0], "active_indices": []},
        {
            "lower": [0.0, 0.0],
            "upper": [1.0, 1.0],
            "active_indices": [0],
            "fixed_values": {0: 0.5},
        },
    ],
)
def test_joint_state_codec_rejects_invalid_configuration(kwargs):
    full_dof = len(kwargs["lower"])
    with pytest.raises(ValueError):
        JointStateCodec(full_dof=full_dof, reference_q=np.zeros(full_dof), **kwargs)


def test_validity_adapter_uses_workspace_and_collision_checks():
    class Planner:
        planning_deadline = None

        def _workspace_position_ok(self, q):
            return q[0] <= 0.8

        def check_robot_collision(self, q):
            return q[1] >= 0.7

    codec = JointStateCodec(2, [0.0, 0.0], [1.0, 1.0], [0, 1], reference_q=[0.0, 0.0])
    stats = {
        "state_validity_calls": 0,
        "state_collision_rejects": 0,
        "workspace_rejects": 0,
        "deadline_rejects": 0,
        "callback_errors": 0,
    }
    adapter = StateValidityAdapter(Planner(), codec, stats)

    assert adapter([0.2, 0.2]) is True
    assert adapter([0.9, 0.2]) is False
    assert adapter([0.2, 0.8]) is False
    assert stats["state_validity_calls"] == 3
    assert stats["workspace_rejects"] == 1
    assert stats["state_collision_rejects"] == 1


def test_planner_factory_maps_all_supported_algorithms_and_parameters():
    created = []

    class FakePlanner:
        def __init__(self, si):
            self.si = si
            self.values = {}
            created.append(self)

        def setRange(self, value):
            self.values["range"] = value

        def setGoalBias(self, value):
            self.values["goal_bias"] = value

        def setRewireFactor(self, value):
            self.values["rewire_factor"] = value

        def setSamplesPerBatch(self, value):
            self.values["samples_per_batch"] = value

        def setPruning(self, value):
            self.values["pruning"] = value

    geometric = types.SimpleNamespace(
        RRT=FakePlanner,
        RRTConnect=FakePlanner,
        RRTstar=FakePlanner,
        InformedRRTstar=FakePlanner,
        BITstar=FakePlanner,
    )
    config = {
        "planner": {
            "range": 0.2,
            "goal_bias": 0.1,
            "rewire_factor": 1.2,
            "bit_star": {"samples_per_batch": 64, "pruning": True},
        }
    }
    for algorithm in ("rrt", "rrt_connect", "rrt_star", "informed_rrt_star", "bit_star"):
        planner = create_ompl_planner(algorithm, object(), config, geometric)
        assert planner.values["range"] == 0.2
    assert created[-1].values["samples_per_batch"] == 64
    assert created[-1].values["pruning"] is True


def test_path_extraction_removes_duplicates_and_restores_endpoints():
    codec = JointStateCodec(2, [-1.0, -1.0], [1.0, 1.0], [0, 1], reference_q=[0.0, 0.0])

    class Path:
        states = [[0.5, 0.5], [0.5, 0.5], [0.75, 0.25]]

        def getStateCount(self):
            return len(self.states)

        def getState(self, index):
            return self.states[index]

    class Planner:
        def _check_collision(self, a, b):
            return False

    path = extract_full_q_path(Path(), codec)
    path = ensure_exact_endpoints(path, [0.0, 0.0], [0.5, -0.5], Planner())

    assert len(path) == 2
    assert np.allclose(path[0], [0.0, 0.0])
    assert np.allclose(path[-1], [0.5, -0.5])


@pytest.mark.parametrize(
    "planner_cls, algorithm",
    [
        (RRTConnect, "rrt_connect"),
        (RRT, "rrt"),
        (RRTStar, "rrt_star"),
        (InformedRRTStar, "informed_rrt_star"),
        (BITStar, "bit_star"),
    ],
)
def test_existing_planner_modules_default_to_ompl(planner_cls, algorithm):
    planner = planner_cls()
    assert planner.planner_backend == "ompl"
    assert planner.algorithm == algorithm


def test_rrt_keeps_legacy_workspace_planning_behavior():
    planner = RRT()
    planner.goal_bias = 1.0
    planner.step_size = 1.0
    planner.max_iter = 5
    start = np.zeros(6)
    goal = np.asarray([0.5, 0.0, 0.0, 0.1, 0.2, 0.3])

    path = planner.generate(start, goal)

    assert path
    assert np.allclose(path[0], start)
    assert np.allclose(path[-1], goal)


def test_installed_ompl_binding_exposes_required_planners():
    if importlib.util.find_spec("ompl") is None:
        pytest.skip("OMPL Python binding is not installed in this environment")
    from ompl import base as ob
    from ompl import geometric as og

    assert ob.RealVectorStateSpace is not None
    for class_name in ("RRT", "RRTConnect", "RRTstar", "InformedRRTstar", "BITstar"):
        assert getattr(og, class_name, None) is not None


def test_ompl_joint_space_pipeline_returns_verified_raw_q_path(tmp_path):
    class Space:
        def __init__(self, dimension):
            self.dimension = dimension

        def setBounds(self, bounds):
            self.bounds = bounds

        def setLongestValidSegmentFraction(self, value):
            self.resolution = value

    class Bounds:
        def __init__(self, dimension):
            self.low = [None] * dimension
            self.high = [None] * dimension

        def setLow(self, index, value):
            self.low[index] = value

        def setHigh(self, index, value):
            self.high[index] = value

    class SpaceInformation:
        def __init__(self, space):
            self.space = space

        def setStateValidityChecker(self, checker):
            self.checker = checker

        def setStateValidityCheckingResolution(self, value):
            self.resolution = value

        def setup(self):
            pass

    class State(list):
        def __init__(self, space):
            super().__init__([0.0] * space.dimension)

    class Cost:
        def __init__(self, value):
            self._value = value

        def value(self):
            return self._value

    class Path:
        def __init__(self, states):
            self.states = states

        def getStateCount(self):
            return len(self.states)

        def getState(self, index):
            return self.states[index]

        def cost(self, objective):
            return Cost(np.linalg.norm(np.asarray(self.states[-1]) - np.asarray(self.states[0])))

    class ProblemDefinition:
        def __init__(self, si):
            self.si = si
            self.exact = False
            self.path = None

        def setStartAndGoalStates(self, start, goal, tolerance):
            self.start = list(start)
            self.goal = list(goal)

        def setOptimizationObjective(self, objective):
            self.objective = objective

        def hasExactSolution(self):
            return self.exact

        def hasApproximateSolution(self):
            return False

        def getSolutionPath(self):
            return self.path

    class Objective:
        def __init__(self, si):
            self.si = si

    class PlannerData:
        def __init__(self, si):
            pass

        def numVertices(self):
            return 2

        def numEdges(self):
            return 1

    class Status:
        def asString(self):
            return "Exact solution"

    class FakePlanner:
        def __init__(self, si):
            self.si = si

        def setRange(self, value):
            self.range = value

        def setProblemDefinition(self, pdef):
            self.pdef = pdef

        def setup(self):
            pass

        def solve(self, timeout):
            assert self.si.checker(self.pdef.start)
            assert self.si.checker(self.pdef.goal)
            self.pdef.path = Path([self.pdef.start, self.pdef.goal])
            self.pdef.exact = True
            return Status()

        def getPlannerData(self, data):
            pass

    ob = types.SimpleNamespace(
        RealVectorStateSpace=Space,
        RealVectorBounds=Bounds,
        SpaceInformation=SpaceInformation,
        StateValidityCheckerFn=lambda callback: callback,
        State=State,
        ProblemDefinition=ProblemDefinition,
        PathLengthOptimizationObjective=Objective,
        PlannerData=PlannerData,
    )
    og = types.SimpleNamespace(RRTConnect=FakePlanner)

    planner = RRTConnect()
    planner._import_ompl = lambda: (ob, og, "test")
    planner.debug_convergence = False
    planner.debug_exploration = False
    planner.ompl_config["debug"] = {"save_summary": False}
    planner.ompl_config["solve"] = {
        "timeout_sec": 0.1,
        "stop_on_first_solution": True,
        "convergence_slice_sec": 0.01,
    }
    planner.configure_joint_space_test_environment(
        dof=3,
        lower_limits=[-1.0, -1.0, -1.0],
        upper_limits=[1.0, 1.0, 1.0],
        collision_fn=lambda q: False,
    )
    planner.configure_fixed_joints(fixed_joint_indices=[1])
    start = np.asarray([0.0, 0.25, 0.0])
    goal = np.asarray([0.5, -0.5, -0.25])

    path = planner.generate(start, goal)

    assert len(path) == 2
    assert np.allclose(path[0], start)
    assert np.allclose(path[-1], [0.5, 0.25, -0.25])
    assert planner.last_planning_status == "success"
    assert planner.last_returned_path_reaches_goal is True
    assert planner.last_ompl_stats["vertex_count"] == 2
