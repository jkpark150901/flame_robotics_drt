"""Shared Artificial Potential Field (APF) helpers.

Used by optimizer plugins (gpmp2.py, trajopt.py) as their obstacle cost, and
by apf_heatmap.py for the on-demand APF field heatmap - one potential
function so "what does the optimizer actually optimize" and "what does the
heatmap show" never drift apart.

Distances come from PinocchioRoboticsBackend.link_obstacle_distances() (real
signed distance from pin.computeDistances), not a boolean collision check or
the dead Open3D RaycastingScene the old gpmp2.py/trajopt.py obstacle cost
relied on (planner.scene - never set by any planner in this codebase).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Distance (m) at which the repulsive potential starts acting - beyond this,
# an obstacle contributes zero cost/gradient (standard Khatib APF).
DEFAULT_D0 = 0.15
# Repulsive gain.
DEFAULT_ETA = 1.0
# Floor distance used in 1/d so a penetrating (d<=0) pose gives a large but
# finite cost instead of a division-by-zero/inf.
MIN_DISTANCE = 1e-3
# Attractive gain - how strongly each waypoint is pulled toward its
# straight-line target (see attractive_potential()).
DEFAULT_K_ATT = 1.0
# Steepness multiplier applied to distance before squaring in
# attractive_potential() - see its docstring for why this differs from k_att.
DEFAULT_ATTRACTIVE_SCALE = 2.0


def repulsive_potential(distance: float, d0: float = DEFAULT_D0, eta: float = DEFAULT_ETA) -> float:
    """Khatib repulsive potential: 0.5*eta*(1/d - 1/d0)^2 for d < d0, else 0."""
    d = max(float(distance), MIN_DISTANCE)
    if d >= d0:
        return 0.0
    return 0.5 * eta * (1.0 / d - 1.0 / d0) ** 2


def attractive_potential(q: np.ndarray, q_target: np.ndarray, k_att: float = DEFAULT_K_ATT,
                          scale: float = DEFAULT_ATTRACTIVE_SCALE) -> float:
    """Khatib quadratic attractive potential: 0.5*k_att*||scale*(q - q_target)||^2.

    scale multiplies the distance BEFORE squaring (unlike k_att, which scales
    the potential after), so it grows the cost quadratically in scale
    instead of linearly - a small bump in scale makes the potential rise
    much more steeply with distance from q_target than the same bump in
    k_att would, useful for making the attractive term competitive with
    repulsive_potential's 1/d blowup near obstacles (see DEFAULT_D0/DEFAULT_ETA
    docstrings) without needing an enormous k_att."""
    diff = scale * (np.asarray(q, dtype=float) - np.asarray(q_target, dtype=float))
    return 0.5 * k_att * float(np.dot(diff, diff))


def straight_line_targets(path_arr: np.ndarray) -> np.ndarray:
    """Per-waypoint target for attractive_potential(): the point at the same
    index on the straight line from path_arr[0] to path_arr[-1], i.e. what a
    pure-smoothness (zero-acceleration) solution converges to with no
    obstacles. Pulling each waypoint toward *this* rather than straight at
    the final goal_conf is what keeps a multi-waypoint trajectory from
    collapsing onto the goal - unlike Khatib's original single-particle APF,
    here every waypoint is optimized simultaneously, so each needs its own
    attractive target along the path, not the same one goal point."""
    n = len(path_arr)
    if n < 2:
        return np.asarray(path_arr, dtype=float).copy()
    alpha = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    return (1.0 - alpha) * path_arr[0] + alpha * path_arr[-1]


def path_attractive_cost(path_arr: np.ndarray, k_att: float = DEFAULT_K_ATT,
                          scale: float = DEFAULT_ATTRACTIVE_SCALE) -> np.ndarray:
    """Per-waypoint attractive cost toward straight_line_targets(path_arr),
    shape (N,) - 0 at the fixed start/goal waypoints by construction. See
    attractive_potential() for why scale (not just k_att) controls steepness."""
    targets = straight_line_targets(path_arr)
    diffs = scale * (np.asarray(path_arr, dtype=float) - targets)
    return 0.5 * k_att * np.sum(diffs ** 2, axis=1)


def robot_backend_and_name(planner):
    """(backend, robot_name) from a PlannerBase, or (None, None) if not wired."""
    getter = getattr(planner, "_robotics_collision_backend", None)
    if getter is None:
        return None, None
    try:
        return getter()
    except Exception:
        return None, None


def filter_excluded(entries: Sequence[Dict[str, Any]], exclude_links: Optional[Sequence[str]],
                     include_links: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """include_links (if given) keeps ONLY those link names first (e.g. just
    the end-effector, to answer "how close is the EE specifically" instead of
    "what's the closest link on the whole robot"); exclude_links then drops
    any of those names (e.g. a fixed rail-base link whose distance doesn't
    depend on the joints being studied and would otherwise dominate/mask
    everything else)."""
    result = list(entries)
    if include_links:
        included = set(include_links)
        result = [e for e in result if e["link"] in included]
    if exclude_links:
        excluded = set(exclude_links)
        result = [e for e in result if e["link"] not in excluded]
    return result


def link_distances(planner, q, exclude_links: Optional[Sequence[str]] = None,
                    include_links: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Per-(link, obstacle) distance entries at q, or [] if unavailable.

    exclude_links: link names to drop before returning (e.g. a link whose
    distance is invariant under the joints being studied - a fixed rail base
    segment - and would otherwise dominate/mask everything else since it's
    the tightest clearance in the robot regardless of what's being swept).
    include_links: if given, keep ONLY these link names (e.g. just the
    end-effector link)."""
    backend, robot_name = robot_backend_and_name(planner)
    if backend is None or not robot_name:
        return []
    try:
        entries = list(backend.link_obstacle_distances(robot_name, q))
    except Exception:
        return []
    return filter_excluded(entries, exclude_links, include_links)


def repulsive_cost(entries: Sequence[Dict[str, Any]], d0: float = DEFAULT_D0, eta: float = DEFAULT_ETA) -> float:
    """Sum of repulsive_potential() over every (link, obstacle) pair."""
    return float(sum(repulsive_potential(e["distance"], d0, eta) for e in entries))


def min_distance(entries: Sequence[Dict[str, Any]]):
    """Minimum distance across entries, or None if entries is empty."""
    if not entries:
        return None
    return float(min(e["distance"] for e in entries))


def repulsive_cost_at(backend, robot_name: str, q, d0: float = DEFAULT_D0, eta: float = DEFAULT_ETA,
                       exclude_links: Optional[Sequence[str]] = None,
                       include_links: Optional[Sequence[str]] = None) -> float:
    """Same as repulsive_cost(), for callers that already have a backend/
    robot_name (e.g. apf_heatmap.py) instead of a PlannerBase."""
    try:
        entries = list(backend.link_obstacle_distances(robot_name, q))
    except Exception:
        entries = []
    return repulsive_cost(filter_excluded(entries, exclude_links, include_links), d0, eta)


def min_distance_at(backend, robot_name: str, q, exclude_links: Optional[Sequence[str]] = None,
                     include_links: Optional[Sequence[str]] = None) -> float:
    """Same as min_distance(), for callers with a backend/robot_name instead
    of a PlannerBase (e.g. apf_heatmap.py's raw-distance sanity check)."""
    try:
        entries = list(backend.link_obstacle_distances(robot_name, q))
    except Exception:
        entries = []
    return min_distance(filter_excluded(entries, exclude_links, include_links))


def path_repulsive_cost(path_arr: np.ndarray, planner, d0: float = DEFAULT_D0, eta: float = DEFAULT_ETA,
                         exclude_links: Optional[Sequence[str]] = None) -> np.ndarray:
    """Per-waypoint repulsive cost for a full path, shape (N,)."""
    return np.array([repulsive_cost(link_distances(planner, q, exclude_links), d0, eta) for q in path_arr])


def path_cost_breakdown(path_arr: np.ndarray, planner, *, d0: float = DEFAULT_D0, eta: float = DEFAULT_ETA,
                         w_smooth: float = 1.0, k_att: float = 0.0,
                         exclude_links: Optional[Sequence[str]] = None) -> Dict[str, List[float]]:
    """Per-waypoint {smoothness, attractive (goal-pull), repulsive, total,
    min_clearance} for the final path - used to save the "what did the
    optimizer trade off" graph regardless of which optimizer produced the
    path. k_att=0 (default) drops the attractive term for optimizers that
    don't use one, rather than reporting a term they never optimized
    against."""
    n = len(path_arr)
    repulsive = np.zeros(n)
    min_clear = [None] * n
    for i, q in enumerate(path_arr):
        entries = link_distances(planner, q, exclude_links)
        repulsive[i] = repulsive_cost(entries, d0, eta)
        min_clear[i] = min_distance(entries)

    smoothness = np.zeros(n)
    if n >= 3:
        acc = path_arr[2:] - 2 * path_arr[1:-1] + path_arr[:-2]
        smoothness[1:-1] = 0.5 * w_smooth * np.sum(acc ** 2, axis=1)

    attractive = path_attractive_cost(path_arr, k_att) if (k_att and n >= 2) else np.zeros(n)

    return {
        "waypoint": list(range(n)),
        "smoothness": smoothness.tolist(),
        "attractive": attractive.tolist(),
        "repulsive": repulsive.tolist(),
        "total": (smoothness + attractive + repulsive).tolist(),
        "min_clearance": min_clear,
    }
