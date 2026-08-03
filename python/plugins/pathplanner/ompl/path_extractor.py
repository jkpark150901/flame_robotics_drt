import numpy as np


def extract_full_q_path(path_geometric, codec, duplicate_tolerance=1e-12):
    path = []
    for index in range(int(path_geometric.getStateCount())):
        q = codec.state_to_full_q(path_geometric.getState(index))
        if path and np.linalg.norm(q - path[-1]) <= float(duplicate_tolerance):
            continue
        path.append(q)
    return path


def ensure_exact_endpoints(path, start_q, goal_q, planner, tolerance=1e-9):
    if not path:
        return []
    start_q = np.asarray(start_q, dtype=float).copy()
    goal_q = np.asarray(goal_q, dtype=float).copy()
    result = [np.asarray(q, dtype=float).copy() for q in path]
    if np.linalg.norm(result[0] - start_q) <= tolerance:
        result[0] = start_q
    else:
        if planner._check_collision(start_q, result[0]):
            return []
        result.insert(0, start_q)
    if np.linalg.norm(result[-1] - goal_q) <= tolerance:
        result[-1] = goal_q
    else:
        if planner._check_collision(result[-1], goal_q):
            return []
        result.append(goal_q)
    return result
