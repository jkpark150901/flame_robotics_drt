import json
import os
import time
from typing import List, Union

import numpy as np

from plugins.pluginbase.plannerbase import PlannerBase


class RRTStar(PlannerBase):
    """RRT* 기반 경로 계획기.

    역할:
        - PlannerBase.generate()가 use_joint_space_planning 속성값을 보고 분기한다.
        - joint-space RRT* 알고리즘 자체는 이 파일에 둔다.
        - PlannerBase는 정규화 거리, steer, 충돌 정보, 탐색 로그 저장 같은 공통 기능만 제공한다.
    """

    use_joint_space_planning = True

    def __init__(self, config_path: str = None):
        """설정 파일을 읽어 RRT* 파라미터를 초기화한다.

        Args:
            config_path: rrt_star.json 경로. None이면 이 파일과 같은 이름의 json을 사용한다.

        Returns:
            None

        계산 과정:
            1. json 설정 파일을 읽는다.
            2. step_size, max_iter, search_radius, goal_bias를 로드한다.
            3. use_joint_space_planning, normalize_joint_space, debug_exploration 값을 설정한다.
            4. PlannerBase.configure_collision()으로 Pinocchio collision backend를 설정한다.
        """
        super().__init__()
        if config_path is None:
            config_path = os.path.splitext(__file__)[0] + ".json"

        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.step_size = self.config.get("step_size", 1.0)
        self.max_iter = self.config.get("max_iter", 1000)
        self.search_radius = self.config.get("search_radius", 5.0)
        self.goal_bias = self.config.get("goal_bias", 0.1)
        self.goal_check_interval = int(self.config.get("goal_check_interval", 10))
        self.early_stop_on_goal = bool(self.config.get(
            "early_stop_on_goal",
            self.config.get("terminate_on_first_solution", False),
        ))
        self.terminate_on_first_solution = self.early_stop_on_goal
        self.solution_patience = int(self.config.get("solution_patience", 0))
        self.normalize_joint_space = bool(self.config.get("normalize_joint_space", True))
        self.use_joint_space_planning = bool(self.config.get("use_joint_space_planning", self.use_joint_space_planning))
        self.debug_exploration = bool(self.config.get("debug_exploration", False))
        self.debug_solution_paths = bool(self.config.get("debug_solution_paths", self.debug_exploration))
        self.debug_output_dir = self.config.get(
            "debug_output_dir",
            os.path.join(os.getcwd(), "debug", "rrt_star"),
        )
        self.last_exploration_csv = None
        self.last_exploration_plot = None
        self.last_solution_paths_json = None
        self.last_planning_status = None
        self.last_returned_path_reaches_goal = False
        self.bounds = self.config.get("workspace_bounds", {
            "x_min": -10.0,
            "x_max": 10.0,
            "y_min": -10.0,
            "y_max": 10.0,
            "z_min": -10.0,
            "z_max": 10.0,
        })

        self.configure_collision(self.config, default_sample_resolution=self.step_size)

    def _generate_workspace(self, current_pose, target_pose, step_callback=None):
        """3D workspace에서 RRT* 경로를 생성한다.

        Args:
            current_pose: 시작 pose. 앞 3개 원소는 xyz, 뒤 원소는 orientation이다.
            target_pose: 목표 pose. orientation에 NaN이 있으면 시작 orientation을 유지한다.
            step_callback: 현재 workspace branch에서는 사용하지 않는 선택 콜백.

        Returns:
            workspace waypoint list. 실패하면 빈 list.

        계산 과정:
            1. workspace bounds 안에서 임의 점을 샘플링하거나 goal을 직접 샘플링한다.
            2. tree에서 샘플과 가장 가까운 노드를 찾는다.
            3. step_size만큼 전진한 새 노드를 만든다.
            4. 충돌이 없으면 주변 노드 중 최저 cost parent를 선택한다.
            5. 새 노드를 추가하고, 새 노드를 거치는 것이 더 저렴한 이웃 노드를 rewire한다.
            6. goal 근처 노드 중 goal까지 충돌 없이 연결 가능한 최저 cost 노드를 찾아 path를 복원한다.
        """
        start_pos = current_pose
        goal_pos = target_pose

        nodes = [start_pos]
        parents = {0: None}
        costs = {0: 0.0}

        for _ in range(self.max_iter):
            if np.random.random() < self.goal_bias:
                rnd_point = goal_pos
            else:
                rnd_point = np.array([
                    np.random.uniform(self.bounds["x_min"], self.bounds["x_max"]),
                    np.random.uniform(self.bounds["y_min"], self.bounds["y_max"]),
                    np.random.uniform(self.bounds["z_min"], self.bounds["z_max"]),
                ])

            dists = np.linalg.norm(np.asarray(nodes) - rnd_point, axis=1)
            nearest_idx = int(np.argmin(dists))
            nearest_node = nodes[nearest_idx]

            direction = rnd_point - nearest_node
            length = float(np.linalg.norm(direction))
            if length == 0:
                continue

            new_point = nearest_node + direction / length * min(float(self.step_size), length)
            if self._check_collision(nearest_node, new_point):
                continue

            new_idx = len(nodes)
            dists_all = np.linalg.norm(np.asarray(nodes) - new_point, axis=1)
            neighbor_indices = np.where(dists_all < self.search_radius)[0]

            min_cost = costs[nearest_idx] + np.linalg.norm(new_point - nearest_node)
            best_parent_idx = nearest_idx
            for nb_idx in neighbor_indices:
                if nb_idx == nearest_idx:
                    continue
                if not self._check_collision(nodes[nb_idx], new_point):
                    cost = costs[nb_idx] + np.linalg.norm(new_point - nodes[nb_idx])
                    if cost < min_cost:
                        min_cost = cost
                        best_parent_idx = int(nb_idx)

            nodes.append(new_point)
            parents[new_idx] = best_parent_idx
            costs[new_idx] = min_cost

            for nb_idx in neighbor_indices:
                if nb_idx == best_parent_idx:
                    continue
                new_cost_to_nb = min_cost + np.linalg.norm(nodes[nb_idx] - new_point)
                if new_cost_to_nb < costs[nb_idx] and not self._check_collision(new_point, nodes[nb_idx]):
                    parents[int(nb_idx)] = new_idx
                    costs[int(nb_idx)] = new_cost_to_nb

        dists_to_goal = np.linalg.norm(np.asarray(nodes) - goal_pos, axis=1)
        close_indices = np.where(dists_to_goal < self.step_size)[0]

        goal_idx = -1
        min_total_cost = float("inf")
        for idx in close_indices:
            if not self._check_collision(nodes[idx], goal_pos):
                cost = costs[idx] + np.linalg.norm(goal_pos - nodes[idx])
                if cost < min_total_cost:
                    min_total_cost = cost
                    goal_idx = int(idx)

        if goal_idx == -1:
            print("RRT* failed to find path")
            return []

        path = []
        pose = np.copy(target_pose)
        final_orient = target_pose[3:]
        current_orient = current_pose[3:]
        pose[3:] = np.where(np.isnan(final_orient), current_orient, final_orient)
        path.append(pose)

        curr_idx = goal_idx
        while curr_idx is not None:
            pose = np.copy(current_pose)
            pose[:3] = nodes[curr_idx]
            pose[3:] = current_pose[3:]
            path.append(pose)
            curr_idx = parents[curr_idx]
        return path[::-1]

    def _generate_joint_space(self, start_q, goal_q, step_callback=None):
        """Pinocchio q-space에서 RRT* 경로를 생성한다.

        Args:
            start_q: 시작 joint vector. raw joint 단위이며 collision check에도 그대로 사용한다.
            goal_q: 목표 joint vector. raw joint 단위이며 collision check에도 그대로 사용한다.
            step_callback: 탐색 중 nodes/parents를 외부로 전달하는 선택 콜백.

        Returns:
            q-space waypoint list. 성공 시 start_q부터 goal_q까지의 raw q 배열 list, 실패 시 빈 list.

        계산 과정:
            1. 시작/목표 q 자체가 충돌인지 먼저 확인한다.
            2. goal bias에 따라 목표 q 또는 random q를 샘플링한다.
            3. PlannerBase의 normalized joint metric으로 nearest node를 찾는다.
            4. PlannerBase의 normalized steer로 새 후보 q를 만든다.
            5. nearest -> new q edge의 충돌 여부와 최초 충돌 q를 확인한다.
            6. 주변 노드 중 충돌 없이 연결 가능한 최저 cost parent를 고른다.
            7. 새 노드를 추가하고 주변 노드를 rewire한다.
            8. goal에 가까운 노드 중 goal까지 충돌 없이 연결 가능한 최저 cost 노드를 찾는다.
            9. parent chain을 따라 start -> goal q path를 복원한다.
        """
        exploration_rows = self._new_exploration_rows()
        self.last_planning_status = "running"
        self.last_returned_path_reaches_goal = False
        start_goal_dist = self._joint_distance(start_q, goal_q)
        start_goal_raw_dist = float(np.linalg.norm(goal_q - start_q))

        if self.check_pinocchio_collision(start_q):
            self._record_exploration(
                exploration_rows,
                iteration=-1,
                phase="start_collision",
                from_q=start_q,
                collision=True,
                reason="start configuration is in collision",
                node_count=1,
            )
            self._save_exploration_debug(exploration_rows, "joint", "start_collision")
            print(
                "RRTStar joint-space summary: failed=start_collision, "
                f"start_goal_dist={start_goal_dist:.6f}, raw_dist={start_goal_raw_dist:.6f}, "
                f"normalized={self.normalize_joint_space}, step_size={float(self.step_size):.6f}, "
                f"early_stop_on_goal={self.early_stop_on_goal}"
            )
            return []

        if self.check_pinocchio_collision(goal_q):
            self._record_exploration(
                exploration_rows,
                iteration=-1,
                phase="goal_collision",
                to_q=goal_q,
                collision=True,
                reason="goal configuration is in collision",
                node_count=1,
            )
            self._save_exploration_debug(exploration_rows, "joint", "goal_collision")
            print(
                "RRTStar joint-space summary: failed=goal_collision, "
                f"start_goal_dist={start_goal_dist:.6f}, raw_dist={start_goal_raw_dist:.6f}, "
                f"normalized={self.normalize_joint_space}, step_size={float(self.step_size):.6f}, "
                f"early_stop_on_goal={self.early_stop_on_goal}"
            )
            return []

        nodes = [np.asarray(start_q, dtype=float).copy()]
        parents = {0: None}
        costs = {0: 0.0}
        stats = {
            "goal_bias_samples": 0,
            "random_samples": 0,
            "edge_collision_rejects": 0,
            "random_edge_collision_rejects": 0,
            "parent_collision_rejects": 0,
            "rewire_collision_rejects": 0,
            "rewires": 0,
            "nodes_added": 0,
            "goal_connection_collision_rejects": 0,
        }
        timings = {
            "start_goal_collision": 0.0,
            "deadline": 0.0,
            "sample_nearest_steer": 0.0,
            "extend_collision": 0.0,
            "choose_parent": 0.0,
            "add_node_log": 0.0,
            "rewire": 0.0,
            "callback": 0.0,
            "connect_goal": 0.0,
            "reconstruct": 0.0,
            "save_debug": 0.0,
        }
        total_t0 = time.perf_counter()
        last_iteration = -1
        last_added_idx = 0
        best_path = None
        best_cost = float("inf")
        best_iteration = -1
        last_solution_improvement_iteration = -1
        solution_paths = []

        def _add_timing(name, t0):
            timings[name] = timings.get(name, 0.0) + (time.perf_counter() - t0)

        def _timing_text():
            total = time.perf_counter() - total_t0
            parts = [f"{key}={value:.3f}s" for key, value in timings.items() if value > 0.0]
            parts.append(f"total={total:.3f}s")
            return ", ".join(parts)

        def _try_update_solution(iteration, reason):
            nonlocal best_path, best_cost, best_iteration, last_solution_improvement_iteration
            stage_t0 = time.perf_counter()
            goal_idx, min_total_cost, close_indices = self._connect_joint_goal(
                nodes,
                costs,
                goal_q,
                exploration_rows,
                stats,
                total_t0=total_t0,
            )
            _add_timing("connect_goal", stage_t0)
            if goal_idx == -1:
                return None, goal_idx, min_total_cost, close_indices

            stage_t0 = time.perf_counter()
            path = self._reconstruct_joint_path(nodes, parents, goal_idx, goal_q)
            _add_timing("reconstruct", stage_t0)
            improved = min_total_cost < best_cost
            snapshot = {
                "iteration": int(iteration),
                "reason": reason,
                "cost": float(min_total_cost),
                "elapsed": time.perf_counter() - total_t0,
                "waypoints": len(path),
                "path": [np.asarray(q, dtype=float).copy() for q in path],
            }
            solution_paths.append(snapshot)
            self._record_exploration(
                exploration_rows,
                iteration=iteration,
                phase="connect_goal",
                nearest_idx=goal_idx,
                from_q=nodes[goal_idx],
                to_q=goal_q,
                collision=False,
                accepted=True,
                reason=reason,
                node_count=len(nodes),
                cost=min_total_cost,
                elapsed_s=time.perf_counter() - total_t0,
                new_node_collision_count=stats.get("edge_collision_rejects", 0),
                random_new_node_collision_count=stats.get("random_edge_collision_rejects", 0),
                rewire_collision_count=stats.get("rewire_collision_rejects", 0),
            )
            if improved:
                best_path = path
                best_cost = float(min_total_cost)
                best_iteration = int(iteration)
                last_solution_improvement_iteration = int(iteration)
            return path, goal_idx, min_total_cost, close_indices

        def _latest_iteration_path():
            if len(nodes) <= 1:
                return [np.asarray(start_q, dtype=float).copy()]
            return self._reconstruct_branch_path(nodes, parents, last_added_idx)

        try:
            for i in range(int(self.max_iter)):
                last_iteration = i
                stage_t0 = time.perf_counter()
                self._check_planning_deadline()
                _add_timing("deadline", stage_t0)

                stage_t0 = time.perf_counter()
                rnd_point, sample_type = self._sample_joint_target(goal_q, stats)
                nearest_idx     = self._nearest_joint_node(nodes, rnd_point)
                nearest_node    = nodes[nearest_idx]
                new_point       = self._steer_joint_state(nearest_node, rnd_point, self.step_size)
                _add_timing("sample_nearest_steer", stage_t0)

                stage_t0 = time.perf_counter()
                hit, pairs, collision_q, collision_alpha = self._edge_collision_info(nearest_node, new_point)
                collision_elapsed = time.perf_counter() - stage_t0
                timings["extend_collision"] = timings.get("extend_collision", 0.0) + collision_elapsed
                if hit:
                    stats["edge_collision_rejects"] += 1
                    if sample_type == "random":
                        stats["random_edge_collision_rejects"] += 1
                    self._record_exploration(
                        exploration_rows,
                        iteration=i,
                        phase="extend",
                        sample_type=sample_type,
                        nearest_idx=nearest_idx,
                        from_q=nearest_node,
                        to_q=new_point,
                        sample_q=rnd_point,
                        collision=True,
                        collision_pairs=pairs,
                        collision_q=collision_q,
                        collision_alpha=collision_alpha,
                        accepted=False,
                        reason="extend_edge_collision",
                        node_count=len(nodes),
                        elapsed_s=time.perf_counter() - total_t0,
                        phase_elapsed_s=collision_elapsed,
                        collision_check_elapsed_s=collision_elapsed,
                        new_node_collision_count=stats.get("edge_collision_rejects", 0),
                        random_new_node_collision_count=stats.get("random_edge_collision_rejects", 0),
                        rewire_collision_count=stats.get("rewire_collision_rejects", 0),
                    )
                    continue

                new_idx = len(nodes)
                neighbor_indices = self._near_joint_nodes(nodes, new_point)
                stage_t0 = time.perf_counter()
                min_cost, best_parent_idx = self._choose_joint_parent(
                    nodes,
                    costs,
                    nearest_idx,
                    neighbor_indices,
                    new_point,
                    rnd_point,
                    sample_type,
                    exploration_rows,
                    stats,
                    i,
                    total_t0=total_t0,
                )
                _add_timing("choose_parent", stage_t0)

                stage_t0 = time.perf_counter()
                nodes.append(new_point)
                last_added_idx = new_idx
                stats["nodes_added"] += 1
                parents[new_idx] = best_parent_idx
                costs[new_idx] = min_cost
                self._record_exploration(
                    exploration_rows,
                    iteration=i,
                    phase="add_node",
                    sample_type=sample_type,
                    nearest_idx=nearest_idx,
                    from_q=nodes[best_parent_idx],
                    to_q=new_point,
                    sample_q=rnd_point,
                    collision=False,
                    accepted=True,
                    reason=f"best_parent={best_parent_idx}",
                    node_count=len(nodes),
                    cost=min_cost,
                    elapsed_s=time.perf_counter() - total_t0,
                    new_node_collision_count=stats.get("edge_collision_rejects", 0),
                    random_new_node_collision_count=stats.get("random_edge_collision_rejects", 0),
                    rewire_collision_count=stats.get("rewire_collision_rejects", 0),
                )
                _add_timing("add_node_log", stage_t0)

                stage_t0 = time.perf_counter()
                self._rewire_joint_neighbors(
                    nodes,
                    parents,
                    costs,
                    new_idx,
                    new_point,
                    neighbor_indices,
                    best_parent_idx,
                    min_cost,
                    rnd_point,
                    sample_type,
                    exploration_rows,
                    stats,
                    i,
                    total_t0=total_t0,
                )
                _add_timing("rewire", stage_t0)

                should_check_goal = (
                    self.goal_check_interval > 0
                    and (i % self.goal_check_interval == 0
                         or sample_type == "goal_bias"
                         or self._joint_distance(new_point, goal_q) <= float(self.step_size))
                )
                if should_check_goal:
                    path, goal_idx, min_total_cost, close_indices = _try_update_solution(
                        i,
                        "periodic_goal_connected",
                    )
                    if path is not None:
                        if self.early_stop_on_goal:
                            stage_t0 = time.perf_counter()
                            self._save_exploration_debug(exploration_rows, "joint", "first_solution", len(path))
                            self._save_solution_paths_debug(solution_paths, "first_solution")
                            _add_timing("save_debug", stage_t0)
                            print(
                                "RRTStar joint-space summary: first_solution, "
                                f"iteration={i}, nodes={len(nodes)}, close_indices={len(close_indices)}, "
                                f"goal_idx={goal_idx}, path_waypoints={len(path)}, "
                                f"min_total_cost={float(min_total_cost):.6f}, stats={stats}"
                            )
                            print(
                                "RRTStar joint-space timing: "
                                f"status=first_solution, iteration={last_iteration}, timings=({_timing_text()})"
                            )
                            self.last_planning_status = "first_solution"
                            self.last_returned_path_reaches_goal = True
                            return path
                        if (
                            self.early_stop_on_goal
                            and
                            self.solution_patience > 0
                            and last_solution_improvement_iteration >= 0
                            and i - last_solution_improvement_iteration >= self.solution_patience
                        ):
                            stage_t0 = time.perf_counter()
                            self._save_exploration_debug(exploration_rows, "joint", "solution_patience", len(best_path))
                            self._save_solution_paths_debug(solution_paths, "solution_patience")
                            _add_timing("save_debug", stage_t0)
                            print(
                                "RRTStar joint-space summary: solution_patience, "
                                f"iteration={i}, best_iteration={best_iteration}, nodes={len(nodes)}, "
                                f"path_waypoints={len(best_path)}, best_cost={float(best_cost):.6f}, stats={stats}"
                            )
                            print(
                                "RRTStar joint-space timing: "
                                f"status=solution_patience, iteration={last_iteration}, timings=({_timing_text()})"
                            )
                            self.last_planning_status = "solution_patience"
                            self.last_returned_path_reaches_goal = True
                            return best_path

                if step_callback is not None:
                    stage_t0 = time.perf_counter()
                    step_callback(np.asarray(nodes), parents)
                    _add_timing("callback", stage_t0)
        except Exception as exc:
            is_timeout = "timeout" in str(exc).lower()
            if not is_timeout:
                stage_t0 = time.perf_counter()
                self._save_exploration_debug(exploration_rows, "joint", "interrupted")
                _add_timing("save_debug", stage_t0)
                print(
                    "RRTStar joint-space timing: "
                    f"failed=interrupted, iteration={last_iteration}, nodes={len(nodes)}, "
                    f"stats={stats}, timings=({_timing_text()})"
                )
                raise

            path, goal_idx, min_total_cost, close_indices = _try_update_solution(
                last_iteration,
                "goal_connected_after_timeout",
            )
            if path is not None:
                stage_t0 = time.perf_counter()
                self._save_exploration_debug(exploration_rows, "joint", "timeout_success", len(path))
                self._save_solution_paths_debug(solution_paths, "timeout_success")
                _add_timing("save_debug", stage_t0)
                print(
                    "RRTStar joint-space summary: timeout_but_goal_connected, "
                    f"start_goal_dist={start_goal_dist:.6f}, raw_dist={start_goal_raw_dist:.6f}, "
                    f"normalized={self.normalize_joint_space}, step_size={float(self.step_size):.6f}, "
                    f"search_radius={float(self.search_radius):.6f}, nodes={len(nodes)}, "
                    f"close_indices={len(close_indices)}, goal_idx={goal_idx}, "
                    f"path_waypoints={len(path)}, min_total_cost={float(min_total_cost):.6f}, stats={stats}"
                )
                print(
                    "RRTStar joint-space timing: "
                    f"status=timeout_success, iteration={last_iteration}, timings=({_timing_text()})"
                )
                self.last_planning_status = "timeout_success"
                self.last_returned_path_reaches_goal = True
                return path

            if best_path is not None:
                stage_t0 = time.perf_counter()
                self._save_exploration_debug(exploration_rows, "joint", "timeout_best_solution", len(best_path))
                self._save_solution_paths_debug(solution_paths, "timeout_best_solution")
                _add_timing("save_debug", stage_t0)
                print(
                    "RRTStar joint-space summary: timeout_return_best_solution, "
                    f"iteration={last_iteration}, best_iteration={best_iteration}, nodes={len(nodes)}, "
                    f"path_waypoints={len(best_path)}, best_cost={float(best_cost):.6f}, stats={stats}"
                )
                print(
                    "RRTStar joint-space timing: "
                    f"status=timeout_best_solution, iteration={last_iteration}, timings=({_timing_text()})"
                )
                self.last_planning_status = "timeout_best_solution"
                self.last_returned_path_reaches_goal = True
                return best_path

            latest_path = _latest_iteration_path()
            stage_t0 = time.perf_counter()
            self._save_exploration_debug(exploration_rows, "joint", "timeout_latest_branch", len(latest_path))
            self._save_solution_paths_debug(solution_paths, "timeout_latest_branch")
            _add_timing("save_debug", stage_t0)
            print(
                "RRTStar joint-space timing: "
                f"status=timeout_latest_branch, iteration={last_iteration}, nodes={len(nodes)}, "
                f"close_indices={len(close_indices)}, latest_waypoints={len(latest_path)}, "
                f"stats={stats}, timings=({_timing_text()})"
            )
            self.last_planning_status = "timeout_latest_branch"
            self.last_returned_path_reaches_goal = False
            return latest_path

        path, goal_idx, min_total_cost, close_indices = _try_update_solution(
            int(self.max_iter),
            "goal_connected",
        )
        if goal_idx == -1:
            if best_path is not None:
                stage_t0 = time.perf_counter()
                self._save_exploration_debug(exploration_rows, "joint", "max_iter_best_solution", len(best_path))
                self._save_solution_paths_debug(solution_paths, "max_iter_best_solution")
                _add_timing("save_debug", stage_t0)
                print(
                    "RRTStar joint-space summary: max_iter_return_best_solution, "
                    f"start_goal_dist={start_goal_dist:.6f}, raw_dist={start_goal_raw_dist:.6f}, "
                    f"normalized={self.normalize_joint_space}, step_size={float(self.step_size):.6f}, "
                    f"search_radius={float(self.search_radius):.6f}, nodes={len(nodes)}, "
                    f"best_iteration={best_iteration}, path_waypoints={len(best_path)}, "
                    f"best_cost={float(best_cost):.6f}, stats={stats}"
                )
                print(
                    "RRTStar joint-space timing: "
                    f"status=max_iter_best_solution, iteration={last_iteration}, timings=({_timing_text()})"
                )
                self.last_planning_status = "max_iter_best_solution"
                self.last_returned_path_reaches_goal = True
                return best_path

            latest_path = _latest_iteration_path()
            stage_t0 = time.perf_counter()
            self._save_exploration_debug(exploration_rows, "joint", "max_iter_latest_branch", len(latest_path))
            self._save_solution_paths_debug(solution_paths, "max_iter_latest_branch")
            _add_timing("save_debug", stage_t0)
            print(
                "RRTStar joint-space summary: max_iter_return_latest_branch, "
                f"start_goal_dist={start_goal_dist:.6f}, raw_dist={start_goal_raw_dist:.6f}, "
                f"normalized={self.normalize_joint_space}, step_size={float(self.step_size):.6f}, "
                f"search_radius={float(self.search_radius):.6f}, nodes={len(nodes)}, "
                f"close_indices={len(close_indices)}, latest_waypoints={len(latest_path)}, stats={stats}"
            )
            print(
                "RRTStar joint-space timing: "
                f"status=max_iter_latest_branch, iteration={last_iteration}, timings=({_timing_text()})"
            )
            self.last_planning_status = "max_iter_latest_branch"
            self.last_returned_path_reaches_goal = False
            return latest_path

        stage_t0 = time.perf_counter()
        self._save_exploration_debug(exploration_rows, "joint", "success", len(path))
        self._save_solution_paths_debug(solution_paths, "success")
        _add_timing("save_debug", stage_t0)
        print(
            "RRTStar joint-space summary: success, "
            f"start_goal_dist={start_goal_dist:.6f}, raw_dist={start_goal_raw_dist:.6f}, "
            f"normalized={self.normalize_joint_space}, step_size={float(self.step_size):.6f}, "
            f"search_radius={float(self.search_radius):.6f}, nodes={len(nodes)}, "
            f"close_indices={len(close_indices)}, goal_idx={goal_idx}, "
            f"path_waypoints={len(path)}, min_total_cost={float(min_total_cost):.6f}, stats={stats}"
        )
        print(
            "RRTStar joint-space timing: "
            f"status=success, iteration={last_iteration}, timings=({_timing_text()})"
        )
        self.last_planning_status = "success"
        self.last_returned_path_reaches_goal = True
        return path

    def _reconstruct_branch_path(self, nodes, parents, node_idx):
        """goal 연결이 없을 때 특정 tree node까지의 branch path를 복원한다."""
        path = []
        curr_idx = int(node_idx)
        while curr_idx is not None:
            path.append(nodes[curr_idx].copy())
            curr_idx = parents[curr_idx]
        return path[::-1]

    def _save_solution_paths_debug(self, solution_paths, status):
        """iteration 중 발견된 solution path snapshot들을 JSON으로 저장한다."""
        if not getattr(self, "debug_solution_paths", False) or not solution_paths:
            return None
        out_dir = os.path.abspath(getattr(self, "debug_output_dir", os.path.join(os.getcwd(), "debug", "rrt_star")))
        os.makedirs(out_dir, exist_ok=True)
        robot_name = "robot"
        try:
            robot_name = str(getattr(self.pin_model, "name", "") or "robot")
        except Exception:
            pass
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(
            out_dir,
            f"{self.__class__.__name__.lower()}_joint_{robot_name}_{stamp}_{status}_solutions.json",
        )
        serializable = []
        for item in solution_paths:
            serializable.append({
                "iteration": int(item.get("iteration", -1)),
                "reason": item.get("reason", ""),
                "cost": float(item.get("cost", 0.0)),
                "elapsed": float(item.get("elapsed", 0.0)),
                "waypoints": int(item.get("waypoints", 0)),
                "path": [np.asarray(q, dtype=float).round(8).tolist() for q in item.get("path", [])],
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
            f.write("\n")
        self.last_solution_paths_json = path
        print(f"RRTStar solution path snapshots saved: json={path}, count={len(serializable)}")
        return path

    def _sample_joint_target(self, goal_q, stats):
        """RRT* 확장을 위한 q-space 샘플을 생성한다.

        Args:
            goal_q: 목표 joint vector.
            stats: 샘플 횟수를 누적하는 dict.

        Returns:
            (sample_q, sample_type). sample_type은 "goal_bias" 또는 "random".

        계산 과정:
            goal_bias 확률이면 목표 q를 그대로 반환하고, 아니면 joint limit 내부 random q를 반환한다.
        """
        if np.random.random() < self.goal_bias:
            stats["goal_bias_samples"] += 1
            return np.asarray(goal_q, dtype=float), "goal_bias"
        stats["random_samples"] += 1
        return self._sample_pinocchio_configuration(), "random"

    def _nearest_joint_node(self, nodes, sample_q):
        """샘플 q와 가장 가까운 tree 노드 index를 찾는다.

        Args:
            nodes: 현재 tree의 raw q node list.
            sample_q: 비교 대상 raw q.

        Returns:
            가장 가까운 node의 index.

        계산 과정:
            PlannerBase._joint_distances를 사용하므로 prismatic/revolute scale을 정규화해서 비교한다.
        """
        dists = self._joint_distances(nodes, sample_q)
        return int(np.argmin(dists))

    def _near_joint_nodes(self, nodes, new_q):
        """새 후보 q 주변의 neighbor node index들을 찾는다.

        Args:
            nodes: 현재 tree의 raw q node list.
            new_q: 새 후보 raw q.

        Returns:
            search_radius 내부에 있는 node index 배열.

        계산 과정:
            normalized joint distance를 계산하고 search_radius보다 작은 index만 선택한다.
        """
        dists = self._joint_distances(nodes, new_q)
        return np.where(dists < self.search_radius)[0]

    def _choose_joint_parent(
        self,
        nodes,
        costs,
        nearest_idx,
        neighbor_indices,
        new_q,
        sample_q,
        sample_type,
        exploration_rows,
        stats,
        iteration,
        total_t0=None,
    ):
        """새 후보 q에 연결할 최저 cost parent를 선택한다.

        Args:
            nodes: 현재 tree의 raw q node list.
            costs: start에서 각 node까지의 누적 cost dict.
            nearest_idx: nearest node index.
            neighbor_indices: parent 후보 neighbor index들.
            new_q: 새 후보 raw q.
            sample_q: 이번 반복에서 샘플링한 raw q.
            sample_type: "goal_bias" 또는 "random".
            exploration_rows: 탐색 CSV에 기록할 row list.
            stats: 충돌/선택 통계 dict.
            iteration: 현재 반복 index.

        Returns:
            (min_cost, best_parent_idx).

        계산 과정:
            neighbor -> new_q edge가 충돌하지 않는 후보 중 누적 cost가 가장 작은 parent를 고른다.
            충돌한 parent 후보는 exploration log에 choose_parent 단계로 기록한다.
        """
        min_cost = costs[nearest_idx] + self._joint_distance(nodes[nearest_idx], new_q)
        best_parent_idx = nearest_idx

        for nb_idx in neighbor_indices:
            if nb_idx == nearest_idx:
                continue
            collision_t0 = time.perf_counter()
            parent_hit, parent_pairs, collision_q, collision_alpha = self._edge_collision_info(nodes[nb_idx], new_q)
            collision_elapsed = time.perf_counter() - collision_t0
            if not parent_hit:
                cost = costs[nb_idx] + self._joint_distance(nodes[nb_idx], new_q)
                if cost < min_cost:
                    min_cost = cost
                    best_parent_idx = int(nb_idx)
                continue

            stats["parent_collision_rejects"] += 1
            self._record_exploration(
                exploration_rows,
                iteration=iteration,
                phase="choose_parent",
                sample_type=sample_type,
                nearest_idx=int(nb_idx),
                from_q=nodes[nb_idx],
                to_q=new_q,
                sample_q=sample_q,
                collision=True,
                collision_pairs=parent_pairs,
                collision_q=collision_q,
                collision_alpha=collision_alpha,
                accepted=False,
                reason="parent_edge_collision",
                node_count=len(nodes),
                elapsed_s=None if total_t0 is None else time.perf_counter() - total_t0,
                phase_elapsed_s=collision_elapsed,
                collision_check_elapsed_s=collision_elapsed,
                new_node_collision_count=stats.get("edge_collision_rejects", 0),
                random_new_node_collision_count=stats.get("random_edge_collision_rejects", 0),
                rewire_collision_count=stats.get("rewire_collision_rejects", 0),
            )

        return min_cost, best_parent_idx

    def _rewire_joint_neighbors(
        self,
        nodes,
        parents,
        costs,
        new_idx,
        new_q,
        neighbor_indices,
        best_parent_idx,
        min_cost,
        sample_q,
        sample_type,
        exploration_rows,
        stats,
        iteration,
        total_t0=None,
    ):
        """새 노드를 통해 cost가 줄어드는 neighbor들을 rewire한다.

        Args:
            nodes: 현재 tree의 raw q node list.
            parents: node index -> parent index dict.
            costs: start에서 각 node까지의 누적 cost dict.
            new_idx: 새로 추가된 node index.
            new_q: 새로 추가된 raw q.
            neighbor_indices: rewire 후보 index들.
            best_parent_idx: new_q의 parent index.
            min_cost: start에서 new_q까지의 cost.
            sample_q: 이번 반복에서 샘플링한 raw q.
            sample_type: "goal_bias" 또는 "random".
            exploration_rows: 탐색 CSV에 기록할 row list.
            stats: 충돌/rewire 통계 dict.
            iteration: 현재 반복 index.

        Returns:
            None. parents/costs/stats를 in-place로 갱신한다.

        계산 과정:
            new_q -> neighbor edge가 충돌하지 않고, new_q를 거치는 cost가 더 작으면 parent를 new_idx로 교체한다.
            충돌한 rewire 후보는 exploration log에 rewire 단계로 기록한다.
        """
        for nb_idx in neighbor_indices:
            if nb_idx == best_parent_idx:
                continue

            new_cost_to_nb = min_cost + self._joint_distance(new_q, nodes[nb_idx])
            if new_cost_to_nb >= costs[nb_idx]:
                continue

            collision_t0 = time.perf_counter()
            rewire_hit, rewire_pairs, collision_q, collision_alpha = self._edge_collision_info(new_q, nodes[nb_idx])
            collision_elapsed = time.perf_counter() - collision_t0
            if not rewire_hit:
                parents[int(nb_idx)] = new_idx
                costs[int(nb_idx)] = new_cost_to_nb
                stats["rewires"] += 1
                self._record_exploration(
                    exploration_rows,
                    iteration=iteration,
                    phase="rewire",
                    sample_type=sample_type,
                    nearest_idx=int(nb_idx),
                    from_q=new_q,
                    to_q=nodes[nb_idx],
                    sample_q=sample_q,
                    collision=False,
                    accepted=True,
                    reason=f"rewired_to={new_idx}",
                    node_count=len(nodes),
                    cost=new_cost_to_nb,
                    elapsed_s=None if total_t0 is None else time.perf_counter() - total_t0,
                    phase_elapsed_s=collision_elapsed,
                    collision_check_elapsed_s=collision_elapsed,
                    new_node_collision_count=stats.get("edge_collision_rejects", 0),
                    random_new_node_collision_count=stats.get("random_edge_collision_rejects", 0),
                    rewire_collision_count=stats.get("rewire_collision_rejects", 0),
                )
                continue

            stats["rewire_collision_rejects"] += 1
            self._record_exploration(
                exploration_rows,
                iteration=iteration,
                phase="rewire",
                sample_type=sample_type,
                nearest_idx=int(nb_idx),
                from_q=new_q,
                to_q=nodes[nb_idx],
                sample_q=sample_q,
                collision=True,
                collision_pairs=rewire_pairs,
                collision_q=collision_q,
                collision_alpha=collision_alpha,
                accepted=False,
                reason="rewire_edge_collision",
                node_count=len(nodes),
                elapsed_s=None if total_t0 is None else time.perf_counter() - total_t0,
                phase_elapsed_s=collision_elapsed,
                collision_check_elapsed_s=collision_elapsed,
                new_node_collision_count=stats.get("edge_collision_rejects", 0),
                random_new_node_collision_count=stats.get("random_edge_collision_rejects", 0),
                rewire_collision_count=stats.get("rewire_collision_rejects", 0),
            )

    def _connect_joint_goal(self, nodes, costs, goal_q, exploration_rows, stats, total_t0=None):
        """tree에서 goal_q에 연결 가능한 최적 노드를 찾는다.

        Args:
            nodes: 현재 tree의 raw q node list.
            costs: start에서 각 node까지의 누적 cost dict.
            goal_q: 목표 raw q.
            exploration_rows: 탐색 CSV에 기록할 row list.
            stats: goal 연결 충돌 통계 dict.

        Returns:
            (goal_idx, min_total_cost, close_indices). 연결 실패 시 goal_idx는 -1이다.

        계산 과정:
            goal_q와 normalized distance가 step_size보다 작은 node만 검사한다.
            node -> goal_q edge가 collision-free인 후보 중 total cost가 가장 작은 node를 선택한다.
        """
        dists_to_goal = self._joint_distances(nodes, goal_q)
        close_indices = np.where(dists_to_goal < self.step_size)[0]

        goal_idx = -1
        min_total_cost = float("inf")
        for idx in close_indices:
            collision_t0 = time.perf_counter()
            goal_hit, goal_pairs, collision_q, collision_alpha = self._edge_collision_info(nodes[idx], goal_q)
            collision_elapsed = time.perf_counter() - collision_t0
            if not goal_hit:
                cost = costs[idx] + self._joint_distance(nodes[idx], goal_q)
                if cost < min_total_cost:
                    min_total_cost = cost
                    goal_idx = int(idx)
                continue

            stats["goal_connection_collision_rejects"] += 1
            self._record_exploration(
                exploration_rows,
                iteration=int(self.max_iter),
                phase="connect_goal",
                nearest_idx=int(idx),
                from_q=nodes[idx],
                to_q=goal_q,
                collision=True,
                collision_pairs=goal_pairs,
                collision_q=collision_q,
                collision_alpha=collision_alpha,
                accepted=False,
                reason="goal_connection_collision",
                node_count=len(nodes),
                elapsed_s=None if total_t0 is None else time.perf_counter() - total_t0,
                phase_elapsed_s=collision_elapsed,
                collision_check_elapsed_s=collision_elapsed,
                new_node_collision_count=stats.get("edge_collision_rejects", 0),
                random_new_node_collision_count=stats.get("random_edge_collision_rejects", 0),
                rewire_collision_count=stats.get("rewire_collision_rejects", 0),
            )

        return goal_idx, min_total_cost, close_indices

    def _reconstruct_joint_path(self, nodes, parents, goal_idx, goal_q):
        """parent chain으로 최종 q path를 복원한다.

        Args:
            nodes: tree의 raw q node list.
            parents: node index -> parent index dict.
            goal_idx: goal_q와 연결되는 마지막 tree node index.
            goal_q: 목표 raw q.

        Returns:
            start_q부터 goal_q까지 순서대로 정렬된 raw q waypoint list.

        계산 과정:
            goal_q를 path 끝에 넣고, goal_idx에서 parent를 따라 root까지 거슬러 올라간 뒤 뒤집는다.
        """
        path = [np.asarray(goal_q, dtype=float).copy()]
        curr_idx = goal_idx
        while curr_idx is not None:
            path.append(nodes[curr_idx].copy())
            curr_idx = parents[curr_idx]
        return path[::-1]
