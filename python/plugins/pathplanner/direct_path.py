import numpy as np
import json
import os
from typing import List, Union

from plugins.pluginbase.plannerbase import PlannerBase


class DirectPath(PlannerBase):
    """테스트용 단순 path planner.

    시작 상태에서 목표 상태까지 충돌 검사 없이 직선(선형 보간) 경로만 만든다.
    q-space 모델이 설정되어 있으면 raw q를 선형 보간하고, 그렇지 않으면 6D pose를
    선형 보간한다. 다른 planner와 비교/디버깅할 때 baseline으로 쓴다.
    """

    def __init__(self, config_path: str = None):
        super().__init__()
        if config_path is None:
            config_path = os.path.splitext(__file__)[0] + '.json'

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.step_size = self.config.get("step_size", 0.1)
        self.max_iter = self.config.get("max_iter", 1)
        self.bounds = self.config.get("workspace_bounds", {
            "x_min": -10.0, "x_max": 10.0,
            "y_min": -10.0, "y_max": 10.0,
            "z_min": -10.0, "z_max": 10.0,
        })

        self.configure_collision(self.config, default_sample_resolution=self.step_size)

    def generate(
        self,
        current_pose: Union[List[float], np.ndarray],
        target_pose: Union[List[float], np.ndarray],
        step_callback=None,
    ) -> List[np.ndarray]:
        current_pose = np.asarray(current_pose, dtype=float)
        target_pose = np.asarray(target_pose, dtype=float)

        dof = self._robot_dof()
        if (
            self._has_robot_q_space_model()
            and current_pose.shape[0] == dof
            and target_pose.shape[0] == dof
        ):
            return self._generate_joint_space(current_pose, target_pose, step_callback=step_callback)
        if self._has_robot_q_space_model():
            raise ValueError(
                "DirectPath is configured for robot q-space planning, so generate() "
                f"must receive q-space states with dof={dof}; "
                f"got {current_pose.shape[0]}->{target_pose.shape[0]}"
            )
        return self._generate_workspace(current_pose, target_pose, step_callback=step_callback)

    def _generate_joint_space(self, start_q, goal_q, step_callback=None):
        start_q = np.asarray(start_q, dtype=float)
        goal_q = np.asarray(goal_q, dtype=float)
        # normalized joint 거리 기준으로 step 수를 정해 raw q를 선형 보간한다.
        distance = float(self._joint_distance(start_q, goal_q))
        steps = max(1, int(np.ceil(distance / max(float(self.step_size), 1e-9))))
        path = [start_q + (goal_q - start_q) * (i / steps) for i in range(steps + 1)]
        if step_callback is not None:
            try:
                step_callback({"nodes": [np.asarray(q, dtype=float) for q in path]})
            except Exception:
                pass
        return path

    def _generate_workspace(self, current_pose, target_pose, step_callback=None):
        current_pose = np.asarray(current_pose, dtype=float)
        target_pose = np.asarray(target_pose, dtype=float)
        # 목표 orientation에 NaN이 있으면 시작 orientation을 유지한다.
        goal = target_pose.copy()
        if goal.shape[0] > 3 and current_pose.shape[0] > 3:
            nan_mask = np.isnan(goal[3:])
            goal[3:][nan_mask] = current_pose[3:][nan_mask]
        distance = float(np.linalg.norm((goal[:3] - current_pose[:3])))
        steps = max(1, int(np.ceil(distance / max(float(self.step_size), 1e-9))))
        return [current_pose + (goal - current_pose) * (i / steps) for i in range(steps + 1)]
