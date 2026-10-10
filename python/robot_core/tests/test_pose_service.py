import numpy as np

from robot_core.pose_service import PoseDeterminationService


class _Optimizer:
    def calculate_pipe_profile(self, target, **kwargs):
        self.target = np.asarray(target, dtype=float)

    def calculate_DDA_RT_pose_for_taking_xray(self, target, **kwargs):
        return [{
            "name": "Inspection pose 1",
            "target_point": np.asarray(target, dtype=float),
            "dda_pose": np.eye(4),
            "rt_pose": np.eye(4),
        }]


def test_pose_result_is_created_and_serialized_by_robot_core(monkeypatch):
    service = PoseDeterminationService({}, robotics_backend=object())
    monkeypatch.setattr(service, "_point_cloud", lambda _points: object())
    monkeypatch.setattr(
        service,
        "_optimizer",
        lambda _cloud: (_Optimizer(), {}, np.array([0.0, -1.0, 0.0])),
    )

    result = service.determine([[1.0, 2.0, 3.0]], np.zeros((10, 3)))

    assert result["status"] == "success"
    assert result["source"] == "robot_core"
    assert result["target_group_count"] == 1
    group = result["target_groups"][0]
    assert group["name"] == "Point 1 - Inspection pose 1"
    assert group["index"] == 0
    assert isinstance(group["dda_pose"], list)


def test_pose_failure_payload_is_created_by_robot_core():
    service = PoseDeterminationService({}, robotics_backend=object())
    result = service.determine([], [])

    assert result["status"] == "failed"
    assert result["source"] == "robot_core"
    assert "not selected" in result["message"]
