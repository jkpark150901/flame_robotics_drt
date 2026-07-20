from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class RobotDescription:
    """Robot model registration data independent of the backend library."""

    name: str
    urdf_path: str
    base_T: np.ndarray = field(default_factory=lambda: np.eye(4))
    package_dirs: Optional[List[str]] = None
    target_frame: Optional[str] = None


@dataclass
class IKOptions:
    """Common IK options passed from UI/planner code."""

    solver: str = "normalized_dls"
    normalize: Optional[bool] = None
    damping: float = 1e-3
    dt: float = 0.35
    tol: float = 1e-4
    max_iter: int = 1000
    position_only_tol: float = 0.01
    backend_solver: str = "quadprog"
    record_trace: bool = True


@dataclass
class IKTracePoint:
    iteration: int
    q: np.ndarray
    err_norm: float
    position_error: float
    orientation_error: float
    tcp_world: np.ndarray


@dataclass
class IKResult:
    success: bool
    q: Optional[np.ndarray]
    solver: str
    normalize: bool
    iterations: int
    elapsed: float
    position_only: bool = False
    position_error: float = float("inf")
    orientation_error: float = float("inf")
    final_T: Optional[np.ndarray] = None
    target_T: Optional[np.ndarray] = None
    trace: List[IKTracePoint] = field(default_factory=list)
    failure_info: Dict[str, Any] = field(default_factory=dict)
    message: str = ""


@dataclass
class CollisionResult:
    collision: bool
    pairs: List[Tuple[str, str]] = field(default_factory=list)
    q: Optional[np.ndarray] = None
    alpha: Optional[float] = None
    backend: str = ""


class RoboticsBackend(ABC):
    """Library-neutral robot backend interface.

    Viewer, path planners, and experiments should depend on this interface
    instead of importing a concrete solver/collision library directly.
    """

    name = "abstract"

    @abstractmethod
    def register_robot(self, description: RobotDescription) -> Any:
        """Load/register one robot and return a backend-specific handle."""

    @abstractmethod
    def robot_model(self, robot_name: str) -> Any:
        """Return backend-specific robot model object."""

    @abstractmethod
    def joint_names(self, robot_name: str) -> List[str]:
        """Return actuated joint names in q order."""

    @abstractmethod
    def neutral_q(self, robot_name: str) -> np.ndarray:
        """Return a neutral q vector for the robot."""

    @abstractmethod
    def joint_limits_for_metric(self, robot_name: str, normalize: bool = True):
        """Return lower/upper/span used by q-space metrics."""

    @abstractmethod
    def normalize_q(self, robot_name: str, q: Sequence[float], normalize: bool = True) -> np.ndarray:
        """Convert raw q to metric q."""

    @abstractmethod
    def denormalize_q(self, robot_name: str, q_metric: Sequence[float], normalize: bool = True) -> np.ndarray:
        """Convert metric q back to raw q."""

    @abstractmethod
    def joint_distance(
        self,
        robot_name: str,
        q_a: Sequence[float],
        q_b: Sequence[float],
        normalize: bool = True,
    ) -> float:
        """Return distance between two raw q vectors in backend metric space."""

    @abstractmethod
    def joint_distances(
        self,
        robot_name: str,
        q_points: Sequence[Sequence[float]],
        q_ref: Sequence[float],
        normalize: bool = True,
    ) -> np.ndarray:
        """Return distances from many raw q vectors to one raw q vector."""

    @abstractmethod
    def steer_joint_state(
        self,
        robot_name: str,
        from_state: Sequence[float],
        to_state: Sequence[float],
        step_size: float,
        normalize: bool = True,
    ) -> np.ndarray:
        """Move from one raw q toward another by step_size in metric space."""

    @abstractmethod
    def sample_configuration(self, robot_name: str) -> np.ndarray:
        """Sample one raw q inside backend joint limits."""

    @abstractmethod
    def end_effector_collision_geometry(
        self,
        robot_name: str,
        end_link_name: str,
        tcp_joint_name,
        pose_to_link_offset=None,
    ):
        """Return end-effector collision mesh and tcp-to-link transform."""

    @abstractmethod
    def frame_id(self, robot_name: str, frame_name: Optional[str] = None) -> int:
        """Return backend frame id for a named frame."""

    @abstractmethod
    def frame_world_T(self, robot_name: str, q: Sequence[float], frame_name: Optional[str] = None) -> np.ndarray:
        """Return world transform of a robot frame."""

    @abstractmethod
    def target_world_T(
        self,
        robot_name: str,
        target_world: Any,
        q_reference: Sequence[float],
        frame_name: Optional[str] = None,
    ) -> np.ndarray:
        """Resolve a pose/vector target into a world-frame 4x4 target transform."""

    @abstractmethod
    def solve_ik(
        self,
        robot_name: str,
        target_world_T: np.ndarray,
        q_init: Sequence[float],
        options: Optional[IKOptions] = None,
        frame_name: Optional[str] = None,
    ) -> IKResult:
        """Solve IK for one robot frame."""

    @abstractmethod
    def classify_ik_failure(
        self,
        robot_name: str,
        q: Sequence[float],
        target_world_T: np.ndarray,
        final_T: Optional[np.ndarray],
        orientation_error: float = float("inf"),
        max_iter: int = 0,
    ) -> Dict[str, Any]:
        """Classify a failed IK result using joint limits and final pose error."""

    @abstractmethod
    def configure_collision(
        self,
        robot_name: str,
        static_meshes: Optional[Iterable[Any]] = None,
        sample_resolution: float = 0.05,
    ) -> None:
        """Build or refresh collision data for one robot."""

    @abstractmethod
    def check_collision(self, robot_name: str, q: Sequence[float], return_pairs: bool = False) -> CollisionResult:
        """Check collision at one q."""

    @abstractmethod
    def check_edge_collision(
        self,
        robot_name: str,
        q_from: Sequence[float],
        q_to: Sequence[float],
        return_pairs: bool = False,
    ) -> CollisionResult:
        """Check collision along a q-space edge."""

    @abstractmethod
    def check_mesh_point_cloud_overlap(
        self,
        link_model: Any,
        tcp_pose: Sequence[float],
        tcp_to_link_pose_T: np.ndarray,
        scan_data: Any,
        margin: float = 0.05,
        sample_count: int = 5000,
        threshold: float = 0.001,
    ) -> bool:
        """Check whether a transformed link mesh overlaps a point cloud."""
