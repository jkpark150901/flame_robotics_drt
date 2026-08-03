from typing import Dict, Iterable, Optional

import numpy as np


class JointStateCodec:
    """Convert full raw robot configurations to normalized active-joint states."""

    def __init__(
        self,
        full_dof: int,
        lower,
        upper,
        active_indices: Iterable[int],
        fixed_values: Optional[Dict[int, float]] = None,
        reference_q=None,
    ):
        self.full_dof = int(full_dof)
        self.lower = np.asarray(lower, dtype=float).reshape(self.full_dof)
        self.upper = np.asarray(upper, dtype=float).reshape(self.full_dof)
        if not np.all(np.isfinite(self.lower)) or not np.all(np.isfinite(self.upper)):
            raise ValueError("OMPL joint bounds must be finite")
        if np.any(self.upper <= self.lower):
            raise ValueError("OMPL joint upper bounds must be greater than lower bounds")

        self.active_indices = np.asarray(list(active_indices), dtype=int)
        if self.active_indices.size == 0:
            raise ValueError("OMPL state must contain at least one active joint")
        if len(set(self.active_indices.tolist())) != self.active_indices.size:
            raise ValueError("active joint indices must be unique")
        if np.any(self.active_indices < 0) or np.any(self.active_indices >= self.full_dof):
            raise ValueError("active joint index is out of range")

        self.fixed_values = {int(k): float(v) for k, v in (fixed_values or {}).items()}
        if set(self.active_indices.tolist()).intersection(self.fixed_values):
            raise ValueError("active and fixed joint indices overlap")
        if any(index < 0 or index >= self.full_dof for index in self.fixed_values):
            raise ValueError("fixed joint index is out of range")

        if reference_q is None:
            reference_q = np.zeros(self.full_dof, dtype=float)
        self.reference_q = np.asarray(reference_q, dtype=float).reshape(self.full_dof).copy()
        self.span = self.upper - self.lower

    @property
    def dimension(self) -> int:
        return int(self.active_indices.size)

    def apply_fixed_joints(self, q_full):
        q = np.asarray(q_full, dtype=float).reshape(self.full_dof).copy()
        for index, value in self.fixed_values.items():
            q[index] = value
        return q

    def full_q_to_state_values(self, q_full):
        q = self.apply_fixed_joints(q_full)
        values = (q[self.active_indices] - self.lower[self.active_indices]) / self.span[self.active_indices]
        return np.minimum(np.maximum(values, 0.0), 1.0)

    def state_to_full_q(self, state):
        values = self._state_values(state)
        if values.size != self.dimension:
            raise ValueError(
                f"OMPL state dimension mismatch: got {values.size}, expected {self.dimension}"
            )
        q = self.reference_q.copy()
        indices = self.active_indices
        q[indices] = self.lower[indices] + values * self.span[indices]
        q = np.minimum(np.maximum(q, self.lower), self.upper)
        return self.apply_fixed_joints(q)

    def _state_values(self, state):
        if isinstance(state, np.ndarray):
            return np.asarray(state, dtype=float).reshape(-1)
        if isinstance(state, (list, tuple)):
            return np.asarray(state, dtype=float).reshape(-1)
        return np.asarray([float(state[i]) for i in range(self.dimension)], dtype=float)
