from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class OMPLRunMetrics:
    algorithm: str
    solve_time: float = 0.0
    first_solution_time: float = None
    best_cost: float = None
    vertex_count: int = 0
    edge_count: int = 0
    exact: bool = False
    approximate: bool = False
    status: str = "not_started"
    counters: Dict[str, int] = field(default_factory=dict)
    final_verification: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self):
        return {
            "algorithm": self.algorithm,
            "solve_time": self.solve_time,
            "first_solution_time": self.first_solution_time,
            "best_cost": self.best_cost,
            "vertex_count": self.vertex_count,
            "edge_count": self.edge_count,
            "exact": self.exact,
            "approximate": self.approximate,
            "status": self.status,
            "counters": dict(self.counters),
            "final_verification": dict(self.final_verification),
        }
