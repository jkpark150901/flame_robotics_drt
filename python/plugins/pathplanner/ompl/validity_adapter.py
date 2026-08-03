import time


class StateValidityAdapter:
    """Bridge an OMPL state validity callback to PlannerBase collision checks."""

    def __init__(self, planner, codec, stats):
        self.planner = planner
        self.codec = codec
        self.stats = stats
        self.last_exception = None

    def __call__(self, state):
        self.stats["state_validity_calls"] += 1
        try:
            deadline = getattr(self.planner, "planning_deadline", None)
            if deadline is not None and time.monotonic() > float(deadline):
                self.stats["deadline_rejects"] += 1
                return False
            q = self.codec.state_to_full_q(state)
            if not self.planner._workspace_position_ok(q):
                self.stats["workspace_rejects"] += 1
                return False
            if self.planner.check_robot_collision(q):
                self.stats["state_collision_rejects"] += 1
                return False
            return True
        except TimeoutError as exc:
            self.last_exception = exc
            self.stats["deadline_rejects"] += 1
            return False
        except Exception as exc:
            self.last_exception = exc
            self.stats["callback_errors"] += 1
            return False
