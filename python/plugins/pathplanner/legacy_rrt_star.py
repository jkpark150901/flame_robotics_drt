from plugins.pathplanner.rrt_star import RRTStar


class LegacyRRTStar(RRTStar):
    """Explicit rollback entry point for the in-house RRT* implementation."""

    def __init__(self, config_path=None):
        super().__init__(config_path)
        self.planner_backend = "legacy"
        if config_path is None:
            self.debug_output_dir = "debug/legacy_rrt_star"
