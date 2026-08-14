"""Public path-planner names shared by SimTool and the robot core."""

# Legacy hand-written per-file q-space planners (plugins/pathplanner/<name>.py).
# Unrelated to OMPL/tesseract - these remain available regardless of whether the
# native OMPL bindings are installed.
Q_SPACE_PLANNER_MODULES = frozenset({
    "bit_star",
    "direct_path",
    "informed_rrt_star",
    "prm",
    "rrt",
    "rrt_connect",
    "rrt_star",
})

__all__ = ["Q_SPACE_PLANNER_MODULES"]
