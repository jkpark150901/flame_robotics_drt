"""Public path-planner names shared by SimTool and the robot core."""

# Legacy hand-written per-file q-space planners (plugins/pathplanner/<name>.py).
# Unrelated to OMPL/tesseract - these remain available regardless of whether the
# native OMPL bindings are installed.
Q_SPACE_PLANNER_MODULES = frozenset({
    "bit_star",
    "direct_path",
    "informed_rrt_star",
    "pp_birrt",          # pybullet_planning (yijiangh) algorithms on the project's q-space/PyBullet collision,
    "pp_lazy_prm",       # see plugins/pathplanner/pp_common.py
    "pp_prm",
    "pp_rrt",
    "pp_rrt_star",
    "prm",
    "rrt",
    "rrt_connect",
    "rrt_star",
})

__all__ = ["Q_SPACE_PLANNER_MODULES"]
