from plugins.pathplanner.pp_common import PPMotionPlannerBase


class PPRRT(PPMotionPlannerBase):
    """pybullet_planning 'rrt' on the project's q-space/PyBullet collision (see pp_common.py)."""
    ALGORITHM = "rrt"
    LABEL = "pp_rrt"
