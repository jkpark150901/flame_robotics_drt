from plugins.pathplanner.pp_common import PPMotionPlannerBase


class PPRRTStar(PPMotionPlannerBase):
    """pybullet_planning 'rrt_star' on the project's q-space/PyBullet collision (see pp_common.py)."""
    ALGORITHM = "rrt_star"
    LABEL = "pp_rrt_star"
