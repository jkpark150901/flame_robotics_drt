from plugins.pathplanner.pp_common import PPMotionPlannerBase


class PPBiRRT(PPMotionPlannerBase):
    """pybullet_planning 'birrt' on the project's q-space/PyBullet collision (see pp_common.py)."""
    ALGORITHM = "birrt"
    LABEL = "pp_birrt"
