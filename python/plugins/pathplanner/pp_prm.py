from plugins.pathplanner.pp_common import PPMotionPlannerBase


class PPPRM(PPMotionPlannerBase):
    """pybullet_planning 'prm' on the project's q-space/PyBullet collision (see pp_common.py)."""
    ALGORITHM = "prm"
    LABEL = "pp_prm"
