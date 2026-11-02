from plugins.pathplanner.pp_common import PPMotionPlannerBase


class PPLazyPRM(PPMotionPlannerBase):
    """pybullet_planning 'lazy_prm' on the project's q-space/PyBullet collision (see pp_common.py)."""
    ALGORITHM = "lazy_prm"
    LABEL = "pp_lazy_prm"
