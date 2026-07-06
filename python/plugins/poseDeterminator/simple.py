"""
Pose Detection Algorithm Plugin
"""

from python.plugins.plugin_pdm.base_pd import PluginPDBase
from typing import List, Tuple
from util.logger.console import ConsoleLogger

class Simple(PluginPDBase):
    def __init__(self):
        super().__init__()
        self.__console = ConsoleLogger.get_logger()  # logger
        self.__console.info("Simple PDM Plugin Initialized")

    def find_pose_candidates(self, g_points:Tuple[float, float, float]) -> List[Tuple[float, float, float, float, float, float]]:
        self.__console.error(f"({self.__class__.__name__}) called find_pose_candidate function")