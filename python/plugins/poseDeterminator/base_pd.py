"""
PlugIn Base Classes for DRT
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from util.logger.console import ConsoleLogger


class PluginPDBase(ABC):
    """ Pose Detection Plugin Base Class """

    @abstractmethod
    def find_pose_candidates(self, pcd:np.ndarray, g_point:np.ndarray) -> List[np.ndarray]:
        """
        (input) pcd : PCD data as numpy array) (shape: (N, 3))
        (Input) g_point : Global 3D Point from 3D Scan Data (*.ply, *.pcd)
        (Output) pose_candidates : List of pose candidates, each pose is represented as a numpy array of shape (4,4) transformation matrix
        """
        pass
