import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Union


class PoseFinderBase(ABC):
    """
    Abstract base class for pose finder algorithms.
    Provides an interface for finding object poses from point cloud data.
    """

    def __init__(self, config_path: str = None):
        self.config = {}

    @abstractmethod
    def find_pose_candidates(
        self,
        ply_filepath: str,
        input_coordinates: np.ndarray,
        max_candidates: int,
        search_parameters: Dict,
    ) -> List[List[Dict[str, Union[np.ndarray, float, Dict]]]]:
        """
        Find pose candidates from a point cloud.

        Args:
            ply_filepath: File path to the PLY point cloud file.
            input_coordinates: Input coordinates as a numpy array.
            max_candidates: Maximum number of candidates to return.
            search_parameters: Dictionary containing search parameters.

        Returns:
            List of lists containing dictionaries with pose candidate information.
        """
        pass
