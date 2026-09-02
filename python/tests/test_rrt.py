import unittest
import numpy as np
import sys
import os

# Adjust path to import core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../core/algorithms')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../core/pluginbase')))

from rrt import RRT

class TestRRT(unittest.TestCase):
    def setUp(self):
        self.planner = RRT()
        # Override config for testing speed
        self.planner.max_iter = 500
        self.planner.step_size = 1.0
        self.planner.goal_bias = 0.2
        self.planner.bounds = {
            "x_min": -10.0, "x_max": 10.0,
            "y_min": -10.0, "y_max": 10.0,
            "z_min": -10.0, "z_max": 10.0
        }

    def test_instantiation(self):
        self.assertIsInstance(self.planner, RRT)

    def test_generate_path_empty_space(self):
        start = [0, 0, 0, 0, 0, 0]
        goal = [5, 5, 5, 0, 0, 0]
        path = self.planner.generate(start, goal)
        
        self.assertTrue(len(path) > 0, "Should find a path in empty space")
        
        # Check start and goal
        np.testing.assert_array_almost_equal(path[0][:3], start[:3])
        np.testing.assert_array_almost_equal(path[-1][:3], goal[:3])

if __name__ == '__main__':
    unittest.main()
