import unittest
import numpy as np
import sys
import os

# Adjust path to import core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../core/algorithms')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../core/pluginbase')))

from rrt_connect import RRTConnect
from rrt_star import RRTStar
from informed_rrt_star import InformedRRTStar
from task_space_rrt import TaskSpaceRRT
from task_space_rrt_star import TaskSpaceRRTStar

class TestRRTAlgorithms(unittest.TestCase):
    def test_rrt_connect_empty(self):
        planner = RRTConnect()
        planner.max_iter = 500
        start = [0, 0, 0, 0, 0, 0]
        goal = [5, 5, 5, 0, 0, 0]
        path = planner.generate(start, goal)
        self.assertTrue(len(path) > 0, "RRT-Connect should find path")
        
    def test_rrt_star_empty(self):
        planner = RRTStar()
        planner.max_iter = 500
        start = [0, 0, 0, 0, 0, 0]
        goal = [5, 5, 5, 0, 0, 0]
        path = planner.generate(start, goal)
        self.assertTrue(len(path) > 0, "RRT* should find path")

    def test_informed_rrt_star_empty(self):
        planner = InformedRRTStar()
        planner.max_iter = 500
        start = [0, 0, 0, 0, 0, 0]
        goal = [5, 5, 5, 0, 0, 0]
        path = planner.generate(start, goal)
        self.assertTrue(len(path) > 0, "Informed RRT* should find path")

    def test_task_space_rrt_empty(self):
        planner = TaskSpaceRRT()
        planner.max_iter = 500
        start = [0, 0, 0, 0, 0, 0]
        goal = [5, 5, 5, 0, 0, 0]
        path = planner.generate(start, goal)
        self.assertTrue(len(path) > 0, "Task-space RRT should find path")

    def test_task_space_rrt_star_empty(self):
        planner = TaskSpaceRRTStar()
        planner.max_iter = 500
        start = [0, 0, 0, 0, 0, 0]
        goal = [5, 5, 5, 0, 0, 0]
        path = planner.generate(start, goal)
        self.assertTrue(len(path) > 0, "Task-space RRT* should find path")

if __name__ == '__main__':
    unittest.main()
