import numpy as np
import sys
import os

# Adjust path to import OptimizerBase
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../pluginbase')))
from plugins.pluginbase.optimizerbase import OptimizerBase

class PathPruning(OptimizerBase):
    def __init__(self, config_path: str = None):
        if config_path is None:
             config_path = os.path.splitext(__file__)[0] + '.json'
        super().__init__(config_path)
        
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except:
            self.config = {}

    def optimize(self, path: list, planner) -> list:
        if not path or len(path) < 3:
            return path
        
        # Shortcut Pruning
        # Try to connect current node to furthest possible node
        optimized_path = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            # Check connection to subsequent nodes in reverse order
            next_idx = current_idx + 1
            
            for i in range(len(path) - 1, current_idx + 1, -1):
                # Check collision for straight line between current and i
                # Using planner's collision check which assumes discrete steps or raycast
                # To be safe for pruning, we should check finely or utilize the raycast if available.
                # PlannerBase usually relies on some _check_collision(p1, p2).
                
                # We assume planner has _check_collision(p1, p2)
                # If valid, we prune everything in between
                
                if not planner._check_collision(path[current_idx], path[i]):
                    next_idx = i
                    break
            
            optimized_path.append(path[next_idx])
            current_idx = next_idx
            
        return optimized_path
