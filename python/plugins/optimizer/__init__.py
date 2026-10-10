"""Registry of path-optimizer plugin module names (mirrors
plugins.pathplanner.Q_SPACE_PLANNER_MODULES's role for planners): the module
stem each optimizer's .py/.json pair lives under, importable via
plugins.optimizer.<name> and loadable by Visualizer._load_path_optimizer().

Each one implements OptimizerBase.optimize(path, planner) - see
plugins/pluginbase/optimizerbase.py. None of them plan from scratch; they all
take an already-generated q_path (from any PathPlanner, most usefully
direct_path's pure straight-line interpolation - see direct_path.py) and
smooth/shortcut/reoptimize it, using `planner` only for collision queries
(planner._check_collision / planner.check_collision_on_path / planner.scene).

There is no "chomp" module - GPMP2 (Gaussian Process Motion Planning) is the
closest available gradient-based trajectory optimizer in this codebase, and
TrajOpt (sequential convex optimization) is conceptually the closest to
CHOMP's "smooth + avoid obstacles" objective if you were looking for that
specifically. A literal CHOMP implementation would need to be added as a new
module if one of these doesn't fit.
"""

OPTIMIZER_MODULES = frozenset({
    "bspline", "gpmp2", "path_pruning", "stomp", "topp_ra", "trajopt",
})
