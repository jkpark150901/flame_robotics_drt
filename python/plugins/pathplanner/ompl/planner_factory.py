import logging


_PLANNER_NAMES = {
    "rrt": "RRT",
    "rrt_connect": "RRTConnect",
    "rrt_star": "RRTstar",
    "informed_rrt_star": "InformedRRTstar",
    "bit_star": "BITstar",
}


def supported_algorithms():
    return tuple(_PLANNER_NAMES)


def create_ompl_planner(algorithm, space_information, config, geometric_module):
    key = str(algorithm).strip().lower()
    class_name = _PLANNER_NAMES.get(key)
    if class_name is None:
        raise ValueError(f"unsupported OMPL algorithm: {algorithm}")
    planner_class = getattr(geometric_module, class_name, None)
    if planner_class is None:
        raise RuntimeError(f"installed OMPL binding does not provide {class_name}")
    planner = planner_class(space_information)
    apply_supported_parameters(planner, key, config or {})
    return planner


def apply_supported_parameters(planner, algorithm, config):
    planner_config = config.get("planner", {}) or {}
    values = {
        "range": config.get("runtime_step_size", planner_config.get("range", config.get("step_size"))),
        "goal_bias": config.get("runtime_goal_bias", planner_config.get("goal_bias", config.get("goal_bias"))),
        "rewire_factor": planner_config.get("rewire_factor"),
    }
    _apply(planner, "setRange", values["range"], "range")
    _apply(planner, "setGoalBias", values["goal_bias"], "goal_bias")
    _apply(planner, "setRewireFactor", values["rewire_factor"], "rewire_factor")

    if algorithm == "bit_star":
        bit_config = planner_config.get("bit_star", {}) or {}
        _apply(
            planner,
            "setSamplesPerBatch",
            bit_config.get("samples_per_batch", config.get("batch_size")),
            "samples_per_batch",
        )
        _apply(planner, "setPruning", bit_config.get("pruning"), "pruning")


def _apply(planner, method_name, value, parameter_name):
    if value is None:
        return
    method = getattr(planner, method_name, None)
    if method is None:
        logging.getLogger(__name__).warning(
            "OMPL planner %s does not support parameter %s",
            planner.__class__.__name__,
            parameter_name,
        )
        return
    method(value)
