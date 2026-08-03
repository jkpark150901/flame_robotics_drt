def create_optimization_objective(objective_type, space_information, base_module):
    objective_type = str(objective_type or "path_length").strip().lower()
    if objective_type != "path_length":
        raise ValueError(f"unsupported OMPL optimization objective: {objective_type}")
    return base_module.PathLengthOptimizationObjective(space_information)
