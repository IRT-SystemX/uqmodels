"""
Numerical operation registry for result expressions.

This module defines the numerical transformations available to the recursive
result computation engine.

Each registered operation specifies:
- its expected number of operands;
- the numerical transformation applied to operand values;
- the associated uncertainty propagation rule when available.

Typical operations include difference, sum, product, ratio, relative
difference, minimum, and maximum.

The registry is independent from KPIView construction, result selection, and
visualization, allowing new numerical transformations to be introduced without
modifying the computation engine.
"""
import inspect
import numpy as np
def _relative_difference(values, threshold=0):
    """
    Compute the relative difference with denominator stabilization.

    Formula
    -------
    (target - reference) / (reference + threshold)
    """
    return (
        values[0] - values[1]
    ) / (
        values[1] + threshold
    )


def _relative_difference_std(values, stds, threshold=0):
    """
    Propagate uncertainty for the stabilized relative difference.

    Assumes independent uncertainties on target and reference.
    """
    denominator = values[1] + threshold

    return np.sqrt(
        (stds[0] / denominator) ** 2
        + (
            (values[0] + threshold)
            * stds[1]
            / denominator**2
        ) ** 2
    )

OPERATION_REGISTRY = {
    "max": {
        "arity": 2,
        "compute": lambda values: np.maximum(
            values[0],
            values[1],
        ),
        "propagate_std": lambda values, stds: np.where(
            values[0] >= values[1],
            stds[0],
            stds[1],
        ),
    },

    "min": {
        "arity": 2,
        "compute": lambda values: np.minimum(
            values[0],
            values[1],
        ),
        "propagate_std": lambda values, stds: np.where(
            values[0] <= values[1],
            stds[0],
            stds[1],
        ),
    },
    
    "difference": {
        "arity": 2,
        "compute": lambda values: values[0] - values[1],
        "propagate_std": lambda values, stds: np.sqrt(
            stds[0]**2 + stds[1]**2
        ),
    },

    "sum": {
        "arity": 2,
        "compute": lambda values: values[0] + values[1],
        "propagate_std": lambda values, stds: np.sqrt(
            stds[0]**2 + stds[1]**2
        ),
    },

    "product": {
        "arity": 2,
        "compute": lambda values: values[0] * values[1],
        "propagate_std": lambda values, stds: np.sqrt(
            (values[1] * stds[0])**2
            + (values[0] * stds[1])**2
        ),
    },

    "ratio": {
        "arity": 2,
        "compute": lambda values: values[0] / values[1],
        "propagate_std": lambda values, stds: np.sqrt(
            (stds[0] / values[1])**2
            + (
                values[0]
                * stds[1]
                / values[1]**2
            )**2
        ),
    },

    "relative_difference": {
    "arity": 2,
    "compute": _relative_difference,
    "propagate_std": _relative_difference_std,
    },  
}

def apply_operation(
    operation_type,
    operands,
    with_std=False,
    registry=None,
    parameters=None):
    """
    Apply a registered numerical operation.
    """
    parameters = parameters or {}
    registry = registry or OPERATION_REGISTRY

    if operation_type not in registry:
        raise ValueError(
            f"Unknown operation '{operation_type}'. "
            f"Available operations: {list(registry)}"
        )

    operation = registry[operation_type]

    if len(operands) != operation["arity"]:
        raise ValueError(
            f"Operation '{operation_type}' expects "
            f"{operation['arity']} operands, "
            f"received {len(operands)}."
        )

    values = [
        operand["means"]
        for operand in operands
    ]

    stds = [
        operand["stds"]
        for operand in operands
    ]

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        
        compute_fn = operation["compute"]

        _validate_operation_parameters(
            operation_type=operation_type,
            function=compute_fn,
            parameters=parameters,
            function_name="compute")

        means = compute_fn(values,**parameters)

        if with_std:
            propagation = operation["propagate_std"]

            if propagation is None:
                raise ValueError(
                    f"No uncertainty propagation defined "
                    f"for operation '{operation_type}'."
                )

            result_stds = propagation(values,stds,**parameters)

        else:
            result_stds = None
    return means, result_stds


def _validate_operation_parameters(
    operation_type,
    function,
    parameters,
    function_name,
):
    """
    Validate parameters passed to an operation function.
    """
    if not parameters:
        return

    signature = inspect.signature(function)
    accepted_parameters = set(signature.parameters)

    unsupported_parameters = (
        set(parameters) - accepted_parameters
    )

    if unsupported_parameters:
        supported_parameters = (
            accepted_parameters
            - {"values", "stds"}
        )

        raise ValueError(
            f"Operation '{operation_type}' does not support "
            f"parameter(s) {sorted(unsupported_parameters)} "
            f"for '{function_name}'. "
            f"Supported parameters are: "
            f"{sorted(supported_parameters)}."
        )