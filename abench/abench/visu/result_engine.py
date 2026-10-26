"""
Generic recursive result computation engine.

This module evaluates ResultView expressions over structured numerical result
grids.

Expressions are recursively composed of:
- source operands with optional dimension selections;
- numerical operations whose operands may themselves be expressions.

This representation supports direct metrics, multi-source derived KPIs,
dimension-based comparisons, and combinations of both, such as comparing a
derived KPI against a reference model or experiment.

Main responsibilities:
- resolve selections over named result dimensions;
- extract source operands from structured result grids;
- recursively evaluate nested computation expressions;
- delegate numerical transformations and uncertainty propagation to the
  operation registry;
- return computed values, uncertainties, and retained dimension indices.

The engine is independent from KPI semantics and visualization.
"""

import numpy as np
from .result_operations import apply_operation

def _compute_operand(
    dict_results,
    operand,
    dimension_map,
    with_std=False,
):
    """
    Resolve one operand from structured results.
    """
    source = operand["source"]

    if source not in dict_results:
        raise KeyError(
            f"Unknown result source: '{source}'."
        )

    values = dict_results[source]

    means_all = values[..., 0]

    if with_std:
        stds_all = values[..., 1]
    else:
        stds_all = None

    selection = operand.get("selection")

    means, selected_indices = _apply_result_selection(
        data=means_all,
        selection=selection,
        dimension_map=dimension_map,
    )

    if with_std:

        stds, _ = _apply_result_selection(
            data=stds_all,
            selection=selection,
            dimension_map=dimension_map,
        )

    else:
        stds = None

    return {
        "source": source,
        "sources": [source],
        "means": means,
        "stds": stds,
        "indices": selected_indices,
    }

def _resolve_selection_indices(size, selector):
    """
    Resolve a dimension selector into explicit integer indices.

    Parameters
    ----------
    size : int
        Size of the selected dimension.
    selector : dict or None
        Selection specification supporting:
        - {"include": [...]}
        - {"exclude": [...]}

    Returns
    -------
    np.ndarray
        Selected integer indices.
    """
    all_indices = np.arange(size)

    if selector is None:
        return all_indices

    if "include" in selector:
        indices = np.asarray(selector["include"], dtype=int)

    elif "exclude" in selector:
        excluded = np.asarray(selector["exclude"], dtype=int)
        mask = ~np.isin(all_indices, excluded)
        indices = all_indices[mask]

    else:
        raise ValueError(
            "Selection must define either 'include' or 'exclude'."
        )

    if np.any(indices < 0) or np.any(indices >= size):
        raise IndexError(
            f"Selection indices {indices} are invalid for dimension size {size}."
        )

    return indices

def _apply_result_selection(data, selection, dimension_map):
    """
    Apply named-dimension selections to an array.

    Parameters
    ----------
    data : np.ndarray
        Input numerical array.
    selection : dict or None
        Selection specification indexed by dimension name.
    dimension_map : dict
        Mapping between dimension names and array axes.

    Returns
    -------
    selected_data : np.ndarray
        Selected array.
    selected_indices : dict
        Retained indices for each structured dimension.
    """
    selection = selection or {}

    selected_data = data
    selected_indices = {}

    for dimension, axis in dimension_map.items():

        selector = selection.get(dimension)

        indices = _resolve_selection_indices(
            size=data.shape[axis],
            selector=selector,
        )

        selected_data = np.take(
            selected_data,
            indices,
            axis=axis,
        )

        selected_indices[dimension] = indices

    return selected_data, selected_indices



def compute_result_view(
    dict_results,
    result_view,
    dimension_map,
    with_std=False,
):
    """
    Evaluate one ResultView.
    """
    result = _evaluate_expression(
        dict_results=dict_results,
        expression=result_view["expression"],
        dimension_map=dimension_map,
        with_std=with_std,
    )

    return {
        "name": result_view["name"],
        "sources": result.get("sources", []),
        "view_type": result_view.get(
            "view_type",
            "absolute",
        ),
        "means": result["means"],
        "stds": result["stds"],
        "indices": result["indices"],
        "style": result_view.get("style", {}),
    }

def _evaluate_expression(
    dict_results,
    expression,
    dimension_map,
    with_std=False,
):
    """
    Recursively evaluate a computation expression.
    """

    # --------------------------------------------------------------
    # Source operand
    # --------------------------------------------------------------
    if "source" in expression:

        return _compute_operand(
            dict_results=dict_results,
            operand=expression,
            dimension_map=dimension_map,
            with_std=with_std,
        )

    # --------------------------------------------------------------
    # Nested operation
    # --------------------------------------------------------------
    if "operation" in expression:

        operation = expression["operation"]

        operands = [
            _evaluate_expression(
                dict_results=dict_results,
                expression=operand,
                dimension_map=dimension_map,
                with_std=with_std,
            )
            for operand in operation["operands"]
        ]

        means, stds = apply_operation(
            operation_type=operation["type"],
            operands=operands,
            with_std=with_std,
            parameters=operation.get("parameters", {}))

        # First operand defines the output dimensions.
        return {
            "means": means,
            "stds": stds,
            "indices": operands[0]["indices"],
            "sources": [
                source
                for operand in operands
                for source in operand.get(
                    "sources",
                    [operand.get("source")],
                )
                if source is not None
            ],
        }

    raise ValueError(
        "Invalid expression: expected 'source' or 'operation'."
    )