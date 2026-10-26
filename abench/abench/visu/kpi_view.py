"""
KPIView specification and expression construction utilities.

This module defines the user-facing configuration layer used to describe
result views and derived KPIs.

A KPIView may represent:
- a direct view over one result source;
- a derived KPI combining several sources through a numerical operation;
- a comparison of either a direct or derived KPI against a reference value
  along a structured result dimension.

KPIViews are progressively transformed into generic recursive expressions
composed of source operands, selections, and nested operations. These
expressions are then expanded into executable ResultViews.

Main responsibilities:
- build and normalize KPIView specifications;
- resolve named dimension references;
- construct and manipulate recursive computation expressions;
- expand KPIViews into one or several ResultViews;
- identify the raw result sources required for evaluation.

Numerical evaluation is delegated to the result engine.
"""

from copy import deepcopy
import numpy as np

def _apply_selection_to_expression(
    expression,
    selection,
):
    """
    Recursively apply a dimension selection to all source operands.
    """
    expression = deepcopy(expression)

    # --------------------------------------------------------------
    # Source leaf
    # --------------------------------------------------------------
    if "source" in expression:

        current_selection = deepcopy(
            expression.get("selection", {})
        )

        current_selection.update(
            deepcopy(selection)
        )

        expression["selection"] = current_selection

        return expression

    # --------------------------------------------------------------
    # Nested operation
    # --------------------------------------------------------------
    if "operation" in expression:

        expression["operation"]["operands"] = [
            _apply_selection_to_expression(
                operand,
                selection,
            )
            for operand in expression["operation"]["operands"]
        ]

        return expression

    raise ValueError(
        "Invalid expression: expected 'source' or 'operation'."
    )

def _build_kpi_expression(kpi_view):
    """
    Convert a KPIView into a generic computation expression.
    """
    if "source" in kpi_view:
        return {
            "source": kpi_view["source"],
        }

    if "operation" in kpi_view:
        return {
            "operation": deepcopy(kpi_view["operation"]),
        }

    raise ValueError(
        "KPIView must define either 'source' or 'operation'."
    )

def make_kpi_view(
    source=None,
    name=None,
    compare_axis=None,
    reference=None,
    comparison_operation="difference",
    comparison_parameters=None,
    show_reference=True,
    show_comparison=True,
    operation=None,
    sources=None,
    cmap="RdYlGn_r",
    comparison_cmap="seismic",
    vmin=None,
    vmax=None,
    comparison_vmin=None,
    comparison_vmax=None,
):
    """
    Build a KPIView configuration.

    Supports
    --------
    - direct views over one source;
    - operations between multiple sources;
    - comparison of either direct or derived KPI views against
      a reference value along one dimension.
    """
    style = {
        "absolute": {
            "cmap": cmap,
            "vmin": vmin,
            "vmax": vmax,
            "vmin_mirror": False,
        },
        "comparison": {
            "cmap": comparison_cmap,
            "vmin": comparison_vmin,
            "vmax": comparison_vmax,
            "vmin_mirror": True,
        },
    }

    # --------------------------------------------------------------
    # Multi-source derived KPI
    # --------------------------------------------------------------
    if operation is not None:

        if source is not None:
            raise ValueError(
                "Use either `source` or `sources`, not both."
            )

        if sources is None or len(sources) < 2:
            raise ValueError(
                "`sources` must contain at least two sources "
                "when `operation` is defined."
            )

        view = {
            "name": name or f"{operation}_" + "_".join(sources),
            "operation": {
                "type": operation,
                "operands": [
                    {"source": current_source}
                    for current_source in sources
                ],
            },
            "style": style,
        }

    # --------------------------------------------------------------
    # Single-source KPI
    # --------------------------------------------------------------
    else:

        if source is None:
            raise ValueError(
                "`source` must be provided when no operation is defined."
            )

        if sources is not None:
            raise ValueError(
                "`sources` can only be used when `operation` is defined."
            )

        view = {
            "name": name or source,
            "source": source,
            "style": style,
        }

    # --------------------------------------------------------------
    # Optional comparison against a dimension reference
    # --------------------------------------------------------------
    if compare_axis is not None:

        if reference is None:
            raise ValueError(
                "`reference` must be provided when "
                "`compare_axis` is defined."
            )

        view["comparison"] = {
            "axis": compare_axis,
            "reference": reference,
            "operation": comparison_operation,
            "parameters": comparison_parameters or {},
        }

        view["display"] = {
            "reference": show_reference,
            "comparison": show_comparison,
        }

    elif reference is not None:
        raise ValueError(
            "`reference` cannot be defined without `compare_axis`."
        )

    return view

def make_kpi_views(
    sources,
    compare_axis=None,
    reference=None,
    comparison_operation="difference",
    comparison_parameters=None,
    show_reference=True,
    show_comparison=True,
    cmap="RdYlGn_r",
    comparison_cmap="seismic",
    vmin=None,
    vmax=None,
    comparison_vmin=None,
    comparison_vmax=None,
    ):
    """
    Build multiple KPIView configurations with shared settings.
    """
    return [
    make_kpi_view(
        source=source,
        compare_axis=compare_axis,
        reference=reference,
        comparison_operation=comparison_operation,
        comparison_parameters=comparison_parameters,
        show_reference=show_reference,
        show_comparison=show_comparison,
        cmap=cmap,
        comparison_cmap=comparison_cmap,
        vmin=vmin,
        vmax=vmax,
        comparison_vmin=comparison_vmin,
        comparison_vmax=comparison_vmax,
    )
    for source in sources
]

def normalize_kpi_view(kpi_view):
    """
    Normalize a KPIView specification.
    """
    if isinstance(kpi_view, str):
        return {
            "name": kpi_view,
            "source": kpi_view,
        }

    if not isinstance(kpi_view, dict):
        raise TypeError(
            "A KPIView must be either a string or a dictionary."
        )

    view = deepcopy(kpi_view)

    has_source = "source" in view
    has_operation = "operation" in view

    if not has_source and not has_operation:
        raise ValueError(
            "KPIView must define either a 'source' or an 'operation'."
        )

    if "name" not in view:
        if has_source:
            view["name"] = view["source"]
        else:
            view["name"] = view["operation"]["type"]

    return view

def _collect_expression_sources(
    expression,
    sources,
):
    """Recursively collect raw sources from an expression."""

    if "source" in expression:

        source = expression["source"]

        if source not in sources:
            sources.append(source)

        return

    if "operation" in expression:

        for operand in expression["operation"]["operands"]:
            _collect_expression_sources(
                operand,
                sources,
            )

        return

    raise ValueError(
        "Invalid expression: expected 'source' or 'operation'."
    )

def get_kpi_view_sources(kpi_views):
    """
    Extract all raw sources required by KPIViews.
    """
    sources = []

    for kpi_view in kpi_views:

        expression = _build_kpi_expression(
            kpi_view
        )

        _collect_expression_sources(
            expression,
            sources,
        )

    return sources


def resolve_dimension_reference(reference, axis, dimension_labels):
    """
    Resolve a dimension reference into an integer index.

    References may be specified either by index or by label.
    """
    if isinstance(reference, (int, np.integer)):
        return int(reference)

    labels = np.asarray(dimension_labels[axis])

    matches = np.where(labels == reference)[0]

    if len(matches) == 0:
        raise ValueError(
            f"Unknown reference '{reference}' for dimension '{axis}'. "
            f"Available values: {labels.tolist()}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Reference '{reference}' is ambiguous for dimension '{axis}'."
        )

    return int(matches[0])

def resolve_kpi_view(kpi_view, dimension_labels):
    """
    Resolve user-facing KPI references into numerical indices.
    """
    view = deepcopy(kpi_view)

    comparison = view.get("comparison")

    if comparison is None:
        return view

    axis = comparison["axis"]

    comparison["reference"] = resolve_dimension_reference(
        reference=comparison["reference"],
        axis=axis,
        dimension_labels=dimension_labels,
    )

    return view

def expand_kpi_view(kpi_view):
    """
    Expand one KPIView into one or several executable ResultViews.
    """
    name = kpi_view["name"]

    style = kpi_view.get("style", {})
    absolute_style = style.get("absolute", {})
    comparison_style = style.get("comparison", {})

    base_expression = _build_kpi_expression(
        kpi_view
    )

    comparison = kpi_view.get("comparison")

    # --------------------------------------------------------------
    # No comparison
    # --------------------------------------------------------------
    if comparison is None:

        return [
            {
                "name": name,
                "expression": base_expression,
                "view_type": "absolute",
                "style": absolute_style,
            }
        ]

    # --------------------------------------------------------------
    # Reference comparison
    # --------------------------------------------------------------
    axis = comparison["axis"]
    reference = comparison["reference"]
    comparison_operation = comparison.get("operation","difference")
    comparison_parameters = comparison.get("parameters",{},)

    display = kpi_view.get(
        "display",
        {
            "reference": True,
            "comparison": True,
        },
    )

    reference_expression = _apply_selection_to_expression(
        base_expression,
        {
            axis: {
                "include": [reference],
            }
        },
    )

    target_expression = _apply_selection_to_expression(
        base_expression,
        {
            axis: {
                "exclude": [reference],
            }
        },
    )

    result_views = []

    # Reference KPI
    if display.get("reference", True):

        result_views.append(
            {
                "name": f"{name}_reference",
                "expression": reference_expression,
                "view_type": comparison_operation,
                "style": absolute_style,
            }
        )

    # KPI difference relative to reference
    if display.get("comparison", True):

        result_views.append(
            {
                "name": f"{name}_comparison",
                "expression": {
                    "operation": {
                        "type": comparison_operation,
                        "parameters": comparison_parameters,
                        "operands": [
                            target_expression,
                            reference_expression,
                        ],
                    }
                },
                "view_type": "comparison",
                "style": comparison_style,
            }
        )

    return result_views