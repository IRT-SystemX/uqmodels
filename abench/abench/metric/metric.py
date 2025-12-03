import numpy as np
from typing import Callable, List, Optional, Union, Sequence
from abench.utils import Extract_dict,build_ctx_mask,build_sets,apply_mask,apply_mask_along_dim
from abc import ABC, abstractmethod

# Encapsulated metrics format :
class ABMetric(ABC):
    """Abstract Encapsulated Metrics class :
    Allow generic manipulation of metrics with output specifyied format"""

    def __init__(self):
        self.name = "metrics"

    def compute(self, y, output, context, target_arg, reduce, **kwarg):
        """Compute metrics

        Args:
            output (array): Model results
            y (array): Targets
            context (array): Additional information
            target_arg (array): Task specification
            reduce (bool): Whether to reduce the results (e.g., average).
        """
        pass

class ABMetricGeneric(ABMetric):
    """
    Generic metric wrapper for evaluation pipelines using a custom metric function.
    
    Allows flexible filtering with masks and contextual constraints, and supports
    computation across multiple data subsets.

    Attributes:
        metric (Callable): The function to compute the metric (e.g., accuracy, MAE).
        name (str): Name identifier for the metric.
        mask (Optional[np.ndarray]): Boolean mask to select specific values within a dimension.
        dim_mask (Optional[int]): Dimension index on which to apply the mask.
        list_ctx_constraint (Optional[List[dict]]): Contextual constraints to filter evaluation.
        target_arg (array): Task specification
        reduce (bool): Whether to reduce the results (e.g., average).
        kwarg (dict): Optional additional keyword arguments passed to the metric function.
    """

    def __init__(
        self,
        metric: Callable,
        name: str = "Metric",
        mask: Optional[np.ndarray] = None,
        dim_mask: Optional[int] = None,
        list_ctx_constraint: Optional[list] = None,
        dict_sets_config: dict = None,
        reduce: bool = True,
        **kwarg
    ):
        """
        Initializes the generic metric wrapper.

        Args:
            metric (Callable): A callable that computes the metric: `metric(y, y_pred) -> float`.
            name (str): Metric name (for logging or tracking).
            mask (np.ndarray, optional): Mask to apply on a given dimension of y/output.
            dim_mask (int, optional): Dimension index to apply the mask on.
            list_ctx_constraint (list, optional): List of filtering conditions to apply on context.
            reduce (bool): Whether to aggregate results (if applicable).
            **kwarg: Additional keyword arguments passed to the metric function.
        """
        self.metric = metric
        self.name = name
        self.reduce = reduce
        self.mask = mask
        self.dim_mask = dim_mask
        self.dict_sets_config = dict_sets_config
        self.list_ctx_constraint = list_ctx_constraint
        self.kwarg = kwarg

    def compute(
        self,
        y: np.ndarray,
        output: np.ndarray,
        context: Optional[np.ndarray],
        target_arg: Optional[dict],
        sets = None,
        **kwarg
    ) -> List[float]:
        """
        Computes the metric over one or multiple subsets of data, optionally filtered by context
        or masked along a specific dimension.

        Args:
            y (np.ndarray): Ground truth labels.
            output (np.ndarray): Model predictions.
            context (np.ndarray): Additional context (e.g., metadata).
            sets (List[np.ndarray], optional): List of boolean index arrays for subset selection.
            target_arg (array): Task specification
            **kwarg: Extra arguments (overridden by self.kwarg if set).

        Returns:
            List[float]: List of metric values computed on each subset.
        """
        perf_res = []
        if self.kwarg:
            kwarg = self.kwarg

        # Default to full set if no subsets provided
        if sets is None:
            if self.dict_sets_config is None:
                sets = [np.ones(len(y), dtype=bool)]
            else:

                list_keys = ['context_mask','context_dim_mask','context_variable_ids']
                context_mask,context_dim_mask,context_variable_ids = Extract_dict(self.dict_sets_config,list_keys=list_keys)
                
                context_filtered = apply_mask_along_dim(context, context_mask, context_dim_mask)

                list_list_mask, list_name = build_sets(context_filtered,context_variable_ids,list_ctx_name=np.arange(context_filtered.shape[-1]))
                sets = list_list_mask[0]

        if self.mask is not None:
            y = apply_mask_along_dim(y, self.mask, self.dim_mask)
            output = apply_mask_along_dim(output, self.mask, self.dim_mask)
            context = apply_mask_along_dim(context, self.mask, self.dim_mask)

        # Build context mask if constraints are defined
        if self.list_ctx_constraint is not None:
            ctx_mask = build_ctx_mask(context, self.list_ctx_constraint)

        for set_ in sets:
            if self.list_ctx_constraint is not None:
                set_ = set_ & ctx_mask

            y_set = apply_mask(y,mask=set_)
            output_set = apply_mask(output,mask=set_)
            context_set = apply_mask(context,mask=set_)
            metric_val = self.metric(y_set, output_set, context=context_set, **target_arg)
            if(self.reduce):
                if isinstance(metric_val, np.ndarray):
                    metric_val= np.mean(metric_val)
                    
                if isinstance(metric_val, (list, tuple)):
                    try:
                        metric_val = np.mean(np.array(metric_val))
                    except Exception:
                        metric_val = metric_val

            perf_res.append(metric_val)

        if(len(perf_res)==1):
            perf_res = perf_res[0]
        return perf_res
