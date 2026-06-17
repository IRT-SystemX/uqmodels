import numpy as np
import pandas as pd
from typing import Callable, List, Optional, Union, Any
from abench.utils import Extract_dict,build_ctx_mask,build_sets,apply_mask,apply_mask_along_dim
from abc import ABC, abstractmethod

Selector = Optional[Union[str, int, Callable[..., Any]]]

def select_from(obj: Any, selector: Selector, *extra_args: Any) -> Any:
    """
    Select data from an object using a compact selector.

    Parameters
    ----------
    obj:
        Source object.

    selector:
        Selection rule:
        - None: return object as-is
        - str: dictionary key
        - int: array column/index
        - callable: custom selection function

    *extra_args:
        Additional objects passed to callable selectors.

    Returns
    -------
    Any
        Selected object.
    """
    if selector is None:
        return obj

    if callable(selector):
        return selector(obj, *extra_args)

    if isinstance(selector, str):
        if not (isinstance(obj, dict) or isinstance(obj, pd.core.frame.DataFrame)):
            raise TypeError(
                f"String selector '{selector}' requires a dict object. "
                f"Got {type(obj)}."
            )
        if selector not in obj:
            raise KeyError(
                f"Selector key '{selector}' not found. "
                f"Available keys: {list(obj.keys())}"
            )
        return obj[selector]

    if isinstance(selector, int):
        arr = np.asarray(obj)

        if arr.ndim == 1:
            return arr[selector]

        return arr[:, selector]
    
    if isinstance(selector, tuple):
        if len(selector) != 2:
            raise ValueError("Tuple selector must be of the form (axis, index).")

        axis, index = selector
        return np.take(np.asarray(obj), indices=index, axis=axis)
  

    raise TypeError(
        f"Unsupported selector type: {type(selector)}. "
        "Expected None, str, int, or callable."
    )

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
    Generic metric wrapper compatible with raw arrays and structured component
    outputs.

    This wrapper supports:
    - target extraction from y or context;
    - output extraction from structured component outputs;
    - subset-based evaluation;
    - optional context filtering;
    - optional masking along a given dimension.

    Expected metric signature
    -------------------------
    metric(y_true, y_pred_or_score, context=None, **kwargs)
    """

    def __init__(
        self,
        metric: Callable,
        name: str = "Metric",
        target_selector: Selector = None,
        output_selector: Selector = None,
        mask_cfg: Optional[dict] = None,
        list_ctx_constraint: Optional[list] = None,
        dict_sets_config: Optional[dict] = None,
        reduce: bool = True,
        **kwarg,
    ):
        """
        Parameters
        ----------
        metric:
            Metric function.

        name:
            Metric name.

        target_selector:
            Rule used to extract the metric target from y or context.

            Supported values:
            - None: use y directly;
            - str: key searched first in y, then in context;
            - int: column/index searched first in y, then in context;
            - callable: function called as target_selector(y, context).

        output_selector:
            Rule used to extract the metric prediction, score, or reconstruction
            from output.

            Supported values:
            - None: use output directly;
            - str: key in output dict;
            - int: column/index in output array;
            - callable: function called as output_selector(output).

        mask_cfg:
            Configuration of masking parameter by data_source:
                ex : {'output':{'axis':1,'n':2},'target'
            Optional mask applied along dim_mask to target, output and context.

        dim_mask:
            Dimension on which mask is applied.

        list_ctx_constraint:
            Optional context constraints used to filter samples.

        dict_sets_config:
            Optional configuration used to build evaluation subsets from context.

        reduce:
            Whether to reduce array-like metric values.

        **kwarg:
            Additional keyword arguments passed to the metric function.
        """
        self.metric = metric
        self.name = name

        self.target_selector = target_selector
        self.output_selector = output_selector

        self.reduce = reduce
        self.mask_cfg = mask_cfg
        self.dict_sets_config = dict_sets_config
        self.list_ctx_constraint = list_ctx_constraint
        self.kwarg = kwarg


    def _build_default_sets_old(
        self,
        target_values: Any,
        context: Optional[Any],
    ) -> List[np.ndarray]:
        """
        Build default evaluation subsets.
        """
        if self.dict_sets_config is None:
            return [np.ones(len(target_values), dtype=bool)]

        if context is None:
            raise ValueError(
                "context is required when dict_sets_config is provided."
            )

        if isinstance(context, dict):
            raise TypeError(
                "dict_sets_config currently expects array-like context, "
                "but context is a dict."
            )

        list_keys = [
            "context_mask",
            "context_dim_mask",
            "context_variable_ids",
        ]

        context_mask, context_dim_mask, context_variable_ids = Extract_dict(
            self.dict_sets_config,
            list_keys=list_keys,
        )

        context_filtered = apply_mask_along_dim(
            context,
            context_mask,
            context_dim_mask,
        )

        list_list_mask, _ = build_sets(
            context_filtered,
            context_variable_ids,
            list_ctx_name=np.arange(context_filtered.shape[-1]),
        )

        return list_list_mask[0]

    @staticmethod
    def _reduce_metric_value(metric_val: Any) -> Any:
        """
        Reduce array-like metric outputs when possible.
        """
        if isinstance(metric_val, np.ndarray):
            return np.mean(metric_val)

        if isinstance(metric_val, (list, tuple)):
            try:
                return np.mean(np.asarray(metric_val))
            except Exception:
                return metric_val

        return metric_val

    @staticmethod
    def _is_array_like_context(context: Any) -> bool:
        """
        Return True if context can be masked sample-wise.
        """
        if context is None:
            return False

        if isinstance(context, dict):
            return False

        return hasattr(context, "__len__")
    
    """
    Generic metric wrapper compatible with array outputs and dict-like component outputs.

    The metric function is expected to follow:
        metric(y_true, y_pred_or_score, context=None, **kwargs)

    If the component output is a dict, the relevant field is selected through
    `output_key` or `output_getter`.
    """

    def _extract_data(
        self,
        y: Any,
        output: Any,
        context: Any = None,
        *,
        selector: Any,
    ) -> Any:
        """
        Extract data from y, output, or context using a unified selector.

        Selector format
        ---------------
        None:
            Return y.

        callable:
            selector(y, output, context)

        (source, sub_selector):
            source in {"y", "output", "context"}
            sub_selector is passed to select_from(...).
        """
        if callable(selector):
            return selector(y, output, context)

        if selector is None:
            return y

        if not isinstance(selector, tuple) or len(selector) != 2:
            raise ValueError(
                "selector must be None, callable, or a tuple "
                "of the form ('target' or 'y'|'output'|'context', sub_selector)."
            )

        source, sub_selector = selector

        if (source == "target") or (source == "y"):
            obj = y
        elif source == "output":
            obj = output
        elif source == "context":
            if context is None:
                raise ValueError("selector source is 'context', but context is None.")
            obj = context
        else:
            raise ValueError(
                f"Unknown source '{source}'. Expected 'y', 'output', or 'context'."
            )
        
        if(sub_selector is None):
            return(obj)
        else:
            return select_from(obj, sub_selector)


    def _build_default_sets(
        self,
        y: np.ndarray,
        context: Optional[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Build default evaluation subsets.
        """
        if self.dict_sets_config is None:
            return [np.ones(len(y), dtype=bool)]

        if context is None:
            raise ValueError("context is required when dict_sets_config is provided.")

        list_keys = [
            "context_mask",
            "context_dim_mask",
            "context_variable_ids",
        ]

        context_mask, context_dim_mask, context_variable_ids = Extract_dict(
            self.dict_sets_config,
            list_keys=list_keys,
        )

        context_filtered = apply_mask_along_dim(
            context,
            context_mask,
            context_dim_mask,
        )

        list_list_mask, _ = build_sets(
            context_filtered,
            context_variable_ids,
            list_ctx_name=np.arange(context_filtered.shape[-1]),
        )

        return list_list_mask[0]

    @staticmethod
    def _reduce_metric_value(metric_val: Any) -> Any:
        """
        Reduce array-like metric outputs when possible.
        """
        if isinstance(metric_val, np.ndarray):
            return np.mean(metric_val)

        if isinstance(metric_val, (list, tuple)):
            try:
                return np.mean(np.asarray(metric_val))
            except Exception:
                return metric_val

        return metric_val
    
    def compute(
        self,
        y: Any,
        output: Any,
        context: Optional[Any] = None,
        target_arg: Optional[dict] = None,
        sets: Optional[List[np.ndarray]] = None,
        **kwarg,
    ) -> Union[float, List[float]]:
        """
        Compute the metric over one or multiple subsets.

        Parameters
        ----------
        y:
            Ground truth, raw target object, or structured target object.

        output:
            Raw model output or structured component output.

        context:
            Optional contextual information.

        target_arg:
            Additional task-specific parameters passed to the metric.

        sets:
            Optional list of boolean masks defining evaluation subsets.

        **kwarg:
            Runtime keyword arguments passed to the metric function.

        Returns
        -------
        float or list
            Metric value or list of metric values over subsets.
        """
        target_arg = target_arg or {}

        if self.kwarg:
            kwarg = self.kwarg

        target_values = self._extract_data(y=y,
                                           output=output,
                                           context=context,
                                           selector=self.target_selector)
        if(self.output_selector is None):
            output_values = output
        else:
            output_values = self._extract_data(y=y,
                                               output=output,
                                               context=context,
                                               selector=self.output_selector)


        if self.mask_cfg is not None:
            if ('target' in self.mask_cfg.keys()) and (self.mask_cfg['target'] is not None) and (self._is_array_like_context(target_values)):
                target_values = apply_mask_along_dim(
                    target_values,
                    mask=self.mask_cfg['target']['mask'],
                    dim_mask=self.mask_cfg['target']['axis'],
                )
            if ('output' in self.mask_cfg.keys()) and (self.mask_cfg. self.mask_cfg['output'] is not None) and (self._is_array_like_context(output_values)):
                output_values = apply_mask_along_dim(
                    output_values,
                    mask=self.mask_cfg['output']['mask'],
                    dim_mask=self.mask_cfg['output']['axis'])

            if  ('context' in self.mask_cfg.keys()) and (self.mask_cfg['context'] is not None) and (self._is_array_like_context(context)):
                context = apply_mask_along_dim(
                    context,
                    mask=self.mask_cfg['context']['mask'],
                    dim_mask=self.mask_cfg['context']['axis'])
                
        
        if sets is None:
            sets = self._build_default_sets(target_values, context)

        ctx_mask = None
        if self.list_ctx_constraint is not None:
            ctx_mask = build_ctx_mask(context, self.list_ctx_constraint)

        perf_res = []

        for set_ in sets:
            if ctx_mask is not None:
                set_ = set_ & ctx_mask

            y_set = apply_mask(target_values, mask=set_)
            output_set = apply_mask(output_values, mask=set_)

            if self._is_array_like_context(context):
                context_set = apply_mask(context, mask=set_)
            else:
                context_set = context
                
            metric_val = self.metric(
                y_set,
                output_set,
                context=context_set,
                **target_arg,
                **kwarg,
            )

            if self.reduce:
                metric_val = self._reduce_metric_value(metric_val)

            perf_res.append(metric_val)

        if len(perf_res) == 1:
            return perf_res[0]

        return perf_res