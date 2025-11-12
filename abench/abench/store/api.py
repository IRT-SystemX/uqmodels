import numpy as np
from abench.utils import Extract_dict, concat
from abench.store.store import write, read
from typing import Any, Dict, List, Iterable, Union, Tuple, Any, Optional, Callable, Sequence, Mapping
from collections import defaultdict
import numpy as np
import pandas as pd

import pandas as pd
import numpy as np
from functools import reduce
import operator, itertools

Number = (int, float, np.integer, np.floating)

# Imports optionnels : si pandas / polars ne sont pas installés,
# la fonction continue de marcher pour les autres types.

def store_list_id(storing,components_name_list=None,trainset_name_list=None,testset_name_list=None):
    list_to_store = []
    if(components_name_list is not None):
        list_to_store.append((["src_data","components_name_list"], components_name_list))
    if(trainset_name_list is not None):
        list_to_store.append((["src_data","trainset_name_list"], trainset_name_list))
    if(testset_name_list is not None):
        list_to_store.append((["src_data","testset_name_list"], testset_name_list))
    for (keys, values) in list_to_store:
        write(storing, keys, values)

      
def get_list_id(storing):
    list_to_read = [["src_data","components_name_list"],["src_data","trainset_name_list"],["src_data","testset_name_list"]]
    components_name_list, trainset_name_list, testset_name_list = [read(storing, keys) for keys in list_to_read]
    if(components_name_list is None):
        components_name_list = []
    if(trainset_name_list is None):
        trainset_name_list = []
    if(testset_name_list is None):
        testset_name_list = []
    return(components_name_list, trainset_name_list, testset_name_list)


# Component
def store_component(storing,trainset_name,component_name,component):
    keys = [trainset_name,component_name,"component"]
    component.save(storing,keys)

def get_component(storing,trainset_name,component_name,component_class):
    keys = [trainset_name,component_name,"component"]
    component = component_class.load(storing,keys)
    return(component)

def get_subcomponent(storing,trainset_name,component_name,subcomponent_name,subcomponent_class):
    keys = [trainset_name,component_name,"component",subcomponent_name]
    component = subcomponent_class.load(storing,keys)
    return(component)

# store_ABDataExperiment
def store_ABDataExperiment(storing,ABDataExperiment):
    write(storing, ["src_data",ABDataExperiment.name], ABDataExperiment)

def get_ABDataExperiment(storing,name):
    return(read(storing, ["src_data",name]))


# ABdata
def store_ABloader(storing,set_name,ABloader):
    write(storing, ["src_data", set_name, "ABloader"], ABloader)

def get_ABloader(storing,set_name):
    ABloader = read(storing, ["src_data", set_name, "ABloader"])
    return(ABloader)

#Output
def store_output(storing,trainset_name,component_name,set_name,output,time_pred=0,time_fit=None):
    list_to_store = [([trainset_name,component_name,set_name,"dictperf"],{"time_fit": time_fit,"time_pred": time_pred}),
                        ([trainset_name,component_name, set_name, "output"], output)]

    for (keys, values) in list_to_store:
        write(storing, keys, values)

def get_output(storing,component_name,trainset_name,set_name):       
    try :
        output = read(storing, [trainset_name,component_name,set_name,'output'])
    
    except :
        output = None

    if output is None:
        print("There is no output of"+str(component_name)+" trained on "+str(trainset_name)+" for testset "+str(set_name))

    return(output)

#Dict_perf
def store_dictperf(storing,trainset_name=None,component_name=None,set_name=None,agg_name=None,dictperf={}):
    keys = []
    if agg_name is not None:
        keys.append(agg_name)
    if trainset_name is not None:
        keys.append(trainset_name)
    if component_name is not None:
        keys.append(component_name)
    if(set_name is not None):
        keys.append(set_name)
    keys.append("dictperf")
    write(storing, keys, dictperf)

def get_dictperf(storing,trainset_name=None,component_name=None,set_name=None,agg_name=None):
    keys = []
    if trainset_name is not None:
        keys.append(trainset_name)
    if component_name is not None:
        keys.append(component_name)
    if(set_name is not None):
        keys.append(set_name)
    keys.append("dictperf") 
    try:
        dictperf = read(storing, keys=keys)
    except:
        print('No dict')
        dictperf = dict()
    if(dictperf is None):
        dictperf = {}
    if(agg_name is not None):
        try:
            dictperf = dictperf[agg_name]
        except:
            raise ValueError(agg_name, 'not in dict_perf keys')
    return(dictperf)

# Perf_agg

def get_data(storing,set_name,keep_X=False):
    ABloader = get_ABloader(storing,set_name)
    List_X,List_y,List_context,List_metadata = [],[],[],[]
    for (X,y),context,metadata in ABloader:
        if(keep_X):
            List_X.append(X)
        List_y.append(y)
        List_context.append(context)
        List_metadata.append(metadata)
    
    if(keep_X):
        List_X = concat(List_X)
    List_y= concat(List_y)
    List_context = concat(List_context)
    return(List_X,List_y,List_context,List_metadata)


def get_data_and_output(storing,component_name,trainset_name,set_name,keep_X=False):       
    output = get_output(storing=storing,
                        component_name=component_name,
                        trainset_name=trainset_name,
                        set_name=set_name)

    X,y,context,metadata = get_data(storing=storing,
                                    set_name=set_name,
                                    keep_X=keep_X)

    return(X,y,output,context,metadata)


# Extract result from dict_perf



# -------------------------
# Auxiliary functions
# -------------------------

def _is_scalar(value: Any) -> bool:
    """Check if a value is a scalar (int or float)."""
    return isinstance(value, Number)

def _get_leaf(agg: dict, component: str, trainset: str, set_name: str, metric: str) -> Any:
    """Safely retrieve a leaf from the nested dictionary. Returns None if missing."""
    d = agg.get(component, {})
    d = d.get(trainset, {}) if isinstance(d, dict) else {}
    d = d.get(set_name, {}) if isinstance(d, dict) else {}
    return d.get(metric, None) if isinstance(d, dict) else None

def _infer_experiment_plan(agg: dict, components: List[str]) -> List[Tuple[str, List[str]]]:
    """
    Infer all trainset -> sets mapping from the data.
    Returns a list of tuples (trainset, sorted_list_of_sets).
    """
    ts_map: Dict[str, set] = {}
    for c in components:
        for ts, sets_dict in (agg.get(c, {}) or {}).items():
            if isinstance(sets_dict, dict):
                ts_map.setdefault(ts, set()).update(sets_dict.keys())
    return [(ts, sorted(list(s))) for ts, s in ts_map.items()]

def _infer_metrics(agg: dict, components: List[str]) -> List[str]:
    """Infer metric names from the first available leaf."""
    for c in components:
        for ts, sets_dict in (agg.get(c, {}) or {}).items():
            if isinstance(sets_dict, dict):
                for s, m_dict in sets_dict.items():
                    if isinstance(m_dict, dict) and m_dict:
                        return list(m_dict.keys())
    return []

def _build_row_structure(
    experiment_plan: Optional[Dict[str, List[str]]], 
    agg: dict, 
    components: List[str]
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Build row keys and labels.
    - If experiment_plan is None, infer them from data.
    - experiment_plan must be a dict: {trainset: [set1, set2, ...]}.
    - Returns (row_keys, row_labels)
    """
    if experiment_plan is None:
        ts_pairs = _infer_experiment_plan(agg, components)
    else:
        # Convert directly from dictionary to list of tuples
        ts_pairs = [(ts, list(sets)) for ts, sets in experiment_plan.items()]

    # Build row_keys
    row_keys = []
    for ts, sets in ts_pairs:          # Loop over each (trainset, list of sets)
        for s in sets:                  # Loop over each validation set
            row_keys.append((ts, s))    # Append tuple (trainset, set)

    # Build row_labels
    row_labels = [f"{ts} | {s}" for ts, s in row_keys]

    return row_keys, row_labels

# -------------------------
# Main function
# -------------------------

def extract_benchmark_tables(
    dict_perf: Dict[str, Any],
    experiment_plan: Optional[Dict[str, List[str]]] = None,
    list_components_name: Optional[Iterable[str]] = None,
    metrics: Optional[Union[str, Iterable[str]]] = None,
    agg_name: str = 'no-agg',
) -> Dict[str, Union[pd.DataFrame, Dict[str, List[Any]]]]:
    """
    Extract metric tables from a 5-level nested dict:
      dict_perf[agg_name][component][trainset][set_name][metric] -> leaf (scalar or array-like).

    Parameters
    ----------
    dict_perf : dict
        Nested dictionary holding the benchmark results.
    agg_name : str
        Aggregation name (level-1 key).
    list_components_name : iterable[str] | None
        Models to include (level-2). None -> all under agg_name.
    experiment_plan : dict[str, list[str]] | None
        Mapping {trainset: [set1, set2, ...]}.
        None -> inferred automatically from data.
    metrics : str | iterable[str] | None
        Metric(s) to extract (level-5). None -> inferred automatically from data.

    Returns
    -------
    dict[str, DataFrame | dict[str, list[Any]]]
        - If metric leaves are purely scalar, value is a DataFrame:
            rows = "trainset | set_name", columns = list_components_name.
        - Otherwise, value is a dict {component: [values]}.
    """
    if agg_name not in dict_perf:
        raise KeyError(f"Aggregation '{agg_name}' not found in dict_perf.")
    agg = dict_perf[agg_name]

    # Determine list_components_name
    all_list_components_name = list(agg.keys())
    list_components_name = (
        all_list_components_name 
        if list_components_name is None 
        else [c for c in list_components_name if c in agg]
    )
    if not list_components_name:
        raise ValueError("No valid list_components_name to extract.")

    # Build row structure
    row_keys, row_labels = _build_row_structure(experiment_plan, agg, list_components_name)
    if not row_keys:
        raise ValueError("No (trainset, set) pairs to extract.")

    # Determine metrics
    if metrics is None:
        metrics = _infer_metrics(agg, list_components_name)
    elif isinstance(metrics, str):
        metrics = [metrics]
    else:
        metrics = list(metrics)
    if not metrics:
        raise ValueError("No metrics found or provided.")

    # Main extraction
    results = {}
    for metric in metrics:
        # Collect values for each row and component
        values = [
            [_get_leaf(agg, c, ts, s, metric) for c in list_components_name]
            for ts, s in row_keys
        ]

        # Determine output type
        any_scalar = any(v is not None and _is_scalar(v) for row in values for v in row)
        any_non_scalar = any(v is not None and not _is_scalar(v) for row in values for v in row)

        if any_scalar and not any_non_scalar:
            # Build DataFrame
            df = pd.DataFrame(
                [[float(v) if _is_scalar(v) else np.nan for v in row] for row in values],
                index=row_labels,
                columns=list_components_name
            )
            results[metric] = df
        else:
            # Build dict of lists
            results[metric] = {
                comp: [row[i] for row in values] for i, comp in enumerate(list_components_name)
            }
    return results

##################
# Filter experiment plan
##################

def get_experiment_plan(
    ABData_Experiments,
    filter_train_ids: Iterable[int] = None,
    filter_train_patterns: Iterable[str] = None,
    filter_test_ids: Iterable[int] = None,
    filter_test_patterns: Iterable[str] = None,
) -> Dict[str, Any]:
    """
    Filter a 2-level experiment plan of the form:
        { train_name: [test1, test2, ...] }  OR  { train_name: { test_name: payload, ... } }

    Filters:
      - *ids* keep items by index (supports negative indices; raises on out-of-range).
      - *patterns* keep items whose (string) name contains at least one pattern (case-insensitive).
    The function returns a new dict and never mutates the input plan.
    """

    # --- Small, fast helpers --------------------------------------------------

    def _gather(seq: List[Any], ids: Iterable[int]) -> List[Any]:
        """Normalize negative indices, bounds-check, then gather in the given order."""
        n = len(seq)
        idx = [i if i >= 0 else n + i for i in ids]
        if any(i < 0 or i >= n for i in idx):
            raise IndexError(f"indices {ids} out of range for length {n}")
        return [seq[i] for i in idx]

    def _apply_filters(
        values: Iterable[Any],
        ids: Iterable[int] | None,
        patterns: Iterable[str] | None,
        key: Callable[[Any], str] = lambda x: x,
    ) -> List[Any]:
        """
        Apply index selection first, then case-insensitive substring matching.
        `key` extracts the string to match on (e.g., the dict key or str(value)).
        """
        vals = list(values)
        if ids is not None:
            vals = _gather(vals, ids)
        if patterns:
            P = [p.casefold() for p in patterns]
            vals = [v for v in vals if any(p in key(v).casefold() for p in P)]
        return vals

    # --- Retrieve and validate the plan --------------------------------------

    plan = ABData_Experiments.get_experiment_plan()
    if not isinstance(plan, Mapping):
        raise TypeError("get_experiment_plan() must return a Mapping (dict-like).")

    # Select train names (order-preserving)
    train_names = _apply_filters(plan.keys(), filter_train_ids, filter_train_patterns, key=str)

    # --- Build filtered output ------------------------------------------------
    out: Dict[str, Any] = {}
    for tr in train_names:
        tests = plan[tr]

        if isinstance(tests, Mapping):
            # Mapping case: filter by test *keys* and rebuild a sub-dict
            test_names = _apply_filters(tests.keys(), filter_test_ids, filter_test_patterns, key=str)
            out[tr] = {k: tests[k] for k in test_names}

        elif isinstance(tests, Sequence) and not isinstance(tests, (str, bytes)):
            # Sequence case: filter by item indices and by stringified items
            seq = _apply_filters(list(tests), filter_test_ids, filter_test_patterns, key=lambda x: str(x))
            out[tr] = seq

        else:
            # Fallback: unsupported test container → copy as-is
            out[tr] = tests

    return out



# Extraction of results

def unwrap_singleton(value):
    """
    Return the sole element if *value* is a 1-element tuple,
    otherwise return *value* unchanged.
    """
    if isinstance(value, tuple) and len(value) == 1:
        return value[0]
    return value


def extract_tabular_data(
    nested_dict: Dict[Any, Any],
    row_levels: Iterable[int],
    col_levels: Iterable[int],
    filter_keys: Optional[Dict[int, Iterable[Any]]] = None,
    mean_key: Optional[Any] = None,
    std_key: Optional[Any] = None,
    round_val=3,
    *,
    return_filter_trace: bool = False):
    """
    Flatten a deeply-nested dictionary into a 2-D mapping and keep a full
    “filter trace” of which keys were encountered, accepted, or rejected.

    Parameters
    ----------
    nested_dict : dict
        The hierarchical structure to flatten.
    row_levels / col_levels : Iterable[int]
        Zero-based depth indices that will become row / column keys.
    mean_key: hashable, optional
        mean_key just before leaf values.
    std_key : hashable, optional
        std_key just before leaf values.
    filter_keys : dict[int, Iterable[hashable]], optional
        At each listed depth, **only** the keys in the iterable are followed.
    return_filter_trace : bool, keyword-only
        If True, return a tuple ``(result, trace)`` instead of just ``result``.

    Returns
    -------
    dict[(tuple) -> dict[(tuple) -> Any]]
        The 2-D mapping suitable for e.g. ``pandas.DataFrame(result)``.  
    or
    (result, filter_trace)  if *return_filter_trace* is True.

    filter_trace format
    -------------------
    {
        depth : {
            'encountered': [...],
            'accepted'  : [...],
            'rejected'  : [...]
        },
        ...
    }

    Raises
    ------
    ValueError
        * If ``row_levels`` or ``col_levels`` reach deeper than any path.
        * If ``filter_keys`` discards every branch — the exception message
          contains a compact, machine-readable trace for debugging.
    """

    # --- Initialise containers ------------------------------------------
    result: Dict[Tuple[Any, ...], Dict[Tuple[Any, ...], Any]] = defaultdict(dict)

    # trace[depth]['encountered' | 'accepted' | 'rejected'] = set()
    filter_trace: Dict[int, Dict[str, set]] = defaultdict(
        lambda: {"encountered": set(), "accepted": set(), "rejected": set()}
    )

    # --- Recursive DFS ---------------------------------------------------
    def recurse(current: Any, path: List[Any] = []) -> None:
        """Depth-first traversal that populates *result* and *filter_trace*."""
        # ---------- Leaf node --------------------------------------------
        if isinstance(current, dict) and mean_key in current.keys():
            deepest_requested = max(
                max(row_levels, default=-1),
                max(col_levels, default=-1)
            )
            if len(path) <= deepest_requested:
                # Path is too short for the requested row/col indices
                return

            row_key = unwrap_singleton(tuple(path[i] for i in row_levels))
            col_key = unwrap_singleton(tuple(path[i] for i in col_levels))
            if mean_key is not None and isinstance(current, dict):
                mean = np.round(current.get(mean_key),round_val)
                if std_key is not None and isinstance(current, dict):
                    std = np.round(current.get(std_key),round_val)
                current = mean,std
                
            result[row_key][col_key] = np.round(current,round_val)

        
        if not isinstance(current, dict):
            deepest_requested = max(
                max(row_levels, default=-1),
                max(col_levels, default=-1)
            )
            if len(path) <= deepest_requested:
                # Path is too short for the requested row/col indices
                return

            row_key = unwrap_singleton(tuple(path[i] for i in row_levels))
            col_key = unwrap_singleton(tuple(path[i] for i in col_levels))
                
            result[row_key][col_key] = np.round(current,round_val)
            return

        # ---------- Branch node ------------------------------------------
        depth = len(path)
        for key, child in current.items():
            # Record every key we see, even if no filter applies here
            filter_trace[depth]["encountered"].add(key)

            # Apply depth-specific filter (if any)
            if filter_keys and depth in filter_keys:
                if key not in filter_keys[depth]:
                    filter_trace[depth]["rejected"].add(key)
                    continue
                filter_trace[depth]["accepted"].add(key)

            recurse(child, path + [key])

    # Start the traversal
    recurse(nested_dict)

    # --- Failure handling -----------------------------------------------
    if not result:
        if filter_keys:
            # Convert sets → sorted lists for readability
            trace_summary = {
                lvl: {k: sorted(v) for k, v in log.items() if v}
                for lvl, log in filter_trace.items()
            }
            raise ValueError(
                "No data extracted: the provided filters ("+str(filter_keys)+") eliminated every "
                "possible branch.\nFilter trace:\n" + repr(trace_summary)
            )
        raise ValueError(
            "No data extracted at all. Check your row/column levels or the "
            "depth of the input structure."
        )

    # Convert inner defaultdicts to plain dicts
    clean_result = {row: dict(cols) for row, cols in result.items()}

    # Return result alone or with the trace
    if return_filter_trace:
        trace_as_lists = {
            lvl: {k: sorted(v) for k, v in log.items()}
            for lvl, log in filter_trace.items()
        }
        return clean_result, trace_as_lists

    return clean_result
