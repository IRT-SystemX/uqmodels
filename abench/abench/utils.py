import numpy as np
import pandas as pd
import itertools
from collections.abc import Iterable
from typing import Callable, List, Optional, Union, Sequence
def concat(obj, axis: int = 0, newaxis: bool = False):
    """
    Combine homogeneous objects.

    Supports
    --------
    • numpy.ndarray
        newaxis=False → np.concatenate(args, axis)
        newaxis=True  → np.stack(args) then move the new axis to *axis*
    • pandas.DataFrame
        pd.concat(args, axis=axis)
    • list
        element-wise extend

    Parameters
    ----------
    obj     : objects to combine (all same type)
    axis     : int, target axis (concat axis or final position of stacked axis)
    newaxis  : bool, default False
               If True and objects are ndarrays, use stack instead of concat.

    Raises
    ------
    ValueError / TypeError on mismatch or unsupported types.
    """
    if not obj:
        raise ValueError("Need at least one argument")

    # ----- NumPy arrays -----------------------------------------------------
    if all(isinstance(a, np.ndarray) for a in obj):
        if newaxis:
            stacked = np.stack(obj, axis=0)      # add a leading axis
            return np.moveaxis(stacked, 0, axis)  # put it where requested
        return np.concatenate(obj, axis=axis)

    # ----- pandas DataFrames -----------------------------------------------
    if all(isinstance(a, pd.DataFrame) for a in obj):
        if newaxis:
            raise ValueError("newaxis applies only to ndarray")
        return pd.concat(obj, axis=axis)

    # ----- Python lists -----------------------------------------------------
    if all(isinstance(a, list) for a in obj):
        if newaxis:
            raise ValueError("newaxis applies only to ndarray")
        out = []
        for a in obj: out.extend(a)
        return out

    raise TypeError("Only ndarray, DataFrame or list are supported")

def apply_mask_along_dim(array: np.ndarray, mask: Union[np.ndarray, Sequence[int]], dim_mask: int) -> np.ndarray:
    """
    Selects elements along the specified dimension using a boolean mask or list of indices.

    Args:
        array (np.ndarray): Input N-dimensional array.
        mask (Union[np.ndarray, Sequence[int]]): Boolean mask or list of indices to apply along `dim_mask`.
        dim_mask (int): Dimension along which to apply the mask.

    Returns:
        np.ndarray: Masked array with reduced size along `dim_mask`.

    Raises:
        ValueError: if dimensions mismatch or mask is invalid.
    """
    if isinstance(mask, np.ndarray) and mask.dtype == bool:
        if mask.shape[0] != array.shape[dim_mask]:
            raise ValueError(f"Boolean mask length {mask.shape[0]} does not match dimension {dim_mask} size {array.shape[dim_mask]}")
        indexer = [slice(None)] * array.ndim
        indexer[dim_mask] = mask
        array = array[tuple(indexer)]
        if(mask.sum()==1):
            array = np.squeeze(array)
        return array

    elif isinstance(mask, (list, np.ndarray)):
        indexer = [slice(None)] * array.ndim
        indexer[dim_mask] = mask
        array = array[tuple(indexer)]
        if(len(mask)==1):
            array = np.squeeze(array)
        return array

    else:
        raise ValueError("mask must be either a boolean NumPy array or a list/array of indices")

def Extract_dict(dictionaire, list_keys):
    """Extract list of values of dictionaire from list_of_keys
    return None if keys isn't in dictionaire


    Args:
        dictionaire (dict): dictionary
        list_keys (str list): list of keys

    Returns:
        list_of_values or values: return list of values,
            if len(list)=1 return value
    """
    list_extract = []
    for keys in list_keys:
        if keys in list(dictionaire.keys()):
            list_extract.append(dictionaire[keys])
        else:
            list_extract.append(None)

    if len(list_extract) == 1:
        list_extract = list_extract[0]
    return list_extract

def apply_mask(list_or_array, mask):
    if(mask is None):
        return(list_or_array)
    if type(list_or_array) in [list,tuple]:
        return [i[mask] for i in list_or_array]
    else:
        return list_or_array[mask]

def stack_iterable_output(batch_iterable, stack_fn=np.concatenate):
    """
    Takes an iterable of tuples (e.g., (X, y, context)), and stacks the elements by position.
    
    Args:
        batch_iterable: iterable of tuples of equal length
        stack_fn: function to stack each field, e.g., np.stack, torch.stack, tf.stack

    Returns:
        tuple of stacked outputs (e.g., stacked_X, stacked_y, stacked_context)
    """
    if(type(batch_iterable) in [np.array,np.ndarray]):
        return(batch_iterable)
    elif (type(batch_iterable) is list) & (len(batch_iterable)==1):
        return(batch_iterable[0])
    else:
        items = list(batch_iterable)
        
        if not items:
            raise ValueError("Input iterable is empty")

        # Transpose list of tuples into tuple of lists
        stack_output = [list(i) for i in list(zip(*items))]
        for i in range(len(stack_output)):
            try:
                stack_output[i] = stack_fn(stack_output[i], axis=0)
            except:
                try:
                    stack_output[i] = stack_fn(stack_output[i])
                except:
                    None
        # Apply stacking function to each group
            if(len(stack_output)==1):
                stack_output = stack_output[0]
        return stack_output

def build_ctx_mask(context: np.ndarray, list_ctx_constraint):
    """
    Build a boolean mask selecting rows in *context* that satisfy **all**
    the constraints given in *list_ctx_constraint*.

    Parameters
    ----------
    context : ndarray of shape (N, D)
        The contextual feature matrix; rows = samples, columns = variables.
    list_ctx_constraint : list[tuple]
        Each tuple describes a constraint:

        • (ctx, val)
            Keep rows where context[:, ctx] == val.
            - If *val* is a scalar → equality test.
            - If *val* is a collection (list / tuple / set / ndarray) →
              membership test (vectorised with `np.isin`).

        • (ctx, min_, max_)
            Keep rows where *min_* < context[:, ctx] < *max_*.
            *min_* or *max_* may be None to ignore that bound.

    Returns
    -------
    ctx_flag : ndarray bool of shape (N,)
        Boolean mask: True for rows that satisfy every constraint.
    """
    meta_flag = []  # will store one boolean vector per elementary condition

    for constraint in list_ctx_constraint:
        if len(constraint) == 3:                     # (ctx, min_, max_)
            ctx, min_, max_ = constraint
            col = context[:, ctx]

            # lower bound
            if min_ is not None:
                meta_flag.append(col > min_)

            # upper bound
            if max_ is not None:
                meta_flag.append(col < max_)

        elif len(constraint) == 2:             
            ctx, val = constraint
            col = context[:, ctx]

            # Case 1 ─ val is an iterable of allowed values (but not a string/bytes)
            if isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
                meta_flag.append(np.isin(col, val))

            # Case 2 ─ val is a scalar → simple equality
            else:
                meta_flag.append(col == val)

        else:
            raise ValueError(
                "Each constraint must be a tuple of length 2 or 3."
            )

    # If there were no constraints, return an “all-True” mask
    if not meta_flag:
        return np.ones(context.shape[0], dtype=bool)

    # Combine all conditions with logical AND (every constraint must hold)
    ctx_flag = np.logical_and.reduce(meta_flag)
    return ctx_flag


def unique_with_nan(arr, return_inverse=True, nan_code=-1):
    """
    Encode an array (object or numeric) into integer codes while handling NaNs.

    Parameters
    ----------
    arr : np.ndarray
        1-D (or N-D) input array.
        • object dtype → may contain strings and np.nan  
        • numeric dtype → integers or floats (floats may include NaN)
    return_inverse : bool, default=True
        If True, returns *codes* aligned with *arr* (see Returns section).
    nan_code : int, default -1
        Integer code assigned to NaN positions (only relevant when NaNs exist).

    Returns
    -------
    uniques : np.ndarray
        Sorted unique non-NaN values.
    codes : np.ndarray
        Integer codes with the same shape as *arr*:
        • NaN → *nan_code*  
        • other values → index in *uniques* (0-based)

    Notes
    -----
    * For object arrays, NaNs are detected via ``x != x``.  
    * For numeric arrays, `np.isnan` is used when dtype is floating.  
    * Complexity dominated by `np.unique` on the non-NaN subset.
    """
    # ------------------------------------------------------------------
    # Fast path: NUMERIC ARRAY
    # ------------------------------------------------------------------
    if arr.dtype != object:
        if np.issubdtype(arr.dtype, np.floating):
            # Floats can contain NaNs → treat them explicitly
            nan_mask = np.isnan(arr)
            uniques, inv = np.unique(arr[~nan_mask], return_inverse=True)
            if not return_inverse:
                return uniques
            codes = np.full(arr.shape, nan_code, dtype=int)
            codes[~nan_mask] = inv
            return uniques, codes
        else:
            # Integer (or other numeric without NaN capability)
            uniques, inv = np.unique(arr, return_inverse=True)
            return (uniques, inv) if return_inverse else uniques

    # ------------------------------------------------------------------
    # OBJECT ARRAY: may mix strings and NaNs
    # ------------------------------------------------------------------
    nan_mask = arr != arr                       # True only for NaNs
    uniques, inv = np.unique(arr[~nan_mask], return_inverse=True)

    if not return_inverse:
        return uniques

    codes = np.full(arr.shape, nan_code, dtype=int)
    codes[~nan_mask] = inv
    return uniques, codes

def build_sets(context,
               list_ctx,
               list_ctx_name=None,
               encoder=unique_with_nan):
    """
    Parameters
    ----------
    context : np.ndarray (N, D) | pd.DataFrame (N, D)
    list_ctx : list[int | str | sequence]
        Chaque élément définit un « bloc » de colonnes :
        • entier ou str      → variable seule
        • séquence (list/tuple) d’indices ou noms → variables combinées
    list_ctx_name : list[str] | None
        • ndarray  → obligatoire (nom associé à chaque indice)
        • DataFrame → facultatif ; si None on prend context.columns
    encoder : callable
        Fonction (array) -> (uniques, codes).  Par défaut unique_with_nan.

    Returns
    -------
    list_sets  : list[list[np.ndarray]]
        Masques booléens pour chaque combinaison.
    list_pairs : list[list[list[tuple]]]
        Description de chaque combinaison :
        [('col_name', valeur_unique), …]
    """
    # ----------------------------------------------------------------
    # 0. Normalisation entrée + noms de colonnes
    # ----------------------------------------------------------------
    is_df = isinstance(context, pd.DataFrame)

    # Mapping indice/clé → nom lisible
    if is_df:
        col_names = list(context.columns) if list_ctx_name is None else list_ctx_name
        getter = lambda col: context[col].to_numpy()
    else:
        if list_ctx_name is None:
            raise ValueError("list_ctx_name must be provided when context is ndarray")
        col_names = list_ctx_name
        getter = lambda idx: context[:, idx]

    # ----------------------------------------------------------------
    # 1. Aucun contexte => un seul ensemble (tout True)
    # ----------------------------------------------------------------
    if list_ctx is None:
        return [[np.ones(len(context), dtype=bool)]], [[[]]]

    # ----------------------------------------------------------------
    # 2. Boucle sur les blocs demandés
    # ----------------------------------------------------------------
    list_sets, list_pairs = [], []

    for block in list_ctx:
        # --- assure que *cols* est un tuple de clés homogènes ---
        if isinstance(block, (list, tuple, np.ndarray)):
            cols = tuple(block)
        else:
            cols = (block,)

        # --- uniques & codes pour chaque colonne ---
        uniques_list, codes_list, names_list = [], [], []
        for col in cols:
            # pos = index si ndarray, sinon nom déjà str
            pos = col if is_df else int(col)
            series_arr = getter(col)
            uniq, codes = encoder(series_arr)
            uniques_list.append(uniq)
            codes_list.append(codes)
            names_list.append(col_names[pos])

        # --- produit cartésien des modalités + masques ---
        block_masks, block_infos = [], []
        for combo in itertools.product(*[range(len(u)) for u in uniques_list]):
            mask = np.ones(len(context), dtype=bool)
            for codes, target in zip(codes_list, combo):
                mask &= (codes == target)
            block_masks.append(mask)

            block_infos.append([
                (name, uniques_list[k][target])
                for k, name, target in zip(range(len(cols)), names_list, combo)
            ])

        list_sets.append(block_masks)
        list_pairs.append(block_infos)

    return list_sets, list_pairs
