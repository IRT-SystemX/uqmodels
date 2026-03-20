import numpy as np
from copy import deepcopy
from uqmodels.utils import apply_mask, apply_axis_masks


def _normalize_weights(weights):
    """Normalize weights so that they sum to 1."""
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total == 0:
        raise ValueError("Reduction weights sum to zero.")
    return weights / total


def _apply_weighted_axis_reduction(array, axis, reduc_filter, keepdims=False):
    """Apply a standard weighted reduction along a given axis.

    Args:
        array (np.ndarray): Input array.
        axis (int): Reduction axis.
        reduc_filter (array-like): Weights along `axis`.
        keepdims (bool): Whether to preserve reduced axis.

    Returns:
        np.ndarray: Reduced array.
    """
    weights = np.asarray(reduc_filter, dtype=float)

    if array.shape[axis] != len(weights):
        raise ValueError(
            f"Incompatible reduction filter length for axis {axis}: "
            f"expected {array.shape[axis]}, got {len(weights)}."
        )

    reshape = [1] * array.ndim
    reshape[axis] = len(weights)
    weighted = array * weights.reshape(reshape)

    return weighted.sum(axis=axis, keepdims=keepdims)


def _apply_rolled_weighted_axis_reduction(
    array,
    axis,
    reduc_filter,
    roll,
    roll_axis=0,
    keepdims=False,
):
    """Apply a weighted reduction with progressive rolling.

    The reduction is performed along `axis`.
    For each position j along `axis`, the weighted slice is rolled by
    `j * roll` along `roll_axis` before accumulation.

    Args:
        array (np.ndarray): Input array.
        axis (int): Reduction axis.
        reduc_filter (array-like): Weights along `axis`.
        roll (int): Roll step multiplier.
        roll_axis (int): Axis along which `np.roll` is applied.
        keepdims (bool): Whether to preserve reduced axis.

    Returns:
        np.ndarray: Reduced array.
    """
    weights = np.asarray(reduc_filter, dtype=float)

    if array.shape[axis] != len(weights):
        raise ValueError(
            f"Incompatible reduction filter length for axis {axis}: "
            f"expected {array.shape[axis]}, got {len(weights)}."
        )

    moved = np.moveaxis(array, axis, 0)
    reduced = moved[0] * weights[0]

    for j in range(1, moved.shape[0]):
        reduced += np.roll(moved[j] * weights[j], shift=j * roll, axis=roll_axis)

    if keepdims:
        reduced = np.expand_dims(reduced, axis=0)
        reduced = np.moveaxis(reduced, 0, axis)

    return reduced

def apply_axis_transformation(
    array,
    axis=None,
    axis_masks=None,
    mask=None,
    mask_mode="bool_array",
    reduc_filter=None,
    normalize_filter=True,
    roll=0,
    roll_axis=0,
    keepdims=False,
    copy=True,
):
    """Apply optional selections across axes and optional weighted reduction.

    Processing order is:
    1. optional multi-axis masking through `axis_masks`,
    2. optional single-axis masking through `mask` on `axis`,
    3. optional weighted reduction on `axis`.

    If no mask and no reduction filter are provided, the input is returned
    unchanged (or copied if `copy=True`).

    Args:
        array (np.ndarray): Input array.
        axis (int | None): Target axis for single-axis masking and reduction.
            Required if `mask` or `reduc_filter` is provided.
        axis_masks (dict[int, array-like] | None): Optional mapping axis -> mask
            for multi-axis selections.
        mask (array-like | None): Optional mask applied on `axis`.
        mask_mode (str): Mask interpretation mode.
            Supported values: "bool_array", "indices".
        reduc_filter (array-like | None): Optional reduction weights for `axis`.
        normalize_filter (bool): Whether to normalize reduction weights.
        roll (int): Optional roll step used during reduction.
            If 0, standard weighted reduction is used.
            If non-zero, rolled weighted reduction is used.
        roll_axis (int): Axis along which rolling is applied when `roll != 0`.
            This axis is interpreted after the reduction axis has been removed
            from the rolled internal representation.
        keepdims (bool): Whether to preserve the reduced axis.
        copy (bool): Whether to work on a copy of the input array.

    Returns:
        np.ndarray: Transformed array.
    """
    transformed = deepcopy(array) if copy else array

    if axis_masks is not None:
        transformed = apply_axis_masks(
            transformed,
            axis_masks=axis_masks,
            mask_mode=mask_mode,
        )

    if mask is not None:
        if axis is None:
            raise ValueError("`axis` must be provided when `mask` is used.")
        if axis_masks is not None and axis in axis_masks:
            raise ValueError(
                f"Axis {axis} is specified in both `axis_masks` and `mask`."
            )
        transformed = apply_mask(
            transformed,
            mask=mask,
            axis=axis,
            mode=mask_mode,
        )

    if reduc_filter is None:
        return transformed

    if axis is None:
        raise ValueError("`axis` must be provided when `reduc_filter` is used.")

    weights = np.asarray(reduc_filter, dtype=float)
    if normalize_filter:
        weights = _normalize_weights(weights)

    if roll:
        return _apply_rolled_weighted_axis_reduction(
            transformed,
            axis=axis,
            reduc_filter=weights,
            roll=roll,
            roll_axis=roll_axis,
            keepdims=keepdims,
        )

    return _apply_weighted_axis_reduction(
        transformed,
        axis=axis,
        reduc_filter=weights,
        keepdims=keepdims,
    )