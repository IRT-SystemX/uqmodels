import sys
from abc import ABC

import numpy as np
import scipy

import uqmodels.postprocessing.UQ_processing as UQ_proc

from .base_metrics import average_coverage

sys.path.insert(1, "/home/kevin.pasini/Workspace/n5_benchmark")
sys.path.insert(1, "/home/kevin.pasini/Workspace/n5_puncc")
sys.path.insert(1, "/home/kevin.pasini/Workspace/n5_uqmodels")

# Metrics wrapper


class Encapsulated_metrics(ABC):
    """Abstract Encapsulated Metrics class :
    Allow generic manipulation of metrics with output specifyied format"""

    def __init__(self):
        self.name = "metrics"

    def compute(self, y, output, sets, context, **kwarg):
        """Compute metrics

        Args:
            output (array): Model results
            y (array): Targets
            sets (array list): Sub-set (train,test)
            context (array): Additional information that may be used in metrics
        """


def build_ctx_mask(context, list_ctx_constraint):
    meta_flag = []
    for ctx, min_, max_ in list_ctx_constraint:
        if min_ is not None:
            meta_flag.append(context[:, ctx] > min_)
        if max_ is not None:
            meta_flag.append(context[:, ctx] < max_)
    ctx_flag = np.array(meta_flag).mean(axis=0) == 1
    return ctx_flag


class Generic_metric(Encapsulated_metrics):
    def __init__(
        self,
        ABmetric,
        name="Metric",
        mask=None,
        list_ctx_constraint=None,
        reduce=True,
        **kwarg
    ):
        """Metrics wrapper from function

        Args:
            ABmetric (function): function to wrap
            name (str, optional): name of the metric. Defaults to "Metric".
            mask (_type_, optional): mask for specify a focused dimension on multidimensional task. Defaults to None.
            list_ctx_constraint (_type_, optional): list of ctx_constraint link to context information.
                Defaults to None.
            reduce (bool, optional): if reduce multidimensional using mean. Defaults to True.
        """
        self.ABmetric = ABmetric
        self.mask = mask
        self.name = name
        self.reduce = reduce
        self.list_ctx_constraint = list_ctx_constraint
        self.kwarg = kwarg

    def compute(self, y, output, sets, context, **kwarg):
        perf_res = []
        if self.kwarg != dict():
            kwarg = self.kwarg

        if self.list_ctx_constraint is not None:
            ctx_mask = build_ctx_mask(context, self.list_ctx_constraint)
        for set_ in sets:
            if self.list_ctx_constraint is not None:
                set_ = set_ & ctx_mask

            perf_res.append(
                self.ABmetric(y, output, set_, self.mask, self.reduce, **kwarg)
            )
        return perf_res


# Metric reworked


def rmse(y, output, set_, mask, reduce, **kwarg):
    """Root mean square error metrics

    Args:
        y (np.array): Targets/observation
        output (np.array): modeling output : (y,UQ)
        set_ (list of mask): subset specification
        mask (bool array): mask the last dimension
        reduce (bool): apply reduction

    Returns:
        val: rmse values
    """
    pred = output[0]
    val = np.sqrt(np.power(pred[set_] - y[set_], 2).mean(axis=0))
    if mask:
        val = val[mask]

    if reduce:
        val = val.mean()
    return val


def UQ_sharpness(y, output, set_, mask, reduce, type_UQ="var", **kwarg):
    """Compute sharpness by transform UQ into 95% coverage PIs then compute size of PIs.

    Args:
        y (np.array): Targets/observation
        output (np.array): modeling output : (y,UQ)
        set_ (list of mask): subset specification
        mask (bool array): mask the last dimension
        reduce (bool): apply reduction
        type_UQ (str, optional): _description_. Defaults to "var".

    Returns:
        _type_: _description_
    """

    pred, UQ = output
    y_lower = UQ_proc.process_UQmeasure_to_quantile(
        UQ, type_UQ, pred, y, type_UQ_params=None, alpha=0.025
    )
    y_upper = UQ_proc.process_UQmeasure_to_quantile(
        UQ, type_UQ, pred, y, type_UQ_params=None, alpha=0.975
    )

    if mask is None:
        val = (y_upper[set_] - y_lower[set_]).mean(axis=0)
    else:
        val = (y_upper[set_] - y_lower[set_]).mean(axis=0)[mask]
    if reduce:
        val = val.mean()
    return val


def UQ_average_coverage(
    y, output, set_, mask, reduce, type_UQ="var", alpha=0.045, mode="UQ", **kwarg
):
    """Compute data coverage by transform UQ into (1-alpha)% coverage PIs

    Args:
        y (np.array): Targets/observation
        output (np.array): modeling output : (y,UQ)
        set_ (list of mask): subset specification
        mask (bool array): mask the last dimension
        reduce (bool): apply reduction
        type_UQ (str, optional): _description_. Defaults to "var".
        alpha (float, optional): _description_. Defaults to 0.045.
        mode (str, optional): _description_. Defaults to 'UQ'.

    Returns:
        _type_: _description_
    """

    if mode == "UQ":
        pred, UQ = output
        y_lower = UQ_proc.process_UQmeasure_to_quantile(
            UQ, type_UQ, pred, y, type_UQ_params=None, alpha=alpha / 2
        )
        y_upper = UQ_proc.process_UQmeasure_to_quantile(
            UQ, type_UQ, pred, y, type_UQ_params=None, alpha=1 - (alpha / 2)
        )

    elif mode == "KPI":
        y_lower, y_upper = output

    if mask is None:
        val = cov_metrics(y[set_], y_lower[set_], y_upper[set_])

    else:
        val = cov_metrics(y[set_], y_lower[set_], y_upper[set_])[mask]

    if reduce:
        val = val.mean()
    return val


def UQ_Gaussian_NLL(y, output, set_, mask, reduce, type_UQ="var", mode=None, **kwarg):
    """Compute Neg likelihood by transform UQ into sigma using guassian assumption

    Args:
        y (np.array): Targets/observation
        output (np.array): modeling output : (y,UQ)
        set_  (list of mask): subset specification
        mask (bool array): mask the last dimension
        reduce (bool): apply reduction
        type_UQ (str, optional): _description_. Defaults to "var".
        mode (_type_, optional): _description_. Defaults to None.

    Returns:
        val: _description_
    """

    pred, UQ = output
    if mode is None:
        sigma = UQ_proc.process_UQmeasure_to_sigma(UQ, type_UQ, pred, y=None)
    elif mode == "A":
        sigma = np.sqrt(UQ[0])
    elif mode == "E":
        sigma = np.sqrt(UQ[1])
    val = (
        -np.log(sigma) - 0.5 * np.log(2 * np.pi) - ((y - pred) ** 2 / (2 * (sigma**2)))
    )
    val[val < -7] = -7
    if mask is None:
        val = val[set_].mean(axis=0)
    else:
        val = val[set_].mean(axis=0)[mask]
    if reduce:
        val = val.mean()
    return val


def UQ_heteroscedasticity_ratio(
    y, output, set_, mask, reduce, type_UQ="var", mode=None, **kwarg
):
    """Compute ratio  by transform UQ into sigma using guassian assumption

    Args:
        y (np.array): Targets/observation
        output (np.array): modeling output : (y,UQ)
        set_  (list of mask): subset specification
        mask (bool array): mask the last dimension
        reduce (bool): apply reduction
        type_UQ (str, optional): _description_. Defaults to "var".
        mode (_type_, optional): _description_. Defaults to None.

    Returns:
        val: _description_
    """

    pred, UQ = output
    if mode is None:
        sigma = UQ_proc.process_UQmeasure_to_sigma(UQ, type_UQ, pred, y=None)
    elif mode == "A":
        sigma = np.sqrt(UQ[0])
    elif mode == "E":
        sigma = np.sqrt(UQ[1])
    # val = NLL_loss(y, pred, sigma)
    val = np.abs((y - pred)) / (sigma)
    val_temoin = np.abs((y - pred)) / (sigma.mean(axis=1))

    val[val < -6] = -6
    val_temoin[val_temoin < -6] = -6

    if mask is None:
        val = (val[set_] / val_temoin[set_]).mean(axis=0)
    else:
        val = (val[set_] / val_temoin[set_]).mean(axis=0)[mask]

    if reduce:
        val = val.mean()
    return val


def UQ_absolute_residu_score(
    y, output, set_, mask, reduce, type_UQ="var", mode=None, **kwarg
):
    """Compute absolute residu score from UQ,pred,y

    Args:
        y (np.array): Targets/observation
        output (np.array): modeling output : (y,UQ)
        set_ (list of mask): subset specification
        mask (bool array): mask the last dimension
        reduce (bool): apply reduction
        type_UQ (str, optional): _description_. Defaults to "var".
        mode (_type_, optional): _description_. Defaults to None.

    Returns:
        val: val
    """
    pred, UQ = output
    residu_score = UQ_proc.process_UQmeasure_to_residu(UQ, type_UQ, pred, y=y)
    # val = NLL_loss(y, pred, sigma)
    val = np.abs(residu_score)
    if mask is None:
        val = val[set_].mean(axis=0)
    else:
        val = val[set_].mean(axis=0)[mask]
    if reduce:
        val = val.mean()
    return val


def UQ_dEI(y, output, set_, mask, reduce, type_UQ="var_A&E", **kwarg):
    """disentangled epistemic indicator from pred,UQ that provide insight about model unreliability

    Args:
        y (np.array): Targets/observation
        output (np.array): modeling output : (y,UQ)
        set_ (list of mask): subset specification
        mask (bool array): mask the last dimension
        reduce (bool): apply reduction

    Returns:
        val: array of metrics values
    """
    if type_UQ != "var_A&E":
        print("erreur metric only compatible with type_UQ = var A&E")
    pred, (var_A, var_E) = output
    var_A, var_E = np.maximum(var_A, 0.00001), np.maximum(var_E, 0.00001)
    val = -0.5 * np.log(1 + (var_A[set_] / var_E[set_])).mean(axis=0)
    val = val
    if mask is not None:
        val = val[mask]
    if reduce:
        val = val.mean()
    return val


# Metric to rework


def calibrate_var(
    y, output, set_, mask, reduce, type_output="all", alpha=0.955, **kwarg
):
    per_rejection = 1 - alpha
    pred, (var_A, var_E) = output
    pred, (var_A, var_E) = output
    if type_output == "epistemic":
        pass
    elif type_output == "aleatoric":
        pass
    elif type_output == "all":
        var_A + var_E

    Empirical_coverage = average_coverage(
        y, output, np.arange(len(y)), mask, reduce, type_output
    )

    Empirical_coef = scipy.stats.norm.ppf(1 - ((1 - Empirical_coverage) / 2), 0, 1)
    True_coeff = scipy.stats.norm.ppf(1 - (per_rejection / 2), 0, 1)
    corr_ratio = np.power(True_coeff / Empirical_coef, 2)
    if type_output == "epistemic":
        new_output = pred, output[1], output[2] * corr_ratio
    elif type_output == "aleatoric":
        new_output = pred, output[1] * corr_ratio, output[2]
    elif type_output == "all":
        new_output = pred, output[1] * corr_ratio, output[2] * corr_ratio
    return new_output


def mae(y, output, set_, mask, reduce, **kwarg):
    pred, (var_A, var_E) = output

    if mask is None:
        val = np.abs(pred[set_] - y[set_]).mean(axis=0)
    else:
        val = np.abs(pred[set_] - y[set_]).mean(axis=0)[mask]

    if reduce:
        val = val.mean()
    return val


def cov_metrics(y, y_lower, y_upper, **kwarg):
    return ((y >= y_lower) & (y <= y_upper)).mean(axis=0)


def dEI(y, output, set_, mask, reduce, type_output="all", **kwarg):
    pred, (var_A, var_E) = output
    var_A, var_E = np.maximum(var_A, 0.00001), np.maximum(var_E, 0.00001)
    val = -0.5 * np.log(1 + (var_A[set_] / var_E[set_])).mean(axis=0)
    val = val
    if mask is not None:
        val = val[mask]
    if reduce:
        val = val.mean()
    return val


def anom_score(
    y, output, set_, mask, reduce, type_output="all", min_A=0.08, min_E=0.02, **kwarg
):
    pred, (var_A, var_E) = output
    if type_output == "epistemic":
        pass
    elif type_output == "aleatoric":
        pass
    elif type_output == "all":
        var_A + var_E
        ind_A = np.sqrt(var_A)

    ind_E = np.sqrt(var_E)
    ind_E[ind_E < min_E] = min_E
    ind_A[ind_A < min_A] = min_A
    anom_score = (np.abs(y - pred) + ind_E) / (
        2 * np.sqrt(np.power(ind_E, 2) + np.power(ind_A, 2))
    )

    if mask is None:
        val = (anom_score[set_]).mean(axis=0)
    else:
        val = (anom_score[set_]).mean(axis=0)[mask]
    if reduce:
        val = val.mean()
    return val


def confidence_score(
    y, output, set_, mask, reduce, type_output="all", min_A=0.08, min_E=0.02, **kwarg
):
    pred, var_A, var_E = output
    if type_output == "epistemic":
        pass
    elif type_output == "aleatoric":
        pass
    elif type_output == "all":
        var_A + var_E

    ind_A = np.sqrt(var_A)
    ind_E = np.sqrt(var_E)
    ind_E[ind_E < min_E] = min_E
    ind_A[ind_A < min_A] = min_A
    confidence_score = ind_E / np.power(ind_A, 0.75)

    if mask is None:
        val = (confidence_score[set_]).mean(axis=0)
    else:
        val = (confidence_score[set_]).mean(axis=0)[mask]
    if reduce:
        val = val.mean()
    return val


# Visualisation tools
