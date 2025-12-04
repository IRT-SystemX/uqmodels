"""
Losses and metrics for Uncertainty Quantification (UQ) and time-series forecasting.

This module provides losses/metrics tailored to:
- deterministic models (MSE),
- probabilistic models with Gaussian outputs (mean + log-variance),
- evidential deep learning (EDL) parameterization (gamma, v, alpha, beta),

with support for 2D, 3D, or 4D tensors typically encountered in time-series:
- 2D: (batch, features) or (batch, horizon/features combined)
- 3D: (batch, time, features)           -> e.g., forecasting t+1 uses the first step
- 4D: (batch, segments, time, features)  -> e.g., uses the last future segment/time = t+1

Notes
-----
- All builders are registered via `@tf.keras.utils.register_keras_serializable`
  so they can be used by name in `model.save()` / `keras.models.load_model()`.
- When `metric=True`, some builders return a *metric-like* value (e.g., RMSE or
  coverage proportion) instead of a training loss. Use those in `metrics=[...]`,
  not as the primary `loss`.
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import callbacks  # kept if you reference callbacks elsewhere


# --------------------------------------------------------------------------- #
# Deterministic MSE (builder)
# --------------------------------------------------------------------------- #

def make_loss_and_metrics(task: str, alpha: float = 0.95):
    """
    Sélectionne loss/metrics par défaut selon la tâche.
    - deterministic : MSE + RMSE@t+1 (metric=True)
    - gaussian     : Gaussian NLL-like + coverage ±2σ
    - edl          : EDL loss + coverage ±2σ
    """
    task = task.lower()
    if task == "deterministic":
        loss = build_MSE_loss(metric=False)
        metrics = [build_MSE_loss(metric=True)]
    elif task == "gaussian":
        loss = build_BNN_loss(alpha=alpha, metric=False, type_output="MC_Dropout")
        metrics = [build_BNN_loss(metric=True, type_output="MC_Dropout")]
    elif task == "edl":
        loss = BNN_loss_edl
        metrics = [BNN_metric_edl]
    else:
        raise ValueError(f"Unknown task '{task}'. Use: deterministic, gaussian, edl.")
    return loss, metrics

@tf.keras.utils.register_keras_serializable(
    package="UQModels_loss", name="build_MSE_loss"
)
def build_MSE_loss(split=2, metric=False, reduce=False, var=1):
    """
    Build a Mean Squared Error (MSE) loss or metric for time-series.

    Parameters
    ----------
    split : int, default=2
        If `true.shape[-1] != pred.shape[-1]`, `pred` is split along the last axis
        into `split` chunks and the first chunk is treated as the mean `mu`.
        This is convenient when predictions concatenate [mu, extra_params].
    metric : bool, default=False
        If True, compute a *metric-like* scalar:
          - For 3D inputs (B, T, C): uses the first prediction step (t+1 → index 0).
          - For 4D inputs (B, S, T, C): uses the last future segment and t+1.
          - Otherwise (2D): uses the full tensor.
        Returns RMSE scaled by `sqrt(var)`.
        If False, returns elementwise MSE reduced by `mean(axis=0)`.
    var : float, default=1
        A scaling variance. Only used when `metric=True` to scale RMSE by `sqrt(var)`.

    Returns
    -------
    callable
        A function `MSE_loss(true, pred)` suitable for `loss` or `metrics`
        depending on `metric`.

    Notes
    -----
    - 3D/4D selection logic is meant for "single-step" evaluation commonly used
      in time-series (e.g., evaluating t+1).
    """
    def MSE_loss(true, pred):
        # If shapes match, assume `pred` already contains mu only.
        if true.shape[-1] == pred.shape[-1]:
            mu = pred
        else:
            # Otherwise, split last axis and take the first chunk as mu.
            pred_split = tf.split(pred, split, axis=-1)
            mu = pred_split[0]

        if metric:  # Only consider a single-step prediction (t+1) for evaluation
            if len(mu.shape) == 3:
                # (B, T, C) -> use first step
                loss = K.pow(true[:, 0] - mu[:, 0], 2)
            elif len(mu.shape) > 3:
                # (B, S, T, C) -> use last future segment and step t+1
                loss = K.pow(true[:, -1, 0] - mu[:, -1, 0], 2)
            else:
                # 2D or other fallback
                loss = K.pow(true - mu, 2)

            reduce_loss = K.sqrt(K.mean(loss)) * np.sqrt(var)  # RMSE * sqrt(var)
        else:
            # Standard MSE reduced over batch/time dims (keeps last axis by default)
            loss = K.pow(true - mu, 2)
            reduce_loss = K.mean(loss, axis=0)

        if(reduce):
            reduce_loss = K.mean(reduce_loss)
        return reduce_loss

    return MSE_loss


# --------------------------------------------------------------------------- #
# Categorical Cross-Entropy (builder, unconventional form)
# --------------------------------------------------------------------------- #

@tf.keras.utils.register_keras_serializable(
    package="UQModels_loss", name="build_CCE_loss"
)
def build_CCE_loss():
    """
    Build a simple categorical cross-entropy–like loss.

    Returns
    -------
    callable
        A function `CCE_loss(true, pred)` that computes:
            loss = -mean(log(true * pred))
            reduce_loss = sum(loss, axis=0)

    Notes
    -----
    - This form assumes that `true` is one-hot (or soft labels) and `pred` are
      probabilities in (0,1]. It is not the standard `tf.keras.losses.CategoricalCrossentropy`
      call signature, but a minimal custom variant.
    """
    def CCE_loss(true, pred):
        loss = -K.mean(K.log(true * pred))
        reduce_loss = K.sum(loss, axis=0)
        return reduce_loss

    return CCE_loss


# --------------------------------------------------------------------------- #
# Bayesian / Probabilistic losses & metrics (Gaussian, EDL)
# --------------------------------------------------------------------------- #

@tf.keras.utils.register_keras_serializable(
    package="UQModels_loss", name="BNN_loss_gaussian"
)
def BNN_loss_gaussian(true, pred, alpha=0.95):
    """
    Gaussian negative log-likelihood–like loss for probabilistic regression.

    Assumes `pred` concatenates [mu, logvar] along the last axis.

    Parameters
    ----------
    true : Tensor
        Ground truth targets.
    pred : Tensor
        Model outputs with last axis concatenating [mu, logvar].
    alpha : float, default=0.95
        Weight for the log-variance term.

    Returns
    -------
    Tensor
        Mean loss reduced along batch/time dimensions (keeps last axis).

    Formula
    -------
    L = (true - mu)^2 * exp(-logvar) + alpha * logvar
    """
    # Split prediction into mean and log-variance
    mu, logvar = tf.split(pred, 2, axis=-1)
    loss = K.pow(true - mu, 2) * K.exp(-logvar) + alpha * logvar
    reduce_loss = K.mean(loss, axis=0)
    return reduce_loss


@tf.keras.utils.register_keras_serializable(
    package="UQModels_loss", name="BNN_loss_edl"
)
def BNN_loss_edl(true, pred, alpha=0.95):
    """
    Evidential Deep Learning (EDL) regression loss in Gaussian form.

    Assumes `pred` concatenates [gamma, v, alpha_edl, beta] along the last axis.
    The implied Gaussian parameters are:
        mu = gamma
        logvar = log( beta / (alpha_edl - 1) )

    Parameters
    ----------
    true : Tensor
        Ground truth targets.
    pred : Tensor
        Model outputs with last axis [gamma, v, alpha_edl, beta].
    alpha : float, default=0.95
        Weight for the log-variance term.

    Returns
    -------
    Tensor
        Mean loss reduced along batch/time dimensions (keeps last axis).
    """
    gamma, vu, alpha_edl, beta = tf.split(pred, 4, axis=-1)
    mu = gamma
    logvar = K.log(beta / (alpha_edl - 1))
    loss = K.pow(true - mu, 2) * K.exp(-logvar) + alpha * logvar
    reduce_loss = K.mean(loss, axis=0)
    return reduce_loss


@tf.keras.utils.register_keras_serializable(
    package="UQModels_loss", name="BNN_metric_gaussian"
)
def BNN_metric_gaussian(true, pred):
    """
    Coverage-style metric for Gaussian predictions.

    Assumes `pred` concatenates [mu, logvar]. Computes the fraction of targets
    that fall within ±2 * sqrt(var) around mu (≈95% interval under Gaussian).

    For time-series:
      - 3D (B, T, C): evaluates the first step (t+1).
      - >3D (B, S, T, C): evaluates the last future segment and t+1.
      - Otherwise: uses full tensor.

    Returns
    -------
    Tensor (scalar)
        Mean of boolean coverage over the batch/time axes.
    """
    mu, logvar = tf.split(pred, 2, axis=-1)

    if len(mu.shape) == 3:
        mu_ = mu[:, 0]
        logvar_ = logvar[:, 0]
        true_ = true[:, 0]
    elif len(mu.shape) > 3:
        mu_ = mu[:, -1, 0]
        logvar_ = logvar[:, -1, 0]
        true_ = true[:, -1, 0]
    else:
        mu_ = mu
        logvar_ = logvar
        true_ = true

    # Coverage: |true - mu| <= 2 * std
    loss = K.greater(2 * K.sqrt(K.exp(logvar_)), K.abs(true_ - mu_))
    reduce_loss = K.mean(loss)
    return reduce_loss


@tf.keras.utils.register_keras_serializable(
    package="UQModels_loss", name="BNN_metric_edl"
)
def BNN_metric_edl(true, pred):
    """
    Coverage-style metric for EDL predictions.

    Assumes `pred` concatenates [gamma, v, alpha_edl, beta].
    Converts to Gaussian parameters via:
        mu = gamma
        logvar = log( beta / (alpha_edl - 1) )

    For time-series:
      - 3D (B, T, C): evaluates the first step (t+1).
      - >3D (B, S, T, C): evaluates the last future segment and t+1.
      - Otherwise: uses full tensor.

    Returns
    -------
    Tensor (scalar)
        Mean boolean coverage (fraction within ±2σ).
    """
    gamma, vu, alpha_edl, beta = tf.split(pred, 4, axis=-1)
    mu = gamma
    logvar = K.log(beta / (alpha_edl - 1))

    if len(mu.shape) == 3:
        mu_ = mu[:, 0]
        logvar_ = logvar[:, 0]
        true_ = true[:, 0]
    elif len(mu.shape) > 3:
        mu_ = mu[:, -1, 0]
        logvar_ = logvar[:, -1, 0]
        true_ = true[:, -1, 0]
    else:
        mu_ = mu
        logvar_ = logvar
        true_ = true

    loss = K.greater(2 * K.sqrt(K.exp(logvar_)), K.abs(true_ - mu_))
    reduce_loss = K.mean(loss)
    return reduce_loss


# --------------------------------------------------------------------------- #
# Unified builder dispatching Gaussian / EDL or deterministic (with metric)
# --------------------------------------------------------------------------- #

@tf.keras.utils.register_keras_serializable(
    package="UQModels_loss", name="build_BNN_loss"
)
def build_BNN_loss(alpha=0.95, metric=False, type_output="MC_Dropout"):
    """
    Build a probabilistic loss or metric depending on the output parameterization.

    Parameters
    ----------
    alpha : float, default=0.95
        Weight of the log-variance term in the Gaussian-style loss.
    metric : bool, default=False
        If True, returns a coverage-style *metric* (fraction within ±2σ).
        If False, returns a Gaussian NLL-like *loss*.
    type_output : {"MC_Dropout", "Deep_ensemble", "EDL"}, default="MC_Dropout"
        - "MC_Dropout" / "Deep_ensemble": `pred` is [mu, logvar]
        - "EDL": `pred` is [gamma, v, alpha_edl, beta] (converted to [mu, logvar])

    Returns
    -------
    callable
        A function `BNN_loss(true, pred)` acting as loss or metric depending on `metric`.

    Notes
    -----
    - For time-series tensors (3D/4D), when `metric=True`, the function evaluates
      a single "t+1" step (first step for 3D, last future segment + t+1 for 4D).
    - If `type_output` is unrecognized, this falls back to treating `pred` as mu only.
    """
    def BNN_loss(true, pred):
        # Decode output parameterization
        if (type_output == "MC_Dropout") or (type_output == "Deep_ensemble"):
            mu, logvar = tf.split(pred, 2, axis=-1)
        elif type_output == "EDL":
            gamma, vu, alpha_edl, beta = tf.split(pred, 4, axis=-1)
            mu = gamma
            logvar = K.log(beta / (alpha_edl - 1))
        else:
            # Fallback: treat pred as mu only
            mu = pred
            _ = np.array([[1], [1]])  # no-op, kept to preserve original behavior

        if metric:
            # Coverage metric within ±2σ on a single-step selection
            if len(mu.shape) == 3:
                mu_ = mu[:, 0]
                logvar_ = logvar[:, 0]
                true_ = true[:, 0]
            if len(mu.shape) > 3:
                mu_ = mu[:, -1, 0]
                logvar_ = logvar[:, -1, 0]
                true_ = true[:, -1, 0]
            else:
                mu_ = mu
                logvar_ = logvar
                true_ = true

            loss = K.greater(2 * K.sqrt(K.exp(logvar_)), K.abs(true_ - mu_))
            reduce_loss = K.mean(loss)
        else:
            # Gaussian NLL-like loss
            loss = K.pow(true - mu, 2) * K.exp(-logvar) + alpha * logvar
            reduce_loss = K.mean(loss, axis=0)
        return reduce_loss

    return BNN_loss


# EDL LOSS
# https://github.com/aamini/evidential-deep-learning/blob/main/evidential_deep_learning/layers/dense.py

def build_EDL_loss(coeff_reg=0.95, coeff_var_pen=1):
    def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):
        twoBlambda = 2 * beta * (coeff_var_pen + v)
        nll = (
            0.5 * tf.math.log(np.pi / v)
            - alpha * tf.math.log(twoBlambda)
            + (alpha + 0.5) * tf.math.log(v * (y - gamma) ** 2 + twoBlambda)
            + tf.math.lgamma(alpha)
            - tf.math.lgamma(alpha + 0.5)
        )
        return tf.reduce_mean(nll) if reduce else nll

    def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
        KL = (
            0.5 * (a1 - 1) / b1 * (v2 * tf.square(mu2 - mu1))
            + 0.5 * v2 / v1
            - 0.5 * tf.math.log(tf.abs(v2) / tf.abs(v1))
            - 0.5
            + a2 * tf.math.log(b1 / b2)
            - (tf.math.lgamma(a1) - tf.math.lgamma(a2))
            + (a1 - a2) * tf.math.digamma(a1)
            - (b1 - b2) * a1 / b1
        )
        return KL

    def NIG_Reg(y, gamma, v, alpha, beta, omega=0.01, reduce=True, kl=False):
        # error = tf.stop_gradient(tf.abs(y-gamma))
        error = tf.abs(y - gamma)
        if kl:
            kl = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1 + omega, beta)
            reg = error * kl
        else:
            evi = 2 * v + (alpha)
            reg = error * evi

        return tf.reduce_mean(reg) if reduce else reg

    def EvidentialRegressionLoss(y_true, pred):
        gamma, v, alpha, beta = tf.split(pred, 4, axis=-1)
        loss_NLL = NIG_NLL(y_true, gamma, v, alpha, beta, reduce=False)
        loss_Reg = NIG_Reg(y_true, gamma, v, alpha, beta, reduce=False)
        reduce_loss = K.mean(loss_NLL + coeff_reg * loss_Reg, axis=0)
        return reduce_loss
    return EvidentialRegressionLoss




