"""
Implementation of usual prediction wrappers.
"""

from copy import deepcopy
from typing import Optional, TypeVar

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    ExpSineSquared,
    RationalQuadratic,
    WhiteKernel,
)

from uqmodels.modelization.UQEstimator import (
    MeanVarUQEstimator,
    QuantileUQEstimator,
    get_UQEstimator_parameters,
)
from uqmodels.utils import add_random_state

Array = TypeVar("Array")
UQmeasure = TypeVar("UQmeasure")
List_Estimators = TypeVar("List_Estimators")
Estimator = TypeVar("Estimator")
Kernel = TypeVar("Kernel")
Dict_params = TypeVar("Dict_params")


class GBRQ_UQEstimator(QuantileUQEstimator):
    """Uncertainty quantification approch based on Quantile Gradient Boosting :
    Instatiation OF QuantileUQEstimators using GBRQ scikilearn models.

    Attributes/properties:
        name (str): Name of the UQ method for future reference.
        type_UQ (str): Method family among the main categories of UQUQEstimators.
        list_estimators (Estimator): List of quantile estimator.
        list_alpha : list of alpha confidence level for each quantile estimators.
        pretuned (bool): Whether to disable parameter model tuning.

    Mean Methods:
        fit: Fit the list of quantile estimators.
        predict:(pred,UQ) Predict for each quantile estimators
    """

    def __init__(
        self,
        list_estimators: List_Estimators = None,
        list_alpha: list = [0.025, 0.5, 0.975],
        type_UQ="quantile",
        name: str = "GBRQ_UQEstimator",
        pretuned: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialise all attributes of the UQEstimatorGBRQ class.

        Args:
            list_estimator: List of provided quatile estimators by default use GradientBoostingRegressor
                as default estimator.
            estimator_qmid: Medium quantile estimator, corresponding to
            the UQEstimator making the forecast. If None, use a
            GradientBoostingRegressor as default estimator.
        """
        if list_estimators is None:
            list_estimators = []
            for i, alpha in enumerate(list_alpha):
                list_estimators.append(
                    GradientBoostingRegressor(
                        random_state=add_random_state(random_state, i),
                        loss="quantile",
                        alpha=alpha,
                    )
                )

        super().__init__(
            list_estimators,
            list_alpha,
            type_UQ=type_UQ,
            name=name,
            random_state=random_state,
        )
        self.pretunned = pretuned

    def _format(
        self,
        X: np.array,
        y: np.array,
        type_transform: str,
        mode_UQ: bool = False,
        skip_format=False,
    ):
        X, y = super()._format(
            X=X,
            y=y,
            type_transform=type_transform,
            mode_UQ=mode_UQ,
            skip_format=skip_format,
        )
        return (X, y)

    def fit(self, X: Array, y: Array, skip_format=False, **kwargs) -> None:
        """Fit GBRQ_UQEstimator list using QuantileUQEstimator fit methods.
        Args:
            X: Features
            y: Target values
        """
        super().fit(X, y, skip_format=skip_format)

    def predict(self, X: Array, skip_format=False, **kwargs):
        """Perform the quantile estimation using QuantileUQEstimator predict methods.

        Args:
            X: Features

        Return:
            pred: Median prediction or None if Median quantile estimators is not in list_estimators
            UQ  : List of quantiles estimatiors
        """
        pred, UQ = super().predict(X, skip_format=skip_format)
        return (pred, UQ)

    def _tuning(
        self,
        X: Array,
        y: Array,
        n_esti: int = 100,
        folds: int = 4,
        params: Dict_params = None,
        **kwarg,
    ) -> None:
        """Perform a random search tuning using a parameter grid with QuantileUQEstimator _tuning methode"""
        if not (self.pretunned):
            super()._tuning(X, y, n_esti, folds, params)


class GPR_UQEstimator(MeanVarUQEstimator):
    """Uncertainty quantification approch based on Gaussian Process Regressor.
    Instatiation OF MeanVarUQEstimator using GP scikilearn model.
    Warning GP has UQ limitation : see  "The pitfalls of using Gaussian Process Regression for normative modeling"
    """

    def __init__(
        self,
        name: str = "Gaussian_Process_UQ",
        kernel: Kernel = None,
        gp_alpha: float = 0.000001,
        drop_ratio: float = 0.0,
        rescale: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        self.gp_alpha = gp_alpha
        self.kernel = kernel
        self.drop_ratio = drop_ratio

        if kernel is None:
            self.kernel = (
                ConstantKernel() * RationalQuadratic()
                + RBF() * ExpSineSquared()
                + RBF() * WhiteKernel(0.0001)
                + WhiteKernel(0.001)
            )
        estimator = GaussianProcessRegressor(
            self.kernel,
            alpha=self.gp_alpha,
            n_restarts_optimizer=4,
            random_state=random_state,
        )
        super().__init__(
            estimator=estimator,
            estimator_var=None,
            type_UQ="var",
            name=name,
            rescale=rescale,
            random_state=random_state,
        )

    def _format(
        self,
        X: np.array,
        y: np.array,
        type_transform: str,
        mode_UQ: bool = False,
        skip_format=False,
    ):
        X, y = super()._format(
            X=X,
            y=y,
            type_transform=type_transform,
            mode_UQ=mode_UQ,
            skip_format=skip_format,
        )
        return (X, y)

    def fit(self, X, y, skip_format=False, **kwargs):
        """Fit procedure of Gaussian process

        Args:
            X: Training vectors.
            y: Target values.
        """
        X, y = self._format(X, y, "fit_transform", skip_format=skip_format)

        # GP model make native variance estimation
        if self.drop_ratio == 0:
            self.estimator.fit(X, y)

        else:
            mask = np.random.rand(len(y)) > self.drop_ratio
            X_mask, y_mask = X[mask], y[mask]
            self.estimator.fit(X_mask, y_mask)

    def predict(self, X: Array, skip_format=False, **kwargs):
        """Perform the prediction task of the forecasting and
        uncertainty models on X.

        Args:
            X: Samples on which to perform the prediction.

        Return:
            A tuple containing the forecast, the predicted lower and
                upper quantiles and var.
        """

        X, _ = self._format(X, None, "transform", skip_format)

        pred, std = self.estimator.predict(X, return_std=True)
        UQ = np.power(std, 2)

        _, pred = self._format(None, pred, "inverse_transform", mode_UQ=True)
        _, UQ = self._format(None, UQ, "inverse_transform", mode_UQ=True)
        return (pred, UQ)

    def _tuning(self, X: Array, y: Array, **kwarg) -> None:
        """Perform random search tuning using a given grid parameter"""
        print("No tunning procedure")


class REGML_UQEstimator(MeanVarUQEstimator):
    """Uncertainty quantification approch based on ML regression of bias and variance
    Instanciation of specific Pred-Biais-Var Regression scheme for uncertainty quantification
    """

    def __init__(
        self,
        estimator=None,
        estimator_var=None,
        pretuned: bool = False,
        type_UQ: str = "var",
        use_biais: bool = True,
        name: str = "REGML_UQEstimator",
        var_min: float = 0.000001,
        rescale: bool = False,
        random_state: Optional[int] = None,
    ) -> None:

        if estimator is None:
            estimator = RandomForestRegressor(random_state=random_state)

        if estimator_var is None:
            estimator_var = deepcopy(estimator)

        name = name + "_" + type_UQ
        type_UQ = type_UQ

        super().__init__(
            estimator=estimator,
            estimator_var=estimator_var,
            name=name,
            type_UQ=type_UQ,
            rescale=rescale,
            var_min=var_min,
            random_state=random_state,
        )

        self.pretuned = pretuned
        self.use_biais = use_biais

        if self.use_biais:
            self.estimator_bias = deepcopy(estimator)

        self.std_norm = 1
        if self.type_UQ in ["var", "res_var"]:
            self.estimator_var = estimator_var

        elif self.type_UQ in ["2var", "res_2var"]:
            self.estimator_var_bot = estimator_var
            self.estimator_var_top = deepcopy(estimator_var)

    def _format(
        self, X, y, type_transform=False, mode_UQ=False, skip_format=False, **kwargs
    ):
        X, y = super()._format(
            X=X,
            y=y,
            type_transform=type_transform,
            mode_UQ=mode_UQ,
            skip_format=skip_format,
        )
        return X, y

    def get_params(self, **kwargs) -> Dict_params:
        dict_params = super().get_params()
        return dict_params

    def fit(self, X: Array, y: Array, skip_format=False, **kwargs) -> None:
        """Train y_lowerh forecasting and UQ models on (X,Y)."""

        X, y = self._format(X, y, "fit_transform", skip_format=skip_format)

        # Train forecaster models and compute residuals
        self.estimator.fit(X, y)
        pred = self.estimator.predict(X)
        residual = np.squeeze(y) - np.squeeze(pred)
        residual = residual.reshape(y.shape)

        if self.use_biais:
            # Train bias model and reduce bias to residuals
            self.estimator_bias.fit(X, residual)
            bias = self.estimator_bias.predict(X)
            residual = np.squeeze(residual) - np.squeeze(bias)
            residual = residual.reshape(y.shape)

            # Normalisation for variance learning
            self.std_norm = 1 / np.abs(residual).mean()

        else:
            # Normalisation for variance learning
            self.std_norm = 1 / np.abs(residual).mean()

        # Train variance estimator of residuals
        if self.type_UQ in ["var", "res_var"]:
            residual = np.power(residual * self.std_norm, 2)
            self.estimator_var.fit(X, residual)

        # Train variance estimators of positive and negative residuals.
        elif self.type_UQ in ["2var", "res_2var"]:
            mask_res_bot = np.squeeze((residual) <= 0)
            mask_res_top = np.squeeze((residual) >= 0)
            if len(mask_res_bot.shape) == 1:
                residual = np.power(residual * self.std_norm, 2)
                self.estimator_var_bot.fit(X[mask_res_bot], residual[mask_res_bot])
                self.estimator_var_top.fit(X[mask_res_top], residual[mask_res_top])
            else:
                residual = np.power(residual * self.std_norm, 2)
                self.estimator_var_bot.fit(X, residual * mask_res_bot)
                self.estimator_var_top.fit(X, residual * mask_res_top)
        else:
            print("type_UQ", self.type_UQ, "not covered")

    def predict(self, X: Array, skip_format=False, **kwargs):
        """Perform the prediction task of the forecasting and UQ models
        on features (X).

        Args:
            X: Samples on which to perform the prediction.

        Return:
            pred, UQ : Prediction and UQmeasure
        """

        X, _ = self._format(X, None, "transform", skip_format=skip_format)

        # predict Forecast and bias
        pred = self.estimator.predict(X)

        if self.use_biais:
            bias = self.estimator_bias.predict(X)
        else:
            bias = 0

        # Predict std of residuals
        var_min = self.var_min
        if self.type_UQ in ["var", "res_var"]:
            var = self.estimator_var.predict(X)
            var[var < var_min] = var_min
            UQ = np.sqrt(var) / np.power(self.std_norm, 2)

        # Predict std of positive and negative residuals
        elif self.type_UQ in ["2var", "res_2var"]:
            var_bot = self.estimator_var_bot.predict(X)
            var_bot[var_bot < var_min] = var_min
            var_bot = var_bot / np.power(self.std_norm, 2)

            var_top = self.estimator_var_top.predict(X)
            var_top[var_top < var_min] = var_min
            var_top = var_top / np.power(self.std_norm, 2)
            UQ = np.concatenate(
                [np.expand_dims(i, 0) for i in [var_bot, var_top]], axis=0
            )

        _, pred = self._format(None, pred + bias, "inverse_transform")
        _, UQ = self._format(None, UQ, "inverse_transform", mode_UQ=True)
        return (pred, UQ)

    def _tuning(
        self,
        X: Array,
        y: Array,
        n_esti: int = 100,
        folds: int = 4,
        params: Dict_params = None,
        **kwarg,
    ) -> None:
        """Perform random search tuning using a given grid parameter."""
        score = "neg_mean_squared_error"

        X, y = self._format(X, y, "fit_transform")

        # IF there is no parameter grid : skip tuning step.
        if params is not None:
            # IF forecast model is tunned, skip it's tuning step.
            if not (self.pretuned):
                print(X.shape, y.shape)
                self.estimator = super()._tuning(
                    self.estimator, X, y, n_esti, folds, score, params
                )

            self.estimator.fit(X, y)
            pred = self.estimator.predict(X)

            # Build residuals and tune bias model by random search on given greed parameters.
            residual = np.squeeze(y) - np.squeeze(pred)
            residual = residual.reshape(y.shape)
            print(X.shape, residual.shape)
            if self.use_biais:
                self.estimator_bias = super()._tuning(
                    self.estimator_bias,
                    X,
                    residual,
                    int(n_esti / 2),
                    folds,
                    score,
                    params,
                )
                self.estimator_bias.fit(X, residual)
                bias = self.estimator_bias.predict(X)

                # Correct residuals and tune variance model by random search on given greed parameters.
                residual = np.squeeze(residual) - np.squeeze(bias)
                residual = residual.reshape(y.shape)

            if self.type_UQ == "var":
                # Gaussian hypothesis : 1 variance model.
                residual = np.power(residual / residual.std(), 2)
                self.estimator_var = super()._tuning(
                    self.estimator_var,
                    X,
                    residual,
                    int(n_esti / 2),
                    folds,
                    score,
                    params,
                )

                self.estimator_var.fit(X, residual)

            elif self.type_UQ == "2var":
                # 2 Gaussian hypothesis :
                # 2 variance models for positive (top) and negative (bot) residuals STD.
                flag_res_top = np.squeeze((residual) >= 0)
                flag_res_bot = np.squeeze((residual) <= 0)
                residual = np.power(residual / residual.std(), 2)
                self.estimator_var_bot = super()._tuning(
                    self.estimator_var_bot,
                    X[flag_res_bot],
                    residual[flag_res_bot],
                    int(n_esti / 2),
                    folds,
                    score,
                    params,
                )

                self.estimator_var_bot.fit(X[flag_res_bot], residual[flag_res_bot])

                self.estimator_var_top = super()._tuning(
                    self.estimator_var_top,
                    X[flag_res_top],
                    residual[flag_res_top],
                    int(n_esti / 2),
                    folds,
                    score,
                    params,
                )

                self.estimator_var_top.fit(X[flag_res_top], residual[flag_res_top])
