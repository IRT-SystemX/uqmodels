################################################################################
# Source that specify UQEstimator class : Estimator performing Uncertainty
# quantification according to serveral paradigms
from abc import ABCMeta, abstractmethod
from typing import Callable, TypeVar

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler

import uqmodels.postprocessing.UQ_processing as UQ_proc
from uqmodels.utils import aux_tuning

# To do : Specify an UQmeasure object & specify the type_UQ list

Array = TypeVar("Array")
List_Estimators = TypeVar("List_Estimators")
Estimator = TypeVar("Estimator")
Dict_params = TypeVar("Dict_params")

# Current support type :
list_type_UQ = [
    "None",
    "var",
    "res_var",
    "2var",
    "res_2var",
    "var_A&E",
    "quantile",
    "res_quantile",
]
# To Do : add var_residuals, 2var_residuals, quantile_residuals


def check_type_UQ(type_UQ):
    if type_UQ in list_type_UQ:
        None
    else:
        raise ValueError(type_UQ, " not in ", list_type_UQ)


def get_UQEstimator_parameters(
    model_parameters={},
    factory_parameters={},
    training_parameters=None,
    type_output=None,
    rescale=False,
    random_state=None,
    **kwargs,
):
    """Generate dict object containing UQEstimators parameters to provide to init().

    Args:
        model_parameters (dict, optional): UQestimators parameters. Defaults to {}.
        factory_parameters (dict, optional): Factory parameters, to data preprocessing. Defaults to {}.
        training_parameters (_type_, optional): Model training parameters for DEEP estimators. Defaults to None.
        type_output (_type_, optional): Specification of deep learning output. Defaults to None.
        rescale (bool, optional): Use or not internal rescale function  Defaults to False.
        random_state (bool): handle experimental random using seed.
    Returns:
        dict: UQEstimators parameters
    """
    UQEstimator_parameters = {
        "factory_parameters": factory_parameters,
        "model_parameters": model_parameters,
        "factory_parameters": factory_parameters,
        "training_parameters": training_parameters,
        "type_output": type_output,
        "rescale": rescale,
        "random_state": random_state,
    }

    for key_arg in kwargs.keys():
        UQEstimator_parameters[key_arg] = kwargs[key_arg]

    return UQEstimator_parameters


class UQEstimator(BaseEstimator, metaclass=ABCMeta):
    """Abstract structure of a UQEstimator : Estimator (fit/predict) that
    perform prediction(optionaly) and UQmeasure estimation
    """

    def __init__(
        self,
        name="UQEstimator",
        type_UQ="None",
        rescale=False,
        type_UQ_params=None,
        var_min=0.000001,
        random_state=None,
    ):
        """UQestimator that perform prediction(optionaly) and UQmeasure estimation

        Args:
            name (str, optional): name. Defaults to 'UQEstimator'.
            type_UQ (str, optional): nature of UQmeasure. Defaults to 'None'.
            rescale (bool, optional): boolean flag that specify to use or not rescalling procedure. Defaults to False.
            type_UQ_params (_type_, optional): additional information about UQMeasure parameters. Defaults to None.
            random_state (bool): handle experimental random using seed.
        """
        self.name = name
        self.type_UQ = type_UQ
        self.type_UQ_params = type_UQ_params
        self.rescale = rescale
        self.random_state = random_state
        self.var_min = var_min
        self.is_fitted = False

        self.len_y_shape = None
        self.scaler = [
            StandardScaler(with_mean=True, with_std=True),
            StandardScaler(with_mean=True, with_std=True),
        ]
        self.random_state = random_state

    def _format(self, X, y, type_transform=False, mode_UQ=False, skip_format=False):
        """Data normalisation procedure (apply StandardScaler scikit learn) apply to features, target,
        prediction and UQmeasure design to hold UQ for several type of UQ use renormalise_UQ function of UQ_processing

        Args:
            X (_type_): Features
            y (_type_): Targets/Observation or prediction or UQmeasure)
            type_transform: action : {fit_transform, : fit scaler and transform
                                      transform, : use fitted scaler to transform
                                      inverse_transform}
            mode_UQ (bool, optional): true if format UQ measure

        Returns:
            X,y: Normalised features, and targets (or prediction or UQmeasure)
        """
        if skip_format:
            return (X, y)

        scalerX, scalerY = self.scaler

        if isinstance(y, tuple):
            y = np.array(list(y))

        if type_transform == "fit_transform":
            self.len_y_shape = len(y.shape)

        if not (hasattr(self, "len_y_shape")):
            if y is not None:
                self.len_y_shape = len(y.shape)

        if self.rescale:
            if y is not None:
                if len(y.shape) == 1:
                    y = y[:, None]

            if type_transform == "fit_transform":  # Fit X&Y Scaler
                if X is not None:
                    X = self.scaler[0].fit_transform(X)
                if y is not None:
                    y = self.scaler[1].fit_transform(y)

            elif type_transform == "transform":
                if X is not None:
                    X = self.scaler[0].transform(X)
                if y is not None:
                    y = self.scaler[1].transform(y)

            elif type_transform == "inverse_transform":  # Inverse Transform pred or UQ
                X_transformer = self.scaler[0].inverse_transform
                Y_transformer = self.scaler[1].inverse_transform
                if X is not None and not (len(X) == 0):
                    X = X_transformer(X)

                if y is not None and not (len(y) == 0):
                    if mode_UQ:  # UQ
                        y = UQ_proc.renormalise_UQ(
                            y, self.type_UQ, self.scaler[1], var_min=self.var_min
                        )

                    else:
                        y = Y_transformer(y)

        if y is not None:
            if self.len_y_shape == 1:
                y = np.squeeze(y)

        return (X, y)

    @abstractmethod
    def fit(self, X: np.array, y: np.array, skip_format=False, **kwargs) -> None:
        """Fit UQestimator using training data.
        Args:
            X: train features
            y: train targets/observations
        """

    @abstractmethod
    def predict(self, X: np.array, skip_format=False, **kwargs):
        """Compute prediction (or provide None) and UQ-measure
        Args:
            X: features
        Returns:
            pred, UQ_measure
        """

    def _tuning(
        self,
        estimator: Estimator,
        X: Array,
        y: Array,
        n_esti: int = 100,
        folds: int = 4,
        score: str = "neg_mean_squared_error",
        params=None,
        **kwarg,
    ) -> None:
        """Fine-tune an estimator using a standards scikit learns gridsearch procedure on params (grid_parameters)
        Args:
            X: features
            y: targets/observations
        """
        return aux_tuning(
            estimator,
            X,
            y,
            params,
            score,
            n_esti,
            folds,
            random_state=self.random_state,
        )

    def get_params(self, deep=False):
        dict_params = {}
        for key in self.__dict__.keys():
            if hasattr(self.__getattribute__(key), "get_params"):
                dict_params[key] = self.__getattribute__(key).get_params(deep=False)
            else:
                dict_params[key] = self.__getattribute__(key)
        return dict_params

    def factory(
        self, X, y, type_transform="transform", only_fit_scaler=False, **kwargs
    ):
        if only_fit_scaler:
            type_transform = "fit_transform"
        if y is None:
            X, _ = self._format(X, None, type_transform=type_transform)
        else:
            X, y = self._format(X, y, type_transform=type_transform)
        return (X, y, None)


class MeanVarUQEstimator(UQEstimator):
    """Mean var UQ Estimator that estimate a mean values and estimate irreductible"""

    def __init__(
        self,
        estimator=None,
        estimator_var=None,
        type_UQ="var",
        name="MeanVarUQEstimator",
        rescale=False,
        var_min=0.00001,
        random_state=None,
    ):
        """Initialization

        Args:
            name (str): Name
            estimator (estimator): Target predictor
            estimator_var (estimator): variance estimators
            random_state (bool): handle experimental random using seed.
        """

        super().__init__(
            name=name,
            type_UQ=type_UQ,
            rescale=rescale,
            var_min=var_min,
            random_state=random_state,
        )
        self.estimator = estimator
        self.estimator_var = estimator_var

    def fit(self, X, y, skip_format=False, **kwargs):
        """Fit UQestimator using training data.
        Args:
            X: train features
            y: train targets/observations
        """
        X, y = self._format(X, y, "fit_transform", skip_format=skip_format)
        self.estimator.fit(X, y)
        pred = self.estimator.predict(X)
        residual = np.power(np.squeeze(y) - np.squeeze(pred), 2)
        residual = residual.reshape(y.shape)
        self.estimator_var.fit(X, residual)
        self.is_fitted = True

    def predict(self, X, skip_format=False, **kwargs):
        """Compute prediction (or provide None) and UQ-measure
        Args:
            X: features
        Returns:
            pred, UQ_measure"""

        X, _ = self._format(X, None, "fit", skip_format=skip_format)

        pred = self.estimator.predict(X)
        UQ = self.estimator_var.predict(X)

        _, pred = self._format(None, pred, "inverse_transform")
        _, UQ = self._format(None, UQ, "inverse_transform", mode_UQ=True)
        return pred, UQ

    def _format(
        self,
        X: np.array,
        y: np.array,
        type_transform: str,
        mode_UQ: bool = False,
        skip_format=False,
    ):
        """Data normalisation procedure : see UQModels _gormat

        Args:
            X (np.array): _description_
            y (np.array): _description_
            type_transform (str): _description_
            mode_UQ (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        X, y = super()._format(X, y, type_transform, mode_UQ, skip_format=skip_format)
        return (X, y)


class QuantileUQEstimator(UQEstimator):
    def __init__(
        self,
        list_estimators,
        list_alpha: list = [0.025, 0.5, 0.975],
        type_UQ: str = "quantile",
        name: str = "QuantileUQEstimator",
        rescale=False,
        var_min=0.00001,
        random_state=None,
        **kwargs,
    ) -> None:
        """Initialise all attributes of the UQEstimatorGBRQ class.

        Args:
            list_estimator: List of provided quatile estimators by default use GradientBoostingRegressor
                as default estimator.
            list_alpha: List of quantile values to estimates.
            type_quantile : quantile or res_quantile : express if y provided is already a residual or None
            list_alpha :list of alpha conf quantile to estimate

        """
        super().__init__(
            name=name,
            type_UQ=type_UQ,
            rescale=rescale,
            var_min=var_min,
            random_state=random_state,
        )
        self.list_alpha = list_alpha
        self.type_UQ_params = {"list_alpha": list_alpha}
        self.list_estimators = list_estimators

    def fit(self, X, y, skip_format=False, **kwargs):
        X, y = self._format(X, y, "fit_transform", skip_format=skip_format)
        for estimator in self.list_estimators:
            estimator.fit(X, y)
        self.is_fitted = True

    def predict(self, X, skip_format=False, **kwargs):
        pred = None
        X, _ = self._format(X, None, "fit", skip_format=skip_format)

        if 0.5 in self.type_UQ_params["list_alpha"]:
            idx = self.type_UQ_params["list_alpha"].index(0.5)
            pred = self.list_estimators[idx].predict(X)

        UQ = []
        for n, estimator in enumerate(self.list_estimators):
            UQ.append(estimator.predict(X))
        UQ = np.stack(UQ)
        _, pred = self._format(None, pred, "inverse_transform", mode_UQ=False)
        _, UQ = self._format(None, UQ, "inverse_transform", mode_UQ=True)

        return pred, UQ

    def _format(
        self,
        X: np.array,
        y: np.array,
        type_transform: str,
        mode_UQ: bool = False,
        skip_format=False,
    ):
        """Data normalisation procedure (apply StandardScaler scikit learn) apply to features, target,
        prediction and UQmeasure.
        Designed to hold UQ for several type of UQ use renormalise_UQ function of UQ_processing

        Args:
            X (_type_): Features
            y (_type_): Targets/Observation or prediction or UQmeasure)
            type_transform: action : {fit_transform, : fit scaler and transform
                                      transform, : use fitted scaler to transform
                                      inverse_transform}
            mode_UQ (bool, optional): true if format UQ measure

        Returns:
            X,y: Normalised features, and targets (or prediction or UQmeasure)
        """
        X, y = super()._format(
            X=X,
            y=y,
            type_transform=type_transform,
            mode_UQ=mode_UQ,
            skip_format=skip_format,
        )
        return (X, y)

    def _tuning(
        self,
        X: Array,
        y: Array,
        n_esti: int = 100,
        folds: int = 4,
        params: Dict_params = None,
        **kwarg,
    ) -> None:
        """Perform a random search tuning using a parameter grid."""
        X, y = self._format(X, y, "fit_transform")

        def make_q_loss(beta: float) -> Callable:
            """Build the quantile loss calculation function.

            Args:
                beta: Must be between 0 and 1. Miss-coverage rate.

            Returns:
                The quantile loss function.
            """

            def quantile_loss(y_true: Array, pred: Array) -> float:
                """Compute the quantile loss between a vector of
                predicted values and one of ground truth values for y.

                The beta parameter is set outside the function (as an
                    argument to the maker function).

                Args:
                    y_true: vector of ground truth values
                    pred: vector of predicted values

                Returns:
                    The loss score.
                """
                delta = y_true - pred
                score = (
                    (beta - 1.0) * delta * (delta < 0) + beta * delta * (delta >= 0)
                ).sum()
                return score

            return quantile_loss

        for n, estimator in enumerate(self.list_estimators):
            alpha = self.list_alpha[n]
            if params is not None:
                # Mean estimator tuning
                if hasattr(self, "pretuned"):
                    if self.pretuned:
                        score = make_scorer(
                            make_q_loss(alpha),
                            greater_is_better=False,
                        )
                        self.list_estimators[n] = super()._tuning(
                            estimator=estimator,
                            X=X,
                            y=y,
                            params=params,
                            score=score,
                            n_esti=n_esti,
                            folds=folds,
                        )

        # else no tuning
