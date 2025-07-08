##########################################################################
# KPI_Processor : Processor object (fit/transform procedure) that aim to
# tranform predictor & UQestimator output into KPI.
# We define a KPI_processor view (Transform any kind of UQ measure in a
# specific KPI form) rather than an UQ_processor (express an UQ measure in
# several KPI forms)

from copy import deepcopy

import numpy as np

import uqmodels.postprocessing.anomaly_processing as anom_proc
import uqmodels.postprocessing.UQ_processing as UQ_proc
from uqmodels.processing import Cache_manager, Processor
from uqmodels.utils import apply_middledim_reduction


class UQKPI_Processor(Processor):
    """KPI processor that aim to process (predictor) & UQestimator output into a KPI."""

    def __init__(
        self,
        name="UQ_processor",
        KPI_parameters={},
        cache=Cache_manager(),
        random_state=None,
        **kwargs
    ):
        """Init UQKPI_Processor

        Args:
            name (str, optional): name. Defaults to 'KPI_processor'.
            KPI_parameters: Set of parameters link to UQKPI to produce
            cache (cahe_manager or None, optional): cache_manager. if None no cache procedure
            random_state (bool): handle experimental random using seed.
        """
        super().__init__(
            name=name,
            cache=cache,
            KPI_parameters=KPI_parameters,
            random_state=random_state,
            **kwargs
        )

        self.type_UQ = None

    def fit(
        self, UQ=None, type_UQ=None, pred=None, y=None, type_UQ_params=None, **kwargs
    ):
        """fitting procedure aim to estimate and store KPI_processor params

        Args:
            UQ (np.array): UQmeasure provide by the UQestiamtor.
            type_UQ (str): Type_UQ of the UQestimators.
            pred (np.array): Prediction of the predictor or the UQestimator.
            y (np.array, optional): Targets/Observations, can be None if processor does't need y to fit
            type_UQ_params (type_UQ_params, optional): Additional information about type_UQ parameters.
                Defaults to None.
        """
        self.type_UQ = type_UQ
        super().fit(None)

    def transform(
        self, UQ=None, type_UQ=None, pred=None, y=None, type_UQ_params=None, **kwargs
    ):
        """Transform procedure aim to transform (predictor) & UQestimator output into a KPI using fitted parameters.
            by default return row UQ provided by UQEstimators.
        Args:
            UQ (np.array): UQmeasure provide by the UQestiamtor.
            type_UQ (str): Type_UQ of the UQestimators.
            pred (np.array): Prediction of the predictor or the UQestimator.
            y (np.array, optional): Targets/Observations, can be None if processor does't need y to fit
            type_UQ_params (type_UQ_params, optional): Additional information about type_UQ parameters.
                Defaults to None.
        """
        if self.type_UQ != type_UQ:
            print("Warning : fitted for", self.type_UQ, "give :", type_UQ)

        if "reduc_filter" in self.KPI_parameters.keys():
            if UQ is not None:
                UQ = np.array(
                    [
                        apply_middledim_reduction(
                            i, self.KPI_parameters["reduc_filter"]
                        )
                        for i in UQ
                    ]
                )

            if pred is not None:
                pred = apply_middledim_reduction(
                    pred, self.KPI_parameters["reduc_filter"]
                )

        if "pred_and_UQ" in self.KPI_parameters.keys():
            if self.KPI_parameters["pred_and_UQ"]:
                return (pred, UQ)

        return UQ


# Predictive_interval_processor :


class NormalPIs_processor(UQKPI_Processor):
    """Processor aiming to transform UQmeasure in Normal predictive intervals"""

    def __init__(
        self,
        name="Normal_PIs_processor",
        KPI_parameters={"list_alpha": [0.05, 0.95], "with_epistemic": True},
        cache=None,
        **kwargs
    ):
        """NormalPIs_processor initialization

        Args:
            name (str, optional): name of processpr. Defaults to 'Normal_PIs_processor'.
            list_alpha (list, optional): Quantile lvl of predictives interval. Defaults to [0.05, 0.95].
            with_epistemic (bool): if type_UQ is 'var_A&E and with epistemic is false, ignore epistemic
                and produce aleatoric confidence uncertainty
            cache (_type_, optional): Cache manager or None.
        """
        super().__init__(
            name=name, KPI_parameters=KPI_parameters, cache=cache, **kwargs
        )

        self.ndUQ_ratio = None
        self.params_ = None

    def fit(self, UQ, type_UQ, pred, y=None, type_UQ_params=None, **kwargs):
        """fitting procedure aim to estimate and store KPI_processor params for NormalPIs computation
        Args:
            UQ (np.array): UQmeasure provide by the UQestiamtor.
            type_UQ (str): Type_UQ of the UQestimators.
            pred (np.array): Prediction of the predictor or the UQestimator.
            y (np.array, optional): Targets/Observations, can be None if processor does't need y to fit
            type_UQ_params (type_UQ_params, optional): Additional information about type_UQ parameters.
                Defaults to None.
        """
        UQ = deepcopy(UQ)

        with_epistemic = True
        if "with_epistemic" in self.KPI_parameters.keys():
            with_epistemic = self.KPI_parameters["with_epistemic"]

        if not (with_epistemic):
            self.ndUQ_ratio = UQ_proc.get_nominal_disentangled_UQ_ratio(
                UQ, q_var=1, q_Eratio=3
            )

            sigma, sigma_E = UQ_proc.process_UQmeasure_to_TOT_and_E_sigma(
                UQ,
                type_UQ,
                pred=None,
                y=None,
                type_UQ_params=None,
                var_min=0,
                var_max=None,
                min_cut=0,
                max_cut=1,
                q_var=1,
                q_var_e=1,
                k_var_e=1,
                q_Eratio=3,
                ndUQ_ratio=self.ndUQ_ratio,
                reduc_filter=None,
                **self.KPI_parameters
            )

            UQ = np.power(sigma, 2), sigma_E * 0

        self.params_ = UQ_proc.fit_PI(
            UQ, type_UQ, pred, y, type_UQ_params=type_UQ_params, **self.KPI_parameters
        )

        super().fit(None, type_UQ=type_UQ)

    def transform(self, UQ, type_UQ, pred, y=None, type_UQ_params=None, **kwargs):
        """Transform procedure aim to transform (predictor) & UQestimator output into normal predictive intervales
        according to the list_alpha stored parameters.

        Args:
            UQ (np.array): UQmeasure provide by the UQestiamtor.
            type_UQ (str): Type_UQ of the UQestimators.
            pred (np.array): Prediction of the predictor or the UQestimator.
            y (np.array, optional): Targets/Observations, can be None if processor does't need y to fit
            type_UQ_params (type_UQ_params, optional): Additional information about type_UQ parameters.
                Defaults to None.

        Returns:
            list_PIs : Quantiles parametrized by list_alpha provided in the init procedure
        """
        super().transform(None, type_UQ=type_UQ)
        UQ = deepcopy(UQ)

        with_epistemic = True
        if "with_epistemic" in self.KPI_parameters.keys():
            with_epistemic = self.KPI_parameters["with_epistemic"]

        if type_UQ == "var_A&E":
            if not (with_epistemic):
                sigma, sigma_E = UQ_proc.process_UQmeasure_to_TOT_and_E_sigma(
                    UQ,
                    type_UQ,
                    pred=None,
                    y=None,
                    type_UQ_params=None,
                    var_min=0,
                    var_max=None,
                    min_cut=0,
                    max_cut=1,
                    q_var=1,
                    q_var_e=1,
                    k_var_e=1,
                    q_Eratio=3,
                    ndUQ_ratio=self.ndUQ_ratio,
                    **self.KPI_parameters
                )
                UQ = np.power(sigma, 2), sigma_E * 0

        list_PIs, _ = UQ_proc.compute_PI(
            UQ,
            type_UQ,
            pred,
            y,
            type_UQ_params,
            params_=self.params_,
            **self.KPI_parameters
        )
        return list_PIs


class Epistemicscorelvl_processor(UQKPI_Processor):
    """Processor aiming to transform UQmeasure (containing epistemic measure) into Epistemic_score_lvl (ADD link)"""

    def __init__(
        self, name="Epistemicscorelvl", KPI_parameters={}, cache=None, **kwargs
    ):
        """Initialization

        Args:
            name (str, optional): name Defaults to 'Epistemicscorelvl'.
            KPI_paramsaters (_type_, optional): Parameters of processor : here none. Defaults to None.
            cache (_type_, optional): Cache manager or none
        """
        super().__init__(
            name=name, KPI_parameters=KPI_parameters, cache=cache, **kwargs
        )

        self.params_ = None

    def fit(self, UQ, type_UQ, pred, y=None, type_UQ_params=None, **kwargs):
        """Fitting procedure aim to estimate and store KPI_processor params for Epistemicscorelvl computation

        Args:
            UQ (np.array): UQmeasure provide by the UQestiamtor.
            type_UQ (str): Type_UQ of the UQestimators.
            pred (np.array): Prediction of the predictor or the UQestimator.
            y (np.array, optional): Targets/Observations, can be None if processor does't need y to fit
            type_UQ_params (type_UQ_params, optional): Additional information about type_UQ parameters.
                Defaults to None.
        """
        self.params_ = UQ_proc.fit_Epistemic_score(
            UQ, type_UQ, pred, y=y, type_UQ_params=type_UQ_params, **self.KPI_parameters
        )
        super().fit(type_UQ=type_UQ)

    def transform(self, UQ, type_UQ, pred, y=None, type_UQ_params=None, **kwargs):
        """Transform procedure aim to transform (predictor) & UQestimator output into a Epistemic_scorelvl
        according Episticscore quantile fitted during training procedure.

        Args:
            UQ (np.array): UQmeasure provide by the UQestiamtor.
            type_UQ (str): Type_UQ of the UQestimators.
            pred (np.array): Prediction of the predictor or the UQestimator.
            y (np.array, optional): Targets/Observations, can be None if processor does't need y to fit
            type_UQ_params (type_UQ_params, optional): Additional information about type_UQ parameters.
                Defaults to None.

        Returns: Epistemic_scorelvl : Epistemic_scorelvl that express the quantile class of Epistemic values
            amoung quantile [0.50, 0.80, 0.90, 0.95, 0.975, 0.99, 0.999]
        """
        super().transform(None, type_UQ=type_UQ)
        Epistemic_scorelvl, _ = UQ_proc.compute_Epistemic_score(
            UQ,
            self.type_UQ,
            pred,
            y=y,
            type_UQ_params=type_UQ_params,
            params_=self.params_,
            **self.KPI_parameters
        )
        return Epistemic_scorelvl


class Anomscore_processor(UQKPI_Processor):
    """Processor aiming to produce an contextual deviation anomaly score from UQmeasure prediction and observation"""

    def __init__(self, name="Anomscore", KPI_parameters=dict(), cache=None, **kwargs):
        """Initialization

        Args:
            name (str, optional): name. Defaults to 'Anomscore'.
            KPI_parameters (_type_, optional): dict of parameters link to compute_score function.
                Defaults to None will use predefined paramters.
            cache (_type_, optional): Cache manager or none
        """
        super().__init__(
            name=name, KPI_parameters=KPI_parameters, cache=cache, **kwargs
        )

        self.params_ = None
        self.fusion_params_ = None

    def fit(
        self, UQ, type_UQ, pred, y=None, type_UQ_params=None, ctx_mask=None, **kwargs
    ):
        """Fitting procedure aim to estimate and store Anomscore_processor params for Anomscore computation

        Args:
            UQ (np.array): UQmeasure provide by the UQestiamtor.
            type_UQ (str): Type_UQ of the UQestimators.
            pred (np.array): Prediction of the predictor or the UQestimator.
            y (np.array, optional): Targets/Observations, can be None if processor does't need y to fit
            type_UQ_params (type_UQ_params, optional): Additional information about type_UQ parameters.
                Defaults to None.
        """
        self.params_ = anom_proc.fit_anom_score(
            UQ,
            type_UQ,
            pred,
            y=y,
            type_UQ_params=type_UQ_params,
            ctx_mask=ctx_mask,
            **self.KPI_parameters,
            **kwargs
        )

        if "fusion" in self.KPI_parameters.keys():
            if self.KPI_parameters["fusion"]:
                anom_score, _ = anom_proc.compute_anom_score(
                    UQ=UQ,
                    type_UQ=type_UQ,
                    pred=pred,
                    y=y,
                    type_UQ_params=type_UQ_params,
                    ctx_mask=ctx_mask,
                    params_=self.params_,
                    **self.KPI_parameters,
                    **kwargs
                )

                self.fusion_params_ = anom_proc.fit_score_fusion(
                    anom_score, **self.KPI_parameters
                )

        super().fit(None, type_UQ=type_UQ)

    def transform(
        self, UQ, type_UQ, pred, y=None, type_UQ_params=None, ctx_mask=None, **kwargs
    ):
        """Transform procedure aim to transform (predictor) & UQestimator output into a Epistemic_scorelvl
        according Episticscore quantile fitted during training procedure.

        Args:
            UQ (np.array): UQmeasure provide by the UQestimator.
            type_UQ (str): Type_UQ of the UQestimators.
            pred (np.array): Prediction of the predictor or the UQestimator.
            y (np.array, optional): Targets/Observations, can be None if processor does't need y to fit
            type_UQ_params (type_UQ_params, optional): Additional information about type_UQ parameters.
                Defaults to None.

        Returns: anom_score : Deviation score (pred-y) process by anom_proc.compute_anom_score
        """

        # Create a bug for python >3.8 : wrapped() missing 1required positional argument :'X'
        # super().transform(type_UQ=type_UQ)

        anom_score, _ = anom_proc.compute_anom_score(
            UQ=UQ,
            type_UQ=type_UQ,
            pred=pred,
            y=y,
            type_UQ_params=type_UQ_params,
            ctx_mask=ctx_mask,
            params_=self.params_,
            **self.KPI_parameters,
            **kwargs
        )

        if "fusion" in self.KPI_parameters.keys():
            if self.KPI_parameters["fusion"]:
                anom_score, _ = anom_proc.compute_score_fusion(
                    anom_score, params_=self.fusion_params_, **self.KPI_parameters
                )
                anom_score = -anom_score
        return anom_score