from copy import deepcopy

import numpy as np

import uqmodels.postprocessing.anomaly_processing as anom_proc
import uqmodels.postprocessing.UQ_processing as UQ_proc
import uqmodels.postprocessing.UQKPI_Processor as UQKPI_proc
from uqmodels.processing import Processor
from uqmodels.utils import apply_middledim_reduction


class Multiscale_Anomscore_processor(UQKPI_proc.UQKPI_Processor):
    """Processor aiming to produce an contextual deviation anomaly score from UQmeasure prediction and observation"""

    def __init__(
        self,
        name="Multiscale_anomscore",
        KPI_parameters=dict(),
        cache=None,
        random_state=None,
        **kwargs
    ):
        """Initialization

        Args:
            name (str, optional): name. Defaults to 'Anomscore'.
            KPI_parameters (_type_, optional): dict of parameters link to compute_score function.
                Defaults to None will use predefined paramters.
            cache (_type_, optional): Cache manager or none
        """

        KPI_parameters_default = {
            "type_norm": "Nsigma_local",
            "q_var": 1,
            "d": 2,
            "beta": 0.001,
            "beta_source": 0.001,
            "beta_global": 0.001,
            "min_cut": 0.002,
            "max_cut": 0.998,
            "per_seuil": 0.999,
            "dim_chan": 1,
            "type_fusion": "mahalanobis",
            "filt": [0.1, 0.2, 0.5, 0.2, 0.1],
        }

        for key in KPI_parameters_default.keys():
            if key not in KPI_parameters.keys():
                KPI_parameters[key] = KPI_parameters_default[key]

        super().__init__(
            name=name,
            KPI_parameters=KPI_parameters,
            cache=cache,
            random_state=random_state,
            **kwargs
        )

        self.params_ = None

    def _aux_proc(
        self,
        list_UQ,
        type_UQ,
        list_pred,
        list_y,
        type_UQ_params=None,
        ctx_mask=None,
        mutliscale_anom_score_params_=None,
        **kwargs
    ):
        """Auxialiar function that implement both fit and tranform procedure for Mutliscale_anom_score.

        It is mainly based on the application of anom_proc.fit/compute_anom_score &
        anom_proc.fit/compute_score_fusion.

        Args:
            UQ (np.array): UQmeasure provide by the UQestiamtor.
            type_UQ (str): Type_UQ of the UQestimators.
            pred (np.array): Prediction of the predictor or the UQestimator.
            y (np.array, optional): Targets/Observations, can be None if processor does't need y to fit
            type_UQ_params (type_UQ_params, optional): Additional information about type_UQ parameters.
                Defaults to None.
        """

        n_dim = len(list_y)
        n_chan = list_y[0].shape[2]
        n_step = max([len(list_y[source_id]) for source_id in range(n_dim)])

        S_anom_chan = np.zeros((n_step, n_dim * n_chan))
        S_anom_source = np.zeros((n_step, n_dim))
        list_bot = []
        list_top = []

        mode_fit = False
        if mutliscale_anom_score_params_ is None:
            mode_fit = True
            mutliscale_anom_score_params_ = {
                "anom_score_loc_params_": [],
                "score_fusion_loc_params_": [],
                "score_fusion_agg_params_": None,
            }

        for n_s in range(n_dim):
            pred, UQ = list_pred[n_s], list_UQ[n_s]
            y = list_y[n_s]

            anom_score_loc_params_ = None
            score_fusion_loc_params_ = None
            if not mode_fit:
                anom_score_loc_params_ = mutliscale_anom_score_params_[
                    "anom_score_loc_params_"
                ][n_s]

                score_fusion_loc_params_ = mutliscale_anom_score_params_[
                    "score_fusion_loc_params_"
                ][n_s]

            # Use fact that if params_ is none then fit is computed internall
            (s_loc, born), anom_score_loc_params_ = anom_proc.compute_anom_score(
                UQ=UQ,
                type_UQ=type_UQ,
                pred=pred,
                y=y,
                type_UQ_params=type_UQ_params,
                ctx_mask=ctx_mask,
                with_born=True,
                params_=anom_score_loc_params_,
                **self.KPI_parameters,
                **kwargs
            )
            mutliscale_anom_score_params_["anom_score_loc_params_"].append(
                anom_score_loc_params_
            )

            # Use fact that if params_ is none then fit is computed internally
            s_agg, score_fusion_loc_params_ = anom_proc.compute_score_fusion(
                s_loc,
                ctx_mask=ctx_mask,
                params_=score_fusion_loc_params_,
                **self.KPI_parameters
            )

            mutliscale_anom_score_params_["score_fusion_loc_params_"].append(
                score_fusion_loc_params_
            )

            mask = (np.arange(n_dim * n_chan) >= (n_s * n_chan)) & (
                np.arange(n_dim * n_chan) < ((n_s + 1) * n_chan)
            )

            S_anom_chan[:, mask] = s_loc
            S_anom_source[:, n_s] = s_agg[:, 0]

            list_bot.append(born[0])
            list_top.append(born[1])

        # type_norm may be Chi² -> Exploratory works

        score_fusion_agg_params_ = None
        if not mode_fit:
            score_fusion_agg_params_ = mutliscale_anom_score_params_[
                "score_fusion_agg_params_"
            ]

        S_anom_agg, score_fusion_agg_params_ = anom_proc.compute_score_fusion(
            S_anom_source,
            ctx_mask=None,
            beta=self.KPI_parameters["beta_global"],
            type_fusion=self.KPI_parameters["type_fusion"],
            type_norm="quantiles_global",
            per_seuil=0.995,
            d=2,
            filt=self.KPI_parameters["filt"],
            params_=score_fusion_agg_params_,
        )

        mutliscale_anom_score_params_["score_fusion_agg_params_"] = (
            score_fusion_agg_params_
        )

        if mode_fit:
            return mutliscale_anom_score_params_

        else:
            return S_anom_chan, S_anom_source, S_anom_agg, list_bot, list_top

    def fit(
        self,
        list_UQ,
        type_UQ,
        list_pred,
        list_y,
        type_UQ_params=None,
        ctx_mask=None,
        **kwargs
    ):
        """Fitting procedure aim to estimate and store Multiscale_Anomscore_processor params for
        Multiscale_Anomscore computation

        Args:
            UQ (np.array): UQmeasure provide by the UQestiamtor.
            type_UQ (str): Type_UQ of the UQestimators.
            pred (np.array): Prediction of the predictor or the UQestimator.
            y (np.array, optional): Targets/Observations, can be None if processor does't need y to fit
            type_UQ_params (type_UQ_params, optional): Additional information about type_UQ parameters.
                Defaults to None.
        """
        self.params_ = self._aux_proc(
            list_UQ,
            type_UQ,
            list_pred,
            list_y,
            type_UQ_params=type_UQ_params,
            ctx_mask=ctx_mask,
        )
        super().fit(type_UQ=type_UQ)

    def transform(
        self,
        list_UQ,
        type_UQ,
        list_pred,
        list_y,
        type_UQ_params=None,
        ctx_mask=None,
        **kwargs
    ):
        """Transform procedure aim to transform (predictor) & UQestimator output into a Multiscale_Anomscore
        according to Multiscale_Anomscore_params

        Args:
            UQ (np.array): UQmeasure provide by the UQestiamtor.
            type_UQ (str): Type_UQ of the UQestimators.
            pred (np.array): Prediction of the predictor or the UQestimator.
            y (np.array, optional): Targets/Observations, can be None if processor does't need y to fit
            type_UQ_params (type_UQ_params, optional): Additional information about type_UQ parameters.
                Defaults to None.

        Returns:
            A tuple (S_anom_chan, S_anom_agg, S_anom_source, list_bot, list_top)
            where S_anom_chan is a mutli-dimensional providing Anom score for each channels of each source
            on compute_anom_score,
            S_anom_source is an aggregated score providing Anom score at source lvl on compute_score_fusion,
            S_anom_agg is an aggregated score providing 1D Anom score based on compute_score_fusion,
            list_bot is an anomalie lower threeshold for each chan of each sensors provided by compute_anom_score
            and list_top is an anomalie upper threeshold for each chan of each sensors provided by compute_anom_score
        """
        super().transform(type_UQ=type_UQ)
        S_anom_chan, S_anom_agg, S_anom_source, list_bot, list_top = self._aux_proc(
            list_UQ,
            type_UQ,
            list_pred,
            list_y,
            type_UQ_params=type_UQ_params,
            ctx_mask=ctx_mask,
            mutliscale_anom_score_params_=self.params_,
            **kwargs
        )
        return (S_anom_chan, S_anom_agg, S_anom_source, list_bot, list_top)
