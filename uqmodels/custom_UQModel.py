###################################################################################
# UQModel : "Model than apply an modeling pipeline composed of an predictor(optional), UQestimator and KPI
# To write a custom UQModel, inherite from UQModel class an use supoer().__init__()

# Class UQmodels : from an UQestimators : perform dadada

import numpy as np

import uqmodels.postprocessing.custom_UQKPI_Processor as custom_UQProc
import uqmodels.postprocessing.UQKPI_Processor as UQProc
import uqmodels.preprocessing.preprocessing as pre_pre
import uqmodels.preprocessing.structure as pre_struc
from uqmodels.modelization.DL_estimator import lstm_ed, transformer_ed
from uqmodels.modelization.DL_estimator.metalayers import mlp
from uqmodels.modelization.DL_estimator.neural_network_UQ import (
    NN_UQ,
    get_params_dict,
    get_training_parameters,
)
from uqmodels.modelization.ML_estimator.random_forest_UQ import (
    RandomForestRegressor,
    RF_UQEstimator,
)
from uqmodels.preprocessing.Custom_Preprocessor import Generic_Features_processor
from uqmodels.preprocessing.features_processing import (
    compute_pca,
    fit_compute_lag_values,
    fit_pca,
)
from uqmodels.processing import Cache_manager
from uqmodels.UQModel import UQModel
from uqmodels.utils import apply_mask, coefficients_spreaded, cut

# We can also create a more complexe UQmodel that handle several UQKPI_Processor to build at inference (UQMesure, Predictive interval and model unreliability score) and after observation (Anomaly score)

# Specification of the UQestimator & Instanciation in a UQmodels wrapper that include post-processing


type_output = "MC_Dropout"

factory_params = {"factory_lag_st": 0, "factory_lag_lt": 0}


class TADKIT_UQModel(UQModel):
    def __init__(
        self,
        y_shape,
        n_window=20,
        model="RF_UQ",
        model_params=None,
        epochs=30,
        reduc_coef=True,
        beta=0.001,
        random_state=None,
        cache_manager=Cache_manager(),
    ):

        lag_values_params = {
            "lag": coefficients_spreaded(n_window),
            "deriv": [0, 1],
            "windows": [1, int(n_window / 4), n_window],
        }

        list_params_features = [lag_values_params]
        list_fit_features = [fit_pca, None]
        list_compute_features = [fit_compute_lag_values]

        targets_params = {"lag": [0]}
        list_params_targets = [targets_params]
        list_fit_targets = [None]
        list_compute_targets = [fit_compute_lag_values]

        Formalizer = Generic_Features_processor(
            name="Generic_Features_processor",
            cache=None,
            list_params_features=list_params_features,
            list_fit_features=list_fit_features,
            list_compute_features=list_compute_features,
            list_params_targets=list_params_targets,
            list_fit_targets=list_fit_targets,
            list_compute_targets=list_compute_targets,
            concat_features=True,
        )
        # Specification of the UQestimator
        X_, y_ = Formalizer.fit_transform(np.zeros((2 * n_window, y_shape)))
        X_shape = X_.shape[-1]

        Anom_param = {
            "type_norm": "Nsigma_global",
            "em" "beta": beta,
            "filt": [0.05, 0.15, 0.8, 0, 0],
            "min_cut": 0.05,
            "max_cut": 0.95,
            "var_min": 0.0001,
            "q_var": 0.7,
            "q_var_e": 0.7,
            "k_var_e": 1,
            "q_Eratio": 1,
            "d": 2,
            "global_median_normalization": False,
            "reduc_filter": None,
            "roll": 0,
            "with_born": False,
        }
        # PostProcesseur Instanciation that compute an epistemics lvl score
        Anom_proc = UQProc.Anomscore_processor(KPI_parameters=Anom_param)

        reduc_filter_pred = None
        if model == "RF_UQ":
            UQEstimator_initializer = RF_UQEstimator
            if model_params is None:
                estimator = RandomForestRegressor(
                    min_samples_leaf=5,
                    n_estimators=75,
                    max_depth=16,
                    ccp_alpha=0.00001,
                    max_samples=0.7,
                    max_features=0.75,
                    random_state=random_state,
                )

                UQEstimator_parameters = {
                    "estimator": estimator,
                    "var_min": 0.002,
                    "type_UQ": "var_A&E",
                    "rescale": True,
                    "random_state": random_state,
                }

        elif model == "MLP_MC_DROPOUT":
            type_output = "MC_Dropout"  # 'Deep_ensemble'
            UQEstimator_initializer = NN_UQ
            mlp_params = get_params_dict(X_shape, y_shape, type_output=type_output)
            trainig_params = get_training_parameters(
                epochs=[epochs], b_s=[64], l_r=[0.005], type_output=type_output
            )
            UQEstimator_initializer = NN_UQ
            UQEstimator_parameters = {
                "rescale": False,
                "model_initializer": mlp,
                "model_parameters": mlp_params,
                "training_parameters": trainig_params,
                "type_output": type_output,
                "random_state": random_state,
            }

        elif model == "MLP_DEEP_ENSEMBLE":
            type_output = "Deep_ensemble"
            UQEstimator_initializer = NN_UQ
            mlp_params = get_params_dict(X_shape, y_shape, type_output=type_output)
            trainig_params = get_training_parameters(
                epochs=[epochs], b_s=[64], l_r=[0.005], type_output=type_output
            )

            UQEstimator_parameters = {
                "rescale": False,
                "model_initializer": mlp,
                "model_parameters": mlp_params,
                "training_parameters": trainig_params,
                "type_output": type_output,
                "random_state": random_state,
            }

        UQ_proc = UQProc.UQKPI_Processor(
            KPI_parameters={
                "pred_and_UQ": True,
                "reduc_filter": reduc_filter_pred,
                "roll": 0,
                "reduc_filter": reduc_filter_pred,
            }
        )

        # Instanciation of the UQmodel modeling pipeline
        UQModels = super().__init__(
            UQEstimator_initializer,
            UQEstimator_parameters,
            preprocessor=Formalizer,
            name="UQModels",
            predictor=None,
            list_predict_KPI_processors=[UQ_proc],
            list_score_KPI_processors=[Anom_proc],
            reduc_filter=reduc_filter_pred,
            cache_manager=cache_manager,
            random_state=random_state,
        )
        return UQModels


class UQModel_KPI(UQModel):
    def __init__(
        self,
        UQEstimator_initializer,
        UQEstimator_params,
        # Necesite with_prediction or to provide a predictor.
        name="UQModel",
        predictor=None,
        preprocessor=None,
        list_alpha=[0.025, 0.16, 0.84, 0.975],
        list_percent_escore=[0.50, 0.80, 0.95, 0.98, 0.995, 1],
        reduc_filter_pred=None,
        reduc_filter_KPI=None,
        roll_KPI=1,
        anom_with_born=False,
        beta=0.01,
        var_min=0.001,
        cache_manager=Cache_manager(),
        q_Eratio=3,
        mode_epistemic_indicator="levels",
        random_state=None,
    ):

        UQ_proc = UQProc.UQKPI_Processor(
            KPI_parameters={
                "pred_and_UQ": True,
                "reduc_filter": reduc_filter_pred,
                "roll": 0,
            }
        )

        # PostProcesseur that compute Predictive intervals
        PIs_proc = UQProc.NormalPIs_processor(
            KPI_parameters={
                "list_alpha": list_alpha,
                "reduc_filter": reduc_filter_pred,
                "roll": 0,
            }
        )

        # PostProcesseur  that compute an epistemics lvl score
        Elvl_proc = UQProc.Epistemicscorelvl_processor(
            KPI_parameters={
                "list_percent": list_percent_escore,
                "reduc_filter": reduc_filter_pred,
                "roll": 0,
                "mode": mode_epistemic_indicator,
                "q_Eratio": q_Eratio,
                "var_min": var_min,
            }
        )

        # PostProcesseur Instanciation that compute an epistemics lvl score
        Anom_proc = UQProc.Anomscore_processor(
            KPI_parameters={
                "beta": beta,
                "d": 2,
                "var_min": var_min,
                "q_var": 1,
                "k_var_e": 1,
                "q_var_e": 1,
                "q_Eratio": 3,
                "filt": [0.05, 0.15, 0.8, 0, 0],
                "with_born": anom_with_born,
                "reduc_filter": reduc_filter_KPI,
                "roll": roll_KPI,
                "debug": False,
            }
        )

        list_predict_KPI_processors = [UQ_proc, PIs_proc, Elvl_proc]
        list_score_KPI_processors = [UQ_proc, Anom_proc, PIs_proc, Elvl_proc]

        super().__init__(
            UQEstimator_initializer=UQEstimator_initializer,
            UQEstimator_params=UQEstimator_params,
            name=name,
            predictor=predictor,
            preprocessor=preprocessor,
            list_predict_KPI_processors=list_predict_KPI_processors,
            list_score_KPI_processors=list_score_KPI_processors,
            reduc_filter=reduc_filter_pred,
            cache_manager=cache_manager,
            list_alpha=list_alpha,
            list_percent_escore=list_percent_escore,
            reduc_filter_pred=reduc_filter_pred,
            reduc_filter_KPI=reduc_filter_KPI,
            roll_KPI=roll_KPI,
            anom_with_born=anom_with_born,
            beta=beta,
            var_min=var_min,
            q_Eratio=q_Eratio,
            mode_epistemic_indicator=mode_epistemic_indicator,
            random_state=random_state,
        )


# Complex UQmodels link to EMS use case #################################################


class MultiDEEPUQModel(UQModel):
    """UQModel class for UQestimators that perform multisource forecast with UQ."""

    def __init__(
        self,
        UQEstimator_initializer,
        UQEstimator_params,
        list_sources,
        cut_params=(0.001, 0.999),
        multiscale_anomscore_params=None,
        name="MultiDEEPUQModel",
        cache_manager=Cache_manager(),
        save_result=True,
        save_models=True,
        with_generator=True,
        reduc_filter=None,
        random_state=None,
    ):
        """Instanciation of UQModels for UQestimators that perform multisource forecast with UQ
            # No predict_KPIs_processor
            # Multiscale_Anomscore_processor as score_predict_KPIS_processor :
            #   to product multidimensional anomaly score matrix


        Args:
            UQEstimator_initializer (obj_init): init method for instanciate an UQEstimator
            UQEstimator_params (dict_params): params for the init method
            list_sources (List_str): Name of source for storage purpose
            cut_params (tuple, optional): Lower and upper quantile to threshold data and remove outliers during
                learning phase. Defaults to (0.001, 0.999).
            multiscale_anomscore_params : params link to postprocessing.UQ_processor.Multiscale_Anomscore_processor
                -> to see defaults parameters.
            name (str, optional): Wrapper name . Defaults to 'MultiDEEPUQModel'.
            cache_manager (cache_manager, optional): cache_manager. Defaults to None.
        """
        self.futur_horizon = (
            UQEstimator_params["model_parameters"]["dim_horizon"]
            * UQEstimator_params["model_parameters"]["step"]
        )
        if multiscale_anomscore_params is None:
            multiscale_anomscore_params = {
                "type_norm": "Nsigma_local",
                "q_var": 1,
                "d": 2,
                "beta": 0.001,
                "beta_source": 0.001,
                "beta_global": 0.001,
                "min_cut": 0.002,
                "max_cut": 0.998,
                "per_seuil": 0.999,
                "reduc_filter": np.ones(self.futur_horizon),
                "roll": 1,
                "dim_chan": 1,
                "type_fusion": "mahalanobis",
                "filt": [0.1, 0.2, 0.5, 0.2, 0.1],
            }

        list_score_KPI_processors = [
            custom_UQProc.Multiscale_Anomscore_processor(
                KPI_parameters=multiscale_anomscore_params
            )
        ]

        super().__init__(
            UQEstimator_initializer,
            UQEstimator_params,
            name,
            predictor=None,
            list_predict_KPI_processors=[],
            list_score_KPI_processors=list_score_KPI_processors,
            cache_manager=cache_manager,
            save_result=save_result,
            save_models=save_models,
            random_state=random_state,
        )

        if reduc_filter is None:
            self.reduc_filter = np.zeros(self.futur_horizon) == 1
            self.reduc_filter[0] = True
        else:
            self.reduc_filter = reduc_filter

        self.with_generator = with_generator
        self.list_sources = list_sources
        self.list_chan = ["Mean", "Std", "Extremum", "Count"]
        self.cut_params = cut_params

    def multi_score_reshape(self, y=None, pred=None, UQ=None, mask=None):
        """Reshape from (n_sample,n_sources*s_chan) to List of n_sources (n_sample,n_chan) elements

        Args:
            y (np.array, optional): y to reshape or None. Defaults to None.
            pred (np.array, optional): pred to reshape or None. Defaults to None.
            UQ (np.array, optional): UQ to reshape or None. Defaults to None.
            return:
                list_y,list_pred,list_UQ
        """
        n_sources = len(self.list_sources)
        n_chan = len(self.list_chan)

        list_y = None
        if y is not None:
            size_data = len(y)
            list_y = [
                np.squeeze(y[:, mask, n_chan * i : n_chan * (i + 1)])
                for i in range(n_sources)
            ]

        # Issues : Reshape DL
        list_pred = None
        if pred is not None:
            size_data = len(pred)
            list_pred = [
                np.squeeze(pred[:, mask, n_chan * i : n_chan * (i + 1)])
                for i in range(n_sources)
            ]

        list_UQ = None
        if UQ is not None:
            # IF UQ measure is not composed
            if (len(UQ.shape) == 2) & (size_data == UQ.shape[1]):
                UQ = UQ[None]
            list_UQ = [
                np.squeeze(
                    np.array(
                        [
                            UQ_chan[:, mask, n_chan * i : n_chan * (i + 1)]
                            for UQ_chan in UQ
                        ]
                    )
                )
                for i in range(n_sources)
            ]

        return (list_y, list_pred, list_UQ)

    def fit(
        self, X, y, sample_weight=None, skip_UQEstimator=False, shuffle=True, **kwargs
    ):
        """Fit method that apply fit method of (predictor), UQEstimators, predict_KPI_processors, score_KPI_processors
        Args:
            X (np.array): Features
            y (np.array): Targets/observations
            sample_weight (np.array or None, optional): sample_weight. Defaults to None.
        """

        # Init deep learning model
        if not self.UQEstimator:
            self.UQEstimator = self.UQEstimator_initializer(**self.UQEstimator_params)

        self.type_UQ = self.UQEstimator.type_UQ
        self.type_UQ_params = None
        if hasattr(self.UQEstimator, "type_UQ_params"):
            self.type_UQ_params = self.UQEstimator.type_UQ_params

        set_ = len(y)
        train_deep = np.arange(set_) < (set_ * 4 / 6)
        np.invert(train_deep)

        # Call to factory method of model to transform features.
        if not (self.with_generator) & hasattr(self.UQEstimator, "factory"):
            train_deep = np.arange(set_) < (set_ * 4 / 6)
            inputs, targets, _ = self.UQEstimator.factory(
                X, y, train_deep, self.cut_params
            )

        else:
            inputs, targets = X, y

        # Call to fit method of model to perform multivarite fiting.
        if not skip_UQEstimator:
            self.UQEstimator.fit(
                inputs,
                targets,
                sample_weight=sample_weight,
                skip_format=True,
                generator=self.with_generator,
                shuffle=shuffle,
            )
            self.is_fitted = True

        if self.save_result:
            self.save()

        # Then call to fit for the post-processing procedure
        pred, UQ = self.UQEstimator.predict(
            inputs, skip_format=True, generator=self.with_generator
        )

        _, y_bis, _ = self.UQEstimator.factory(
            None, y, np.arange(len(y)), self.cut_params
        )

        y, pred, UQ = self.multi_score_reshape(
            y_bis, pred, UQ, mask=(np.ones(self.futur_horizon) == 1)
        )

        self._fit_score_KPI_processors(UQ, pred, y, **kwargs)

    def predict(self, X, name_save="output.p"):
        """Predict method that apply predictor and UQestimators predict method, then compute predict_KPIs by use
        transform methods of predict_KPI_Processors

        Args:
            X (np.array): feature
            name_save (str, optional): file_name for (Predictor) and UQEstimator outputs save file.
                Defaults to "output.p".

        Returns:
            pred, UQ : prediction and UQmeasure
        """
        if not (self.with_generator) & hasattr(self.UQEstimator, "factory"):
            inputs, _, _ = self.UQEstimator.factory(
                X, None, np.arange(len(X)), self.cut_params
            )

        else:
            inputs, _ = X, None

        pred, UQ = self.UQEstimator.predict(
            inputs, beta=0.1, skip_format=True, generator=self.with_generator
        )

        mask = np.zeros(self.futur_horizon)
        mask[0] = 1

        _, list_pred, list_UQ = self.multi_score_reshape(
            None, pred=pred, UQ=UQ, mask=(mask == 1)
        )

        if self.save_result:
            # Save in a multi_folder by source ways model outputs
            for n, source in enumerate(self.list_sources):
                query = {"name": source + "_" + name_save}
                self.cache_manager.save(query, (list_pred[n], list_UQ[n]))
        return pred, UQ

    def _recovers_KPI(self, mode, name_save="output.p", **kwargs):
        if mode == "output":
            if name_save is None:
                name_save = "output.p"
            query = {"name": name_save}
            loaded_data = self.cache_manager.load(query)

        if mode == "score":
            if name_save is None:
                name_save = "dict_res_anom.p"
            query = {"name": name_save}
            loaded_data = self.cache_manager.load(query)
        return loaded_data

    def score(self, X, y, **kwargs):
        """Score method that produce score KPI that transform y observation and (pred,UQ)
        UQestimator outputs into a KPI according to score_KPI_processors

        Args:
            X (np.array): Features
            y (np.array): Targets/obserbations

        Returns:
            list_KPI_output: (List_Pred,list_UQ),KPI_anom
        """

        pred, UQ = self.predict(X)
        _, y_bis, _ = self.UQEstimator.factory(
            None, y, np.arange(len(y)), self.cut_params
        )

        list_y, list_pred, list_UQ = self.multi_score_reshape(
            y=y_bis, pred=pred, UQ=UQ, mask=(np.ones(self.futur_horizon) == 1)
        )

        KPIs_anom = self._transform_score_KPI_processors(
            list_UQ, list_pred, list_y, **kwargs
        )

        S_anom_chan, S_anom_agg, S_anom_sensor, list_bot, list_top = KPIs_anom

        # Turn Multi-pred_horizon into

        list_y, list_pred, list_UQ = self.multi_score_reshape(
            y=y_bis, pred=pred, UQ=UQ, mask=self.reduc_filter
        )
        dict_res_anom = {
            "name": self.name,
            "S_anom_chan": S_anom_chan,
            "S_anom_agg": S_anom_agg,
            "S_anom_sensor": S_anom_sensor,
            "list_y": list_y,
            "list_pred": list_pred,
            "list_UQ": list_UQ,
            "list_bot": list_bot,
            "list_top": list_top,
        }

        if self.save_result:
            query = {"name": "dict_res_anom"}
            self.cache_manager.save(query, dict_res_anom)
        return (list_pred, list_UQ), KPIs_anom


# @To do refactoring as UQestimator:


class MultiUQModel(UQModel):
    """Instanciation of UQModels that apply an UQestimator for each sources and combine results for compute
    multidimensional anomaly score

    No predict_KPIs_processor
    Multiscale_Anomscore_processor as score_predict_KPIS_processor : to product multidimensional anomaly score matrix
    """

    def __init__(
        self,
        UQEstimator_initializer,
        UQEstimator_params,
        tunning_params,
        list_sources,
        list_delta=None,
        target_delta=None,
        multiscale_anomscore_params=None,
        models_in_ram=True,
        cut_params=(0.001, 0.999),
        name="Multi_RFUQModels",
        cache_manager=Cache_manager(),
        save_models=True,
        save_result=True,
        random_state=None,
    ):
        """Initialization of MultiUQModel class

        Args:
            UQEstimator_initializer (obj_init): init method for instanciate an UQEstimator
            UQEstimator_params (dict_params): params for the init method of the UQEstimator
            tunning_params (dict_grid_params): grid params for tuning UQEstiamtor using scikit
            list_sources (List_str): Name of source for storage purpose
            models_in_ram : specify if manipulate list of UQestimators in ram (heavy in ram), or store and load.
            cut_params (tuple, optional): Lower and upper quantile to threshold data and remove outliers during
                learning phase. Defaults to (0.001, 0.999).
            multiscale_anomscore_params : params link to postprocessing.UQ_processor.Multiscale_Anomscore_processor
                -> to see defaults parameters.
            name (str, optional): Wrapper name . Defaults to 'MultiDEEPUQModel'.
            cache_manager (cache_manager, optional): cache_manager. Defaults to None.
        """
        if multiscale_anomscore_params is None:
            multiscale_anomscore_params = {
                "type_norm": "Nsigma_local",
                "q_var": 1,
                "d": 2,
                "beta": 0.001,
                "beta_source": 0.001,
                "beta_global": 0.001,
                "min_cut": 0.002,
                "max_cut": 0.998,
                "per_seuil": 0.999,
                "type_fusion": "mahalanobis",
                "filt": [0.1, 0.2, 0.5, 0.2, 0.1],
            }

        list_score_KPI_processors = [
            custom_UQProc.Multiscale_Anomscore_processor(
                KPI_parameters=multiscale_anomscore_params
            )
        ]

        super().__init__(
            UQEstimator_initializer,
            UQEstimator_params,
            name,
            list_score_KPI_processors=list_score_KPI_processors,
            cache_manager=cache_manager,
            save_models=save_models,
            save_result=save_result,
            random_state=random_state,
        )
        self.tunning_params = tunning_params
        self.list_sources = list_sources
        self.list_delta = list_delta
        self.target_delta = target_delta
        self.UQEstimators_in_ram = models_in_ram
        self.cut_params = cut_params
        if self.UQEstimators_in_ram:
            self.list_UQEstimators = []
            for _ in list_sources:
                self.list_UQEstimators.append([])

    def get_model(self, id_source):
        """Get model of an id_source

        Args:
            model (UQEstimator): UQEstimator to set
            id_source (int): Id_source
        """
        print("get model", id_source)
        if self.UQEstimators_in_ram:
            idx = self.list_sources.index(id_source)
            print("get model", idx)
            model = self.list_UQEstimators[idx]
        else:
            query = {"name": id_source + "_" + self.name}
            model = self.cache_manager.load(query)
        return model

    def set_model(self, model, id_source):
        """Set model of an id_source

        Args:
            model (UQEstimator): UQEstimator to set
            id_source (int): Id_source
        """
        if self.UQEstimators_in_ram:
            idx = self.list_sources.index(id_source)
            self.list_UQEstimators[idx] = model
        else:
            query = {"name": id_source + "_" + self.name}
            self.cache_manager.save(query, model)
            del model

    def _aux_fit(
        self,
        X,
        y,
        sample_weight=None,
        train=None,
        skip_UQEstimator=False,
        source=None,
        **kwargs
    ):
        """Fit method that apply fit method of (predictor) for source, the
        Args:
            X (np.array): Features
            y (np.array): Targets/observations
            sample_weight (np.array or None, optional): sample_weight. Defaults to None.
        """

        if not skip_UQEstimator:
            # Instanciate model
            model = self.UQEstimator_initializer(**self.UQEstimator_params)

            # Store type_UQ
            self.type_UQ = model.type_UQ
            self.type_UQ_params = None
            if hasattr(model, "type_UQ_params"):
                self.type_UQ_params = model.type_UQ_params

            if train is None:
                train = np.ones(len(X)) == 1
            # Run tunning if having gridsearch params
            if self.tunning_params is not None:
                model._tuning(
                    X[train], y[train], n_esti=100, folds=4, params=self.tunning_params
                )

            print("Fitting model source :", source)
            y = cut(y, self.cut_params[0], self.cut_params[1])
            model.fit(X[train], y[train], sample_weight=sample_weight[train], **kwargs)

            self.set_model(model, source)

    def _recovers_all_results(self, mode):
        """auxiliar function aim to load stored output for each source_UQEstimators

        Args:
            mode (str): specify 'output' to load output, score to load Anom kpi

        Returns:
            results: List_pred,List_UQ if output, Anom-KPIs if 'score'
        """

        if mode == "output":
            # recover pred,output for each source_UQestimator

            list_outputs = [
                self.cache_manager.load({"name": source + "_output"})
                for source in self.list_sources
            ]
            loaded_data = (
                [output[0] for output in list_outputs],
                [output[1] for output in list_outputs],
            )
            del list_outputs

        if mode == "score":
            query = {"name": "dict_res_anom"}
            loaded_data = self.cache_manager.load(query)

        return loaded_data

    def fit(self, X, y, sample_weight=None, skip_UQEstimator=False, **kwargs):
        """Fit method that apply fit method of (predictor) for eachs source_UQEstimators,
        then fit the Anom_KPI_processors.

        Args:
            X (np.array): Features
            y (np.array): Targets/observations
            sample_weight (np.array or None, optional): sample_weight. Defaults to None.
        """
        for n, (X_, y_) in enumerate(zip(X, y)):
            source = self.list_sources[n]
            if sample_weight is not None:
                sample_weight_ = sample_weight[n]
                self._aux_fit(
                    X_,
                    y_,
                    sample_weight_,
                    train=None,
                    skip_UQEstimator=skip_UQEstimator,
                    source=source,
                    **kwargs
                )
            del X_, y_
        self.is_fitted = True
        list_pred, list_UQ = self.predict(X, recovers_at_end=False, **kwargs)
        self._fit_score_KPI_processors(list_UQ, list_pred, y, **kwargs)
        if self.save_models:
            self.save()

    def fit_with_generator(
        self, data_generator, skip_UQEstimator=False, shuffle=True, **kwargs
    ):
        """Specific fit method that handle data_generator

        Args:
            data_generator (datagenerator): Iterative object that provide : X, y, sample_weight, x_split, context,
                objective, source (see data_generator)
            shuffle (bool, optional): use shuffle or not. Defaults to True.
        """
        list_y = []
        list_UQ = []
        list_pred = []
        for _n, dataset in enumerate(data_generator):
            # Recovers data from data_generator
            X, y, sample_weight, x_split, _context, _objective, source = dataset
            train = None
            if x_split is not None:
                train = x_split == 1
                X, y = apply_mask(X, train), apply_mask(y, train)
                if sample_weight is not None:
                    sample_weight = sample_weight[train]
            # Called fit procedure for each dataset
            self._aux_fit(
                X,
                y,
                sample_weight=sample_weight,
                train=None,
                skip_UQEstimator=skip_UQEstimator,
                source=source,
                **kwargs
            )
            pred, UQ = self._aux_predict(X, source, **kwargs)
            list_y.append(y)
            list_pred.append(pred)
            list_UQ.append(UQ)
        self._fit_score_KPI_processors(list_UQ, list_pred, list_y, **kwargs)

    def _aux_predict(self, X, source, **kwargs):
        model = self.get_model(source)

        pred, UQ = model.predict(X, **kwargs)
        del model

        # Store results
        if self.save_result:
            query = {"name": source + "_output.p"}
            self.cache_manager.save(query, (pred, UQ))

        return pred, UQ

    def predict(self, X, recovers_at_end=False, **kwargs):
        """Predict method that applies predictor and UQestimators predict method, then computes predict_KPIs
        by using transform methods of predict_KPI_Processors.

        Args:
            X (np.array): feature
            name_save (str, optional): file_name for (Predictor) and UQEstimator outputs save file.
                Defaults to "output.p".

        Returns:
            pred, UQ : prediction and UQmeasure
        """
        list_pred = []
        list_UQ = []
        for n, X_ in enumerate(X):
            source = self.list_sources[n]
            pred, UQ = self._aux_predict(X_, source, **kwargs)
            if not recovers_at_end:
                list_pred.append(pred)
                list_UQ.append(UQ)
            del pred, UQ, X_

        if recovers_at_end:
            pred, UQ = self._recovers_at_end("output")

        return (pred, UQ)

    def predict_with_generator(self, data_generator, recovers_at_end=True, **kwargs):
        list_pred = []
        list_UQ = []
        for _n, (X, _, _, _, _, _, source) in enumerate(data_generator):
            # Recovers data from data_generator
            output = self._aux_predict(X, source, **kwargs)
            if not (recovers_at_end):
                list_pred.append(output[0])
                list_UQ.append(output[1])
            del output, X

        if recovers_at_end:
            list_pred, list_UQ = self._recovers_all_results("output")
        return (list_pred, list_UQ)

    def _aux_score(self, list_y, list_pred, list_UQ, **kwargs):
        if (self.list_delta is not None) & (self.target_delta is not None):
            for n in np.arange(len(list_y)):
                if not self.list_delta[n] == self.target_delta:
                    current_delta = pre_struc.check_delta(self.list_delta[n])
                    target_delta = pre_struc.check_delta(self.target_delta)
                    ratio = current_delta / target_delta

                    # Scale différent entre capteurs
                    x = np.arange(0, list_y[n])
                    x_new = np.arange(0, len(list_y[n]), ratio)

                    list_y[n] = pre_pre.interpolate(
                        x=x, y=list_y[n], xnew=x_new, type_interpolation="previous"
                    )[1]

                    list_pred[n] = pre_pre.interpolate(
                        x=x, y=list_pred[n], xnew=x_new, type_interpolation="previous"
                    )[1]

                    UQ_temp = [
                        pre_pre.interpolate(
                            x=x, y=UQ_part, xnew=x_new, type_interpolation="previous"
                        )[1]
                        for UQ_part in list_UQ
                    ]
                    if len(UQ_temp) == 1:
                        UQ_temp = UQ_temp[0]

                    list_UQ[n] = UQ_temp

        KPIs_anom = self._transform_score_KPI_processors(
            list_UQ, list_pred, list_y, **kwargs
        )
        S_anom_chan, S_anom_agg, S_anom_sensor, list_bot, list_top = KPIs_anom

        dict_res_anom = {
            "name": self.name,
            "list_y": list_y,
            "list_pred": list_pred,
            "list_UQ": list_UQ,
            "S_anom_chan": S_anom_chan,
            "S_anom_agg": S_anom_agg,
            "S_anom_sensor": S_anom_sensor,
            "list_bot": list_bot,
            "list_top": list_top,
        }

        if self.cache_manager is not None:
            query = {"name": "dict_res_anom"}
            self.cache_manager.save(query, dict_res_anom)

        return (S_anom_chan, S_anom_agg, S_anom_sensor)

    def score(self, X, y, **kwargs):
        """Score method that produces score KPI by transforming y observation and (pred,UQ) UQestimator outputs
        into a KPI according to score_KPI_processors.

        Args:
            X (np.array): Features
            y (np.array): Targets/obserbations

        Returns:
            list_KPI_output: list_KPI_output or KPI_ouput if len(list)==1
        """
        list_pred, list_UQ = self.predict(X)
        KPI_anom = self._aux_score(y, list_pred, list_UQ)
        return (list_pred, list_UQ), KPI_anom

    def score_with_generator(self, data_generator, recovers_at_end=True, **kwargs):
        list_y = []
        list_pred = []
        list_UQ = []
        for _n, (X, y, _, _, _, _, source) in enumerate(data_generator):
            list_y.append(y)
            output = self._aux_predict(X, source, **kwargs)
            if not recovers_at_end:
                list_pred.append(output[1])
                list_UQ.append(output[2])
            del output, X, y

        if recovers_at_end:
            list_pred, list_UQ = self._recovers_all_results("output")

        KPI_anom = self._aux_score(list_y, list_pred, list_UQ)
        return (list_pred, list_UQ), KPI_anom
