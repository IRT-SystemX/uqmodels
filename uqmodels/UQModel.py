###################################################################################
# UQModel : "Model than apply an modeling pipeline composed of an predictor(optional), UQestimator and KPI
# To write a custom UQModel, inherite from UQModel class an use supoer().__init__()

# Class UQmodels : from an UQestimators : perform dadada

import inspect
import os

import numpy as np
from sklearn.base import BaseEstimator

import uqmodels.modelization.ML_estimator.random_forest_UQ as RF_UQ
import uqmodels.postprocessing.UQKPI_Processor as UQProc
import uqmodels.preprocessing.preprocessing as pre_pre
import uqmodels.preprocessing.structure as pre_struc
from uqmodels.processing import Cache_manager
from uqmodels.utils import add_random_state, apply_mask, apply_middledim_reduction, cut


class UQModel(BaseEstimator):
    """
    UQModel class : instanciate a pipeline of model estimation, model UQ estimation and post processing as
    a (Fit,Predict,Score) triplet of method
    """

    def __init__(
        self,
        UQEstimator_initializer=RF_UQ.RF_UQEstimator,
        UQEstimator_params=RF_UQ.get_params_dict(),
        # Necesite with_prediction or to provide a predictor.
        name="UQModel",
        predictor=None,
        preprocessor=None,
        list_predict_KPI_processors=None,  # List of UQProcessor
        list_score_KPI_processors=None,
        reduc_filter=None,
        roll=None,
        cache_manager=Cache_manager(),
        save_result=False,
        save_models=False,
        random_state=None,
        **kwargs
    ):
        """Initialize UQModel class

        Args:
            UQEstimator_initializer (obj_init): init method for instanciate an UQEstimator
            UQEstimator_params (dict_params): params for the init method
            name (str, optional): Wrapper name . Defaults to 'UQModel'.
            predictor (predictor or None, optional): Predictor instanciate if None,
                assume that UQEstiamtor also make prediction. Defaults to None.
            list_predict_KPI_processors (list_processor or None, optional): List of instanciated
                predict_KPI_processors. Defaults to None.
            list_score_KPI_processors (list_processor or None, optional): List of instanciated
                score_KPI_processors. Defaults to None
            reduc_filter (np.array or None): Weigth ponderation for reduc middle dimension (if needed)
            cache_manager (cache_manager or None, optional): Cache manager. Defaults to None.
            random_state : Controls randoms
        """

        self.name = name

        self.predictor = predictor
        if predictor is not None:
            self.predictor_initializer = predictor.__init__()
            self.predictor_params = predictor.get_params()

        self.UQEstimator_initializer = UQEstimator_initializer
        self.preprocessor = preprocessor
        self.UQEstimator_params = UQEstimator_params
        self.list_predict_KPI_processors = list_predict_KPI_processors
        self.list_score_KPI_processors = list_score_KPI_processors
        self.reduc_filter = reduc_filter
        self.roll = roll
        self.cache_manager = cache_manager
        self.random_state = random_state
        if self.random_state is not None:
            self.UQEstimator_params["random_state"] = random_state

            for n, predict_KPI_processor in enumerate(list_predict_KPI_processors):
                predict_KPI_processor.random_state = add_random_state(random_state, n)

            for n, score_KPI_processor in enumerate(list_score_KPI_processors):
                score_KPI_processor.random_state = add_random_state(
                    random_state, 10 + n
                )

        self.UQEstimator = None
        self.is_fitted = None
        self.type_UQ = None
        self.type_UQ_params = None
        self.save_result = save_result
        self.save_models = save_models

        for key_arg in kwargs.keys():
            setattr(self, key_arg, kwargs[key_arg])

    def _load(self, entity, path, name):
        """Auxiliar load procedure for an specific entity : try to use entity.load method
        or force save using UQModel cache_manager

        Todo:
            * Raise fail error -> part already handle by cache manager

        Args:
            entity (obj): Object to load
            path (str): Path to load
            name (str): name to load
        """
        if hasattr(entity, "load"):
            arg_list = list(inspect.signature(entity.load).parameters)
            if "name" in arg_list:
                entity.load(path, name=name)
            else:
                path = os.path.join(path, name)
                entity.load(path)
        else:
            query = {"storing": path, "name": name}
            entity = self.cache_manager.load(query)
        return entity

    def load(self, path=None, name="UQModel"):
        """UQ model load procedure : recovers UQ model structure, then predictor (if existe), then UQestimators

        Todo:
            * Raise fail error -> part already handle by cache manager

        Args:
            path (str, optional): path of UQmodels save if none use path store in cache manager
            name (str, optional): name of UQmodels to load uf none use UQModel name
        """
        if path is None:
            path = self.cache_manager.storing

        query = {"storing": path, "name": name}
        _UQModel = self.cache_manager.load(query)

        for key in _UQModel.__dict__.keys():
            self.__setattr__(key, _UQModel.__getattribute__(key))

        if self.predictor:
            self.predictor = self.predictor_initializer(**self.predictor_params)
            self.predictor = self._load(self.predictor, path, "predictor")

        if self.UQEstimator:
            self.UQEstimator = self.UQEstimator_initializer(**self.UQEstimator_params)
            self.UQEstimator = self._load(self.UQEstimator, path, "UQEstimator")

        # TO do :
        # Specific load procedure for predict_KPI_processors and score_KPI_processors
        # Specific load procedure for preprocessor object
        # Hold currently by global __dict__ parameters save

    def _save(self, entity, path, name, UQModel_name=None):
        """Auxiliar save procedure for an specific entity : try to use entity.save
        method or force save using UQmMdels cache_manager

        Todo:
            * Raise fail error -> part already handle by cache manager

        Args:
            entity (obj): Object to save
            path (str): Path to save
            name (str): name to save
        """
        if UQModel_name is None:
            UQModel_name = self.name

        if hasattr(entity, "save"):
            arg_list = list(inspect.signature(entity.save).parameters)
            if "name" in arg_list:
                path = os.path.join(path, UQModel_name)
                entity.save(path, name=name)
            else:
                path = os.path.join(path, UQModel_name, name)
                entity.save(path)
        else:
            query = {"storing": path, "keys": [UQModel_name, name]}
            if self.cache_manager is not None:
                self.cache_manager.save(query, entity)
            else:
                print("No_cache_manager : cannot save")

    def save(self, path=None, name=None):
        """UQ model save procedure : recovers UQ model structure, then predictor (if existe), then UQestimators

        Todo:
            * Raise fail error

        Args:
            path (path, optional): path of UQmodels save
            name (name, optional): name of UQmodels to load
        """
        if name is None:
            name = self.name

        if path is None:
            path = self.cache_manager.storing

        if self.predictor is not None:
            self._save(self.predictor, path, "predictor", UQModel_name=name)
            tmp_predictor = self.predictor
            self.predictor = True

        if self.UQEstimator:
            self._save(self.UQEstimator, path, "UQEstimator", UQModel_name=name)
            tmp_UQestimator = self.UQEstimator
            self.UQEstimator = True

        # TO do :
        # Specific save procedure for predict_KPI_processors and score_KPI_processors
        # Specific save procedure for preprocessor object
        # Hold currently by global __dict__ parameters save

        query = {"storing": path, "keys": [name, "UQModel"]}
        self.cache_manager.save(query, self)

        if self.predictor is not None:
            self.predictor = tmp_predictor

        if self.UQEstimator:
            self.UQEstimator = tmp_UQestimator

    def _init_UQEstimator(self, X=None, y=None):
        if not self.UQEstimator:
            self.UQEstimator = self.UQEstimator_initializer(**self.UQEstimator_params)

        if hasattr(self.UQEstimator, "factory"):
            _ = self.UQEstimator.factory(X, y, only_fit_scaler=True)

        self.type_UQ = self.UQEstimator.type_UQ
        if hasattr(self.UQEstimator, "type_UQ_params"):
            self.type_UQ_params = self.UQEstimator.type_UQ_params

    def _fit_predict_KPI_processors(self, UQ, pred, y, **kwargs):
        """Auxiliar method that apply fit method of predict_KPI_processors

        Args:
            UQ (np.array): Features
            pred (np.array): Prediction from predictor or UQEstimator
            y (np.array): Targets/observations
        """
        for processor in self.list_predict_KPI_processors:
            processor.fit(
                UQ,
                self.type_UQ,
                pred,
                y=y,
                type_UQ_params=self.type_UQ_params,
                **kwargs
            )

    def _fit_score_KPI_processors(self, UQ, pred, y, **kwargs):
        """Auxiliar method that apply fit method of score_KPI_processors

        Args:
            UQ (np.array): Features
            pred (np.array): Prediction from predictor or UQEstimator
            y (np.array): Targets/observations
        """
        for processor in self.list_score_KPI_processors:
            processor.fit(
                UQ, self.type_UQ, pred, y, type_UQ_params=self.type_UQ_params, **kwargs
            )

    def fit(self, X, y=None, sample_weight=None, skip_UQEstimator=False, **kwargs):
        """Fit method that apply fit method of (predictor), UQEstimators, predict_KPI_processors, score_KPI_processors
        Args:
            X (np.array): Features
            y (np.array): Targets/observations
            sample_weight (np.array or None, optional): sample_weight. Defaults to None.
        """

        if self.preprocessor is not None:
            self.preprocessor.fit(X)
            X, y = self.preprocessor.transform(X)

        y_bis = y

        # --- Fit predictor if it's not direclty the UQEstimator
        if self.predictor is not None:
            if hasattr(
                self.predictor, "is_fitted"
            ):  # To do replace with sklearn is_fitted check
                pred = self.predictor.fit(X)
                y_bis = y - pred

            if self.save_models:
                self.save()

            if "res" not in self.type_UQ:
                y_bis = y
            else:
                print(
                    "Warning use predictor with non residual base UQestimator > UQMeasure"
                    " will not based on model output"
                )

        # Call to fit method of model to perform multivarite fiting.

        self._init_UQEstimator(X, y_bis)

        # ------
        # Format data
        if hasattr(self.UQEstimator, "factory"):  # Apply factory to transform input
            generator = False
            if (
                (hasattr(self.UQEstimator, "training_parameters"))
                and ("generator" in self.UQEstimator.training_parameters.keys())
                and (self.UQEstimator.training_parameters["generator"])
            ):
                # Apply factory by batch during generator unfolding
                generator = self.UQEstimator.training_parameters["generator"]
                X_norm, y_bis_norm = X, y_bis

                # But recover totality of model target to provide it to KPI-Score-KPIProcessor
                _, y_bis, _ = self.UQEstimator.factory(None, y_bis, "transform")

            else:  # Apply factory
                X_norm, y_bis_norm, w_ = self.UQEstimator.factory(X, y_bis, "transform")
                y_bis = y_bis_norm

            # Unormalized model target that can be multi-horizon object
            _, y_bis = self.UQEstimator._format(None, y_bis, "inverse_transform")

        else:  # Data normalised is row data
            X_norm, y_bis_norm = X, y_bis
        # ---

        if not (skip_UQEstimator):
            self.UQEstimator.fit(
                X_norm,
                y_bis_norm,
                sample_weight,
                skip_format=True,
                generator=generator,
                **kwargs
            )
            self.is_fitted = True

        pred = None
        if self.predictor is not None:
            pred = self.predictor.predict(X)

        # Call to "fit" function of processor
        pred_bis, UQ = self.UQEstimator.predict(
            X_norm, skip_format=True, generator=generator, **kwargs
        )

        if pred is None:
            pred = pred_bis

        # Side effect fit KPI processor
        self._fit_predict_KPI_processors(UQ, pred, y_bis, **kwargs)
        self._fit_score_KPI_processors(UQ, pred, y_bis, **kwargs)

    def fit_with_generator(self, data_generator, shuffle=True, **kwargs):
        """Specific fit method that handle data_generator

        Args:
            data_generator (datagenerator): Iterative object that provide : X, y, sample_weight, x_split, context,
                objective, source (see data_generator)
            shuffle (bool, optional): use shuffle or not. Defaults to True.
        """
        for _n, dataset in enumerate(data_generator):
            # Recovers data from data_generator
            X, y, sample_weight, x_split, _context, _objective, _source = dataset
            train = None
            if x_split is not None:
                train = x_split == 1
                X, y = apply_mask(X, train), apply_mask(y, train)
                if sample_weight is not None:
                    sample_weight = sample_weight[train]
            # Called fit procedure for each dataset
            self.fit(X, y, sample_weight=sample_weight, shuffle=shuffle, **kwargs)

    def _transform_predict_KPI_processors(self, UQ, pred, **kwargs):
        """Auxiliar method that apply transform method of predict_KPI_processors

        Args:
            UQ (np.array): UQmeasure from an UQestimator
            pred (np.array): Predictor from an predictor or an UQestimator

        Returns:
            list_KPI_output: list_KPI_output pr KPI if len(list==1)
        """
        list_KPI_output = []
        for processor in self.list_predict_KPI_processors:
            KPI_output = processor.transform(
                UQ=UQ,
                type_UQ=self.type_UQ,
                pred=pred,
                y=None,
                type_UQ_params=self.type_UQ_params,
                **kwargs
            )

            list_KPI_output.append(KPI_output)

        if len(list_KPI_output) == 1:
            list_KPI_output = list_KPI_output[0]

        return list_KPI_output

    def predict(self, X, name_save="output.p", **kwargs):
        """Predict method that apply predictor and UQestimators predict method, then compute predict_KPIs by use
        transform methods of predict_KPI_Processors

        Args:
            X (np.array): feature
            name_save (str, optional): file_name for (Predictor) and UQEstimator outputs save file.
                Defaults to "output.p".

        Returns:
            pred, list_KPI_output : prediction and list of KPIs or KPI if len(list==1)
        """

        if self.preprocessor is not None:
            X, _ = self.preprocessor.transform(X, training=False)

        # --- Run predictor if it's not direclty the UQEstimator
        pred = None
        if self.predictor is not None:
            pred = self.predictor.predict(X)
            if self.save_result:
                query = {"name": name_save + "_predictor"}
                self.cache_manager.save(query, pred)

        # --- Format data
        if hasattr(self.UQEstimator, "factory"):

            X_norm, _, _ = self.UQEstimator.factory(X, None)

            generator = False
            if (
                (hasattr(self.UQEstimator, "training_parameters"))
                and ("generator" in self.UQEstimator.training_parameters.keys())
                and (self.UQEstimator.training_parameters["generator"])
            ):
                generator = True
                X_norm = X
            else:
                generator = False
                X_norm, _, _ = self.UQEstimator.factory(X, None)

        else:
            X_norm = X
        # ---

        pred_bis, UQ = self.UQEstimator.predict(
            X_norm, skip_format=True, generator=generator, **kwargs
        )

        if pred is None:
            pred = pred_bis

        list_KPI_output = self._transform_predict_KPI_processors(
            UQ=UQ, pred=pred, **kwargs
        )

        if self.reduc_filter is not None:
            pred = apply_middledim_reduction(
                pred, reduc_filter=self.reduc_filter, roll=self.roll
            )

        if self.save_result:
            query = {"name": name_save + "_UQEstimator"}
            self.cache_manager.save(query, (pred, UQ))
            query = {"name": name_save + "_predict_KPI_processors"}
            self.cache_manager.save(query, list_KPI_output)

        return pred, list_KPI_output

    def predict_with_generator(self, data_generator, **kwargs):
        """specific predict method that handle data_generator

        Args:
            data_generator (datagenerator): Iterative object that provide : X, y, sample_weight, x_split, context,
                objective, source (see data_generator)

        Returns:
             list of (pred, UQ)
        """
        list_pred = []
        for _n, dataset in enumerate(data_generator):
            # Recovers data from data_generator
            X, _y, _sample_weight, _x_split, _context, _objective, source = dataset
            list_pred.append(self.predict(X, name_save=source + "_output.p", **kwargs))
        return list_pred

    def _transform_score_KPI_processors(self, UQ, pred, y, **kwargs):
        """auxilliair method that apply transform method of the score_KPI_processors

        Args:
            UQ (np.array): UQmeasure from an UQestimator
            pred (np.array): Prediction for an predictot or UQestimator
            y (np.array): Targets/Observation

        Returns:
            list_KPI_output: list_KPI_output or KPI_ouput if len(list)==1
        """

        list_KPI_output = []
        for processor in self.list_score_KPI_processors:
            list_KPI_output.append(
                processor.transform(
                    UQ,
                    self.type_UQ,
                    pred,
                    y,
                    type_UQ_params=self.type_UQ_params,
                    **kwargs
                )
            )

        if len(list_KPI_output) == 1:
            list_KPI_output = list_KPI_output[0]

        return list_KPI_output

    def score(self, X, y=None, name_save="output", **kwargs):
        """Score method that produce score KPI that transform y observation and (pred,UQ) UQestimator outputs
        into a KPI according to score_KPI_processors

        Args:
            X (np.array): Features
            y (np.array): Targets/obserbations

        Returns:
            list_KPI_output: list_KPI_output or KPI_ouput if len(list)==1
        """
        if self.preprocessor is not None:
            X, y = self.preprocessor.transform(X)

        y_bis = y

        # Run predictor if distinct of the UQEstimator
        pred = None
        if self.predictor is not None:
            # To do replace with sklearn is_fitted check
            if hasattr(self.predictor, "is_fitted"):
                pred = self.predictor.predict(X)
                y_bis = y - pred

            if self.save_result:
                query = {"name": name_save + "_predictor"}
                self.cache_manager.save(query, pred)

            if "res" not in self.type_UQ:
                y_bis = y
            else:
                print(
                    "Warning use predictor with non residual base UQestimator > UQMeasure"
                    " will not based on model output"
                )

        # ---
        # Format data for UQEstimator distinct mode : With or without factory + with or without generator
        if hasattr(self.UQEstimator, "factory"):  # Apply factory to transform input
            generator = False
            if (
                (hasattr(self.UQEstimator, "training_parameters"))
                and ("generator" in self.UQEstimator.training_parameters.keys())
                and (self.UQEstimator.training_parameters["generator"])
            ):
                # Apply factory by batch during generator unfolding
                generator = self.UQEstimator.training_parameters["generator"]
                X_norm, y_bis_norm = X, y

                # But recover totality of model target to provide it to KPI-Score-KPIProcessor
                _, y_bis, _ = self.UQEstimator.factory(None, y_bis, "transform")

            else:  # Apply factory
                X_norm, y_bis_norm, w_ = self.UQEstimator.factory(X, y_bis, "transform")
                y_bis = y_bis_norm

            # Unormalized model target that can be multi-horizon object
            _, y_bis = self.UQEstimator._format(None, y_bis, "inverse_transform")

        else:  # Data normalised is row data
            X_norm, y_bis_norm = X, y_bis
        # ---

        pred_bis, UQ = self.UQEstimator.predict(
            X_norm, skip_format=True, generator=generator, **kwargs
        )

        if pred is None:
            pred = pred_bis

        list_KPI_output = self._transform_score_KPI_processors(
            UQ, pred, y_bis, **kwargs
        )
        if self.save_result:
            query = {"name": name_save + "_UQEstimator"}
            self.cache_manager.save(query, (pred_bis, UQ))
            query = {"name": name_save + "_score_KPI_processors"}
            self.cache_manager.save(query, list_KPI_output)
        return list_KPI_output

    def score_with_generator(self, data_generator, **kwargs):
        """specitic score method that handle data_generator

        Args:
            data_generator (datagenerator): Iterative object that provide : X, y, sample_weight, x_split, context,
                objective, source (see data_generator)

        Returns:
            list of score_KPI or score_KPI if len(list)==1
        """
        list_res = []
        n_req = 0
        for _n, (X, y, _, _, _, _, _source) in enumerate(data_generator):
            res = self.score(X, y, **kwargs)
            list_res.append(res)
            n_req += 1

        if n_req == 1:
            list_res = list_res[0]

        return list_res
