#####################################################################################
# Source link to Preprocessor class :
# Preprocessing pipeline can combine several Preprocessing.
# We suggest to split : Raw_data -> (Raw_data_preprossing) -> Clean_data -> (ML-Preprocessor_Porcessing) -> ML-Dataset
# Then we can produce from a same clean_data several ML-Dataset
# Cache mecanism aim to avoid to do same preprocessing calculation


import numpy as np

import uqmodels.preprocessing.features_processing as FE_proc
from uqmodels.preprocessing.Preprocessor import Preprocessor


class dict_to_TS_Dataset(Preprocessor):
    def __init__(self, name="dict_to_TS_Dataset"):
        """Init Preprocessor that turn a dict_data that contains all preprocessed info into a dataset"""
        super().__init__(name=name)

    def fit(self, data, query):
        """Do nothing"""
        super().fit(data, query)

    def transform(self, data, query):
        """Provide dataset as list of array : [X,y,context,train,test,X_split]"""
        data = super().transform(data, query)
        X = data["X"]

        y = data["Y"].reshape(len(X), -1)

        context = None
        if "context" in data.keys():
            context = data["context"]

        sample_weight = None
        if "sample_weight" in data.keys():
            sample_weight = data["sample_weight"]

        objective = None
        if "objective" in data.keys():
            objective = data["objective"]

        name = "data"
        if "name" in query.keys():
            name = query["name"]

        x_split = np.zeros(len(X))
        if "X_split" in data.keys():
            x_split = data["X_split"]

        if "train" in data.keys():
            x_split = data["train"]

        return (X, y, sample_weight, x_split, context, objective, name)


class Generic_Features_processor(Preprocessor):
    def __init__(
        self,
        name="Generic_Features_processor",
        cache=None,
        structure=None,
        update_query=None,
        list_params_features=[],
        list_fit_features=[],
        list_compute_features=[],
        list_update_params_features=None,
        list_params_targets=[],
        list_fit_targets=[],
        list_compute_targets=[],
        list_update_params_targets=None,
        normalise_data=False,
        normalise_context=False,
        dataset_formalizer=None,
        min_size=1,
        concat_features=False,
        concat_targets=True,
        **kwargs
    ):
        """Preprocessor class (inherit from Processor) : that aim to preprocess data in a (fit/transform) scheme
        and hold a cache manager functionality to save/load object

        Args:
            name (str, optional): Name of processor. Defaults to 'processor'.
            cache (Cache_manager or None, optional): Cache manager. Defaults to None : no save/load procedure
            structure (obj or None): structure that contains specification about how data has to be structured
                after preprocessing
            update_query (function, optional): Function to update query due to Processor application if needed.
                Defaults to default_update_query : no update/
            fit_function = function to apply in fit procedure. Defaults to fit_default that does nothing.
            transform_function = function to apply in tranform procedure. Defaults to transform_default that
                does nothing.

        """

        super().__init__(
            name=name,
            cache=cache,
            structure=structure,
            update_query=update_query,
            list_params_features=list_params_features,
            list_fit_features=list_fit_features,
            list_compute_features=list_compute_features,
            list_update_params_features=list_update_params_features,
            list_params_targets=list_params_targets,
            list_fit_targets=list_fit_targets,
            list_compute_targets=list_compute_targets,
            list_update_params_targets=list_update_params_targets,
            normalise_data=normalise_data,
            normalise_context=normalise_context,
            dataset_formalizer=dataset_formalizer,
            min_size=min_size,
            concat_features=concat_features,
            concat_targets=concat_targets,
            **kwargs
        )

    def fit(self, data, query={}, **kwargs):
        """Fit Preprocessing using data and fit_function procedure

        Args:
            data (obj, optional): data. Defaults to None.
            query: dict_query that generated the data
            save_formaliser (bool, optional): boolean flag that inform if we have to save preprocessor or not
        """
        if len(data) == 2:
            data, context = data
            if self.normalise_context:
                context, context_scaler = FE_proc.normalise_panda(
                    context, mode="fit_transform"
                )
                self.context_scaler = context_scaler
        else:
            context = None

        if self.normalise_data:
            data, data_scaler = FE_proc.normalise_panda(data, mode="fit_transform")
            self.data_scaler = data_scaler

        for n, (params, fit_func) in enumerate(
            zip(self.list_params_features, self.list_fit_features)
        ):
            if fit_func is not None:
                if self.list_update_params_features is not None:
                    if self.list_update_params_features[n] is not None:
                        params = self.list_update_params_features[n](query, params)
                params["params_"] = fit_func(data, context, **params)

        for n, (params, fit_func) in enumerate(
            zip(self.list_params_targets, self.list_fit_targets)
        ):
            if fit_func is not None:
                if self.list_update_params_targets is not None:
                    if self.list_update_params_targets[n] is not None:
                        params = self.list_update_params_targets[n](query, params)
                params["params_"] = fit_func(data, context, **params)
        return super().fit(data)

    def transform(self, data, query={}, training=True, **kwarg):
        """Apply transform_function to data
        Args:
            data (obj, optional): data. Defaults to None.
            query: dict_query that generated the data

        Return
            data : Preprocessed data
        """
        if len(data) < self.min_size:
            raise (ValueError("not enough data. min_size: " + str(self.min_size)))

        if len(data) == 2:
            data, context = data

            if self.normalise_context:
                context = FE_proc.normalise_panda(
                    context, mode="transform", scaler=self.context_scaler
                )

        else:
            context = None

        if self.normalise_data:
            data = FE_proc.normalise_panda(
                data, mode="transform", scaler=self.data_scaler
            )

        list_features = []
        list_targets = []

        # Computation of features
        for n, (params, compute_func) in enumerate(
            zip(self.list_params_features, self.list_compute_features)
        ):
            if compute_func is not None:
                if self.list_update_params_features is not None:
                    if self.list_update_params_features[n] is not None:

                        params = self.list_update_params_features[n](query, params)

                features, _ = compute_func(data, context, **params)
                list_features.append(features)

        # Computation of Targets
        for n, (params, compute_func) in enumerate(
            zip(self.list_params_targets, self.list_compute_targets)
        ):
            if compute_func is not None:
                if self.list_update_params_targets is not None:
                    if self.list_update_params_targets[n] is not None:
                        params = self.list_update_params_targets[n](query, params)

                targets, _ = compute_func(data, context, **params)
                list_targets.append(targets)

        X = list_features
        if self.concat_features:
            X = np.concatenate(X, axis=1)

        y = list_targets
        if self.concat_targets:
            y = np.concatenate(list_targets, axis=1)

        if self.dataset_formalizer is not None:
            return self.dataset_formalizer(X, y, query)
        else:
            return (X, y)

    def fit_transform(self, data, query={}, **kwarg):
        self.fit(data, query)
        data = self.transform(data, query)
        return data


def init_Features_processor(
    name="Features_processor",
    dict_params_FE_ctx=None,
    dict_params_FE_dyn=None,
    dict_params_FE_targets=None,
    update_params_FE_ctx=None,
    update_params_FE_dyn=None,
    update_params_FE_targets=None,
    normalise_data=False,
    normalise_context=False,
    dataset_formalizer=None,
    min_size=1,
    structure=None,
    cache=None,
):
    list_params_features = []
    list_fit_features = []
    list_compute_features = []
    list_update_params_features = []

    if dict_params_FE_ctx is not None:
        list_params_features.append({"dict_FE_params": dict_params_FE_ctx})
        list_fit_features.append(FE_proc.fit_feature_engeenering)
        list_compute_features.append(FE_proc.compute_feature_engeenering)
        list_update_params_features.append(update_params_FE_ctx)

    if dict_params_FE_dyn is not None:
        list_params_features.append({"dict_FE_params": dict_params_FE_dyn})
        list_fit_features.append(FE_proc.fit_feature_engeenering)
        list_compute_features.append(FE_proc.compute_feature_engeenering)
        list_update_params_features.append(update_params_FE_dyn)

    list_params_targets = [{"dict_FE_params": dict_params_FE_targets}]
    list_fit_targets = [FE_proc.fit_feature_engeenering]
    list_compute_targets = [FE_proc.compute_feature_engeenering]
    list_update_params_targets = [update_params_FE_targets]

    Formalizer = Generic_Features_processor(
        name=name,
        cache=cache,
        structure=structure,
        list_params_features=list_params_features,
        list_fit_features=list_fit_features,
        list_compute_features=list_compute_features,
        list_update_params_features=list_update_params_features,
        list_params_targets=list_params_targets,
        list_fit_targets=list_fit_targets,
        list_compute_targets=list_compute_targets,
        list_update_params_targets=list_update_params_targets,
        normalise_data=normalise_data,
        normalise_context=normalise_context,
        dataset_formalizer=dataset_formalizer,
        min_size=min_size,
    )

    return Formalizer
