import copy
import inspect
import os
import random

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

import uqmodels.modelization.DL_estimator.loss as uqloss
import uqmodels.processing as uqproc
from uqmodels.modelization.DL_estimator.metalayers import mlp
from uqmodels.modelization.DL_estimator.utils import set_global_determinism
from uqmodels.modelization.UQEstimator import UQEstimator, get_UQEstimator_parameters
from uqmodels.utils import add_random_state, apply_mask, cut, generate_random_state


def Identity_factory(X, y, **kwargs):
    return (X, y, None)


class NN_UQ(UQEstimator):
    "Neural Network UQ"

    def __init__(
        self,
        model_initializer,
        model_parameters,
        factory_parameters=dict(),
        training_parameters=dict(),
        type_output=None,
        rescale=False,
        n_ech=5,
        train_ratio=0.9,
        var_min=0.000001,
        name="NN",
        random_state=None,
    ):
        self.model_initializer = model_initializer

        if random_state is not None:
            params_list = list(inspect.signature(model_initializer).parameters)

            if "seed" in params_list:
                model_parameters["seed"] = random_state

            elif "random_state" in params_list:
                model_parameters["random_state"] = random_state

            else:
                print(
                    'Warning model_initializer have not "seed" or "random_state" parameters'
                )

        self.model_parameters = model_parameters
        self.factory_parameters = factory_parameters
        self.training_parameters = training_parameters
        self.type_output = type_output
        self.initialized = False
        self.history = []
        self.n_ech = n_ech
        self.train_ratio = train_ratio

        type_UQ = "var_A&E"
        super().__init__(
            name=name,
            type_UQ=type_UQ,
            rescale=rescale,
            var_min=var_min,
            random_state=random_state,
        )
        if self.random_state is not None:
            model_parameters["random_state"] = random_state

        if "generator" not in self.training_parameters.keys():
            self.training_parameters["generator"] = False

        if "test_batch_size" not in self.training_parameters.keys():
            self.training_parameters["test_batch_size"] = 20000

        # Additional deep ensemble parameter
        self.ddof = 1
        if "ddof" in model_parameters.keys():
            self.ddof = model_parameters["ddof"]

        if "train_ratio" in model_parameters.keys():
            self.train_ratio = model_parameters["train_ratio"]

        if "n_ech" in model_parameters.keys():
            self.n_ech = model_parameters["n_ech"]

        self.snapshot = False
        if "snapshot" in model_parameters.keys():
            self.snapshot = model_parameters["snapshot"]

        self.data_drop = 0
        if "data_drop" in model_parameters.keys():
            self.data_drop = model_parameters["data_drop"]

        if "k_fold" in model_parameters.keys():
            self.k_fold = model_parameters["k_fold"]

    def build_loss(self, loss, param_loss=None):
        """Build loss from str or loss and loss_parameters

        Args:
            loss (_type_): _description_
            param_loss (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if loss == "BNN":
            loss = uqloss.build_BNN_loss
            if param_loss is None:
                param_loss = {}
        elif loss == "EDL":
            loss = uqloss.build_EDL_loss
            if param_loss is None:
                param_loss = {}
        elif loss == "MSE":
            loss = uqloss.build_MSE_loss
            if param_loss is None:
                param_loss = {}
        else:
            pass

        if param_loss is not None:
            if isinstance(param_loss, dict):
                loss = loss(**param_loss)

            else:
                loss = loss(param_loss)
        else:
            loss = loss

        return loss

    def build_metrics(self, metrics):
        """Build list of metrics from str or metrics.

        Args:
            metrics (_type_): _description_
        """
        list_metrics = []
        for metric in metrics:
            if metric == "MSE":
                output_size = 1
                if self.type_output in ["MC_Dropout", "Deep_ensemble"]:
                    output_size = 2
                elif self.type_output in ["EDL"]:
                    output_size = 4
                metric = uqloss.build_MSE_loss(output_size, metric=True)
            elif metric == "BNN":
                metric = uqloss.build_BNN_loss(
                    0.95, metric=True, type_output=self.type_output
                )
            else:
                pass
            list_metrics.append(metric)
        return list_metrics

    def _format(self, X, y, type_transform, mode_UQ=False):
        X, y = super()._format(X, y, type_transform=type_transform, mode_UQ=mode_UQ)
        return (X, y)

    def factory(self, X, y, mask=None, cut_param=None, only_fit_scaler=False):
        if y is not None:
            self.y_shape = y.shape
            if cut_param is None:
                y = y

            else:
                print("cuting_target")
                min_cut, max_cut = cut_param
                y = cut(y, min_cut, max_cut)

        if self.rescale:
            X, y = self._format(X, y, type_transform="fit_transform")

        if only_fit_scaler:
            return ()

        # X, y = [X], y

        return (X, y, mask)

    def save(self, path=None, name=None):
        if name is None:
            name = self.name

        if self.type_output == "Deep_ensemble":
            for n, model in enumerate(self.model):
                cur_name = name + "_" + str(n)
                new_path = os.path.join(path, cur_name)
                model.save_weights(new_path)
        else:
            new_path = os.path.join(path, name)
            self.model.save_weights(new_path)

        model_tmp = self.model
        self.model = True
        dict_parameters = self.__dict__
        uqproc.write(path, [name + "_params"], dict_parameters)
        self.model = model_tmp

    def load(self, path, name=None):
        old_level_info = os.environ["TF_CPP_MIN_LOG_LEVEL"]
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        if name is None:
            name = self.name

        dict_parameters = uqproc.read(path, [name + "_params"])
        for attributes, values in dict_parameters.items():
            self.__setattr__(attributes, values)

        self.init_neural_network()
        if self.type_output == "Deep_ensemble":
            for n, model in enumerate(self.model):
                new_path = os.path.join(path, name + "_" + str(n))
                model.load_weights(new_path)
        else:
            new_path = os.path.join(path, name)
            self.model.load_weights(new_path)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = old_level_info

    def compile(self, step=0, optimizer=None, loss=None, metrics=None, **kwarg):
        if optimizer is None:
            l_r = self.training_parameters["l_r"][step]
            optimizer = tf.keras.optimizers.experimental.Nadam(learning_rate=l_r)
        kwarg["optimizer"] = optimizer

        if loss is None:
            loss_ = self.training_parameters["list_loss"][step]
            param_loss_current = self.training_parameters["param_loss"][step]
            loss = self.build_loss(loss_, param_loss_current)
        kwarg["loss"] = loss

        if metrics is None:
            metrics = self.training_parameters["metrics"]
        kwarg["metrics"] = metrics
        self.model.compile(**kwarg)

    def modify_dropout(self, dp):
        self.model.save_weights(self.name)
        self.model_parameters["dp"] = dp
        self.model = self.model_initializer(**self.model_parameters)
        self.initialized = True
        self.model.load_weights(self.name)

    def reset(self):
        del self.model
        self.initialized = False

    def init_neural_network(self):
        "apply model_initializer function with model_parameters and store in self.model"
        if self.random_state is not None:
            set_global_determinism(seed=self.random_state)

        if self.type_output == "Deep_ensemble":
            self.model = []
            for i in range(self.n_ech):
                self.model.append(self.model_initializer(**self.model_parameters))
        else:
            self.model = self.model_initializer(**self.model_parameters)
        self.initialized = True

    def fit(
        self,
        Inputs,
        Targets,
        train=None,
        test=None,
        training_parameters=None,
        verbose=None,
        **kwargs
    ):

        print("start_fit")

        if training_parameters is None:
            training_parameters = copy.deepcopy(self.training_parameters)
            if verbose is not None:
                training_parameters["verbose"] = verbose

        if not (self.initialized) or not hasattr(self, "model"):
            self.init_neural_network()

        if train is None:
            last_val = False
            if hasattr(self, "model_parameters") and (
                "size_window" in self.model_parameters.keys()
            ):
                last_val = True

            train, test = generate_train_test(
                len_=len(Targets),
                train_ratio=self.train_ratio,
                last_val=last_val,
                random_state=self.random_state,
            )

        if test is None:
            test = np.invert(train)

        history = self.basic_fit(
            Inputs, Targets, train, test, **self.training_parameters
        )

        for i in history:
            self.history.append(history)

    # Basic_predict function

    def basic_fit(
        self,
        Inputs,
        Targets,
        train=None,
        test=None,
        epochs=[1000, 1000],
        b_s=[100, 20],
        l_r=[0.01, 0.005],
        sample_w=None,
        verbose=1,
        list_loss=["mse"],
        metrics=None,
        generator=None,
        steps_per_epoch=None,
        shuffle=True,
        callbacks="default",
        validation_freq=1,
        param_loss=None,
        test_batch_size=None,
        **kwargs
    ):

        # Training function
        history = []
        list_history = []

        if self.random_state is not None:
            set_global_determinism(seed=self.random_state)

        if generator is None:
            generator = self.training_parameters["generator"]

        if test_batch_size is None:
            test_batch_size = self.training_parameters["test_batch_size"]

        if train is None:
            last_val = False
            if hasattr(self, "model_parameters") and (
                "size_window" in self.model_parameters.keys()
            ):
                last_val = True

            train, test = generate_train_test(
                len_=len(Targets),
                train_ratio=self.train_ratio,
                last_val=last_val,
                random_state=self.random_state,
            )

        if test is None:
            test = np.invert(train)

        list_history = []

        if not (hasattr(self, "scaler")):
            _ = self.factory(Inputs, Targets, only_fit_scaler=True)

        n_model = 1
        if self.type_output == "Deep_ensemble":
            n_model = self.n_ech
            list_sampletoremove = generate_K_fold_removing_index(
                n_model,
                k_fold=self.k_fold,
                train=train,
                data_drop=self.data_drop,
                random_state=self.random_state,
            )

        for n_model in range(n_model):
            train_ = np.copy(train)
            test_ = np.copy(test)

            # Deep_ensemble : Submodel dataset differentiation if kfold activated
            if self.type_output == "Deep_ensemble":
                train_[list_sampletoremove[n_model]] = False
                test_[list_sampletoremove[n_model]] = True

            for n, loss in enumerate(list_loss):
                for i, (batch_size, learning_rate) in enumerate(zip(b_s, l_r)):

                    loss_ = self.build_loss(loss, param_loss[n])
                    metrics = self.build_metrics(metrics)

                    if self.type_output == "Deep_ensemble":
                        if (self.snapshot) & (n_model > 0):
                            self.model[n_model] = tf.keras.clone_model(self.model[0])

                        current_model = self.model[n_model]

                    else:
                        current_model = self.model

                    current_model.compile(
                        optimizer=tf.keras.optimizers.experimental.Nadam(
                            learning_rate=learning_rate
                        ),
                        loss=loss_,
                        metrics=metrics,
                    )

                    (
                        In_,
                        Tar_,
                        validation_data_,
                        validation_steps,
                        steps_per_epoch,
                        batch_size,
                    ) = self.dataset_generator(
                        Inputs=apply_mask(Inputs, train_),
                        Targets=apply_mask(Targets, train_),
                        validation_data=(
                            apply_mask(Inputs, test_),
                            apply_mask(Targets, test_),
                        ),
                        batch_size=batch_size,
                        generator=generator,
                        shuffle=shuffle,
                        test_batch_size=test_batch_size,
                    )

                    if callbacks == "default":
                        callbacks = uqloss.default_callbacks()

                    history = current_model.fit(
                        x=In_,
                        y=Tar_,
                        validation_data=validation_data_,
                        epochs=epochs[i],
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        batch_size=batch_size,
                        sample_weight=sample_w,
                        shuffle=shuffle,
                        callbacks=callbacks,
                        validation_freq=validation_freq,
                        verbose=verbose,
                    )

                    current_model.compile()

                    list_history.append(history)

        return list_history

    def dataset_generator(
        self,
        Inputs,
        Targets,
        validation_data,
        batch_size,
        shuffle,
        generator,
        test_batch_size=None,
    ):
        """Hold case with or without data generator

        Args:
            Inputs (_type_): _description_
            Targets (_type_): _description_
            validation_data (_type_): _description_
            batch (_type_): _description_
            shuffle (_type_): _description_
            generator (_type_): _description_

        Returns:
            _type_: _description_
        """
        if generator:
            In_ = self.Build_generator(
                Inputs, Targets, batch=batch_size, shuffle=shuffle, train=True
            )
            Tar_ = None

            if test_batch_size is None:
                test_batch_size = self.training_parameters["test_batch_size"]

            validation_data_ = self.Build_generator(
                validation_data[0],
                validation_data[1],
                batch=test_batch_size,
                shuffle=False,
                train=True,
            )

            steps_per_epoch = In_.__len__()
            validation_steps = validation_data_.__len__()
            batch_size = None
        else:
            In_ = Inputs
            Tar_ = Targets
            validation_data_ = validation_data
            validation_steps = None
            steps_per_epoch = None
        return (
            In_,
            Tar_,
            validation_data_,
            validation_steps,
            steps_per_epoch,
            batch_size,
        )

    def predict(self, X, type_output=None, generator=None, **kwargs):
        if type_output is None:
            type_output = self.type_output

        pred, UQ = self.basic_predict(
            X, n_ech=self.n_ech, type_output=type_output, generator=generator, **kwargs
        )

        if self.rescale:
            _, pred = self._format(None, pred, type_transform="inverse_transform")
            _, UQ = self._format(
                None, UQ, type_transform="inverse_transform", mode_UQ=True
            )

        return (pred, UQ)

    def basic_predict(
        self,
        Inputs,
        n_ech=6,
        type_output="MC_Dropout",
        generator=None,
        test_batch_size=None,
        **kwarg
    ):
        # Variational prediction + variance estimation for step T+1 et T+4(lag)
        if generator is None:
            generator = self.training_parameters["generator"]

        if self.random_state is not None:
            set_global_determinism(seed=self.random_state)

        if generator:
            if test_batch_size is None:
                test_batch_size = self.training_parameters["test_batch_size"]
            Inputs = self.Build_generator(
                Inputs, None, batch=test_batch_size, shuffle=False, train=False
            )

        if type_output in ["MC_Dropout", "MC_Dropout_no_PNN"]:
            pred, UQ = Drawn_based_prediction(
                Inputs,
                self.model,
                n_ech,
                ddof=self.ddof,
                generator=generator,
                type_output=type_output,
            )

        elif type_output == "Deep_ensemble":
            pred, UQ = Ensemble_based_prediction(
                Inputs,
                self.model,
                ddof=self.ddof,
                generator=generator,
                type_output=type_output,
            )

        elif type_output in ["EDL", "PNN", "None", None]:
            pred, UQ = Deterministic_prediction(
                Inputs,
                self.model,
                ddof=self.ddof,
                generator=generator,
                type_output=type_output,
            )

        else:
            raise Exception(
                "Unknown type_output : choose 'MC_Dropout' or 'Deep_esemble' or 'EDL' or 'Non' or None"
            )

        return (pred, UQ)

    def Build_generator(self, X, y, batch=32, shuffle=True, train=True):
        return default_Generator(X, y, self, batch=batch, shuffle=shuffle, train=train)


def Drawn_based_prediction(
    Inputs, model, n_ech, ddof, generator=False, type_output="MC_Dropout"
):
    """Prediction (mu,sigma) of Inputs using  Drawn_based UQ-paragim (Ex : MC_dropout)

    Args:
        model (tf.model): neural network
        n_ech (n_draw): number of dropout drawn
        Inputs (_type_): Inputs of model
        ddof (_type_): ddof
        generator (bool, optional): specify if Inputs is generator or not. Defaults to False.

    Returns:
        _type_: _description_
    """

    if generator:
        pred = []
        var_a = []
        var_e = []
        for Inputs_gen, _ in Inputs:  # by batch do n inf and aggreagate results
            output = []
            for i in range(n_ech):
                output.append(model.predict(Inputs_gen))

            if type_output == "MC_Dropout_no_PNN":
                pred_ = np.array(output)
                var_a.append(0 * pred_.mean(axis=0))

            if type_output == "MC_Dropout":
                pred_, logvar = np.split(np.array(output), 2, -1)
                var_a.append(np.exp(logvar).mean(axis=0))

            pred.append(pred_.mean(axis=0))
            var_e.append(pred_.var(axis=0))

        pred = np.concatenate(pred, axis=0)
        var_a = np.concatenate(var_a, axis=0)
        var_e = np.concatenate(var_e, axis=0)
        UQ = np.concatenate([var_a[None, :], var_e[None, :]], axis=0)
    else:
        output = []
        for i in range(n_ech):
            output.append(model.predict(Inputs))

        if type_output == "MC_Dropout_no_PNN":
            pred_ = np.array(output)
            var_a = 0 * pred_.mean(axis=0)

        if type_output == "MC_Dropout":
            pred_, logvar = np.split(np.array(output), 2, -1)
            var_a = np.exp(logvar).mean(axis=0)
        var_e = np.var(pred_, axis=0, ddof=ddof)
        pred = pred_.mean(axis=0)
    UQ = np.concatenate([var_a[None, :], var_e[None, :]], axis=0)
    return (pred, UQ)


def Deterministic_prediction(Inputs, model, ddof, generator=False, type_output=None):
    """Prediction (mu,sigma) of Inputs using Deterministic UQ-paragim (Ex : EDL)

    Args:
        model (tf.model): neural network
        n_ech (n_draw): number of dropout drawn
        Inputs (_type_): Inputs of model
        ddof (_type_): ddof
        generator (bool, optional): specify if Inputs is generator or not. Defaults to False.
        type_output : type_output (EDL)

    Returns:
        _type_: _description_
    """

    if generator:
        output = []
        for Inputs_gen, _ in Inputs:  # by batch do n inf and aggreagate results
            output.append(model.predict(Inputs_gen))
        output = np.concatenate(output, axis=0)

    else:
        output = model.predict(Inputs)

    if type_output == "EDL":
        gamma, vu, alpha, beta = np.split(output, 4, -1)
        alpha = alpha + 10e-6
        pred = gamma
        var_A = beta / (alpha - 1)
        # WARNING sqrt or not sqrt ?
        var_E = beta / (vu * (alpha - 1))
        if (var_E == np.inf).sum() > 0:
            print("Warning inf values in var_E replace by s-min")
        if (var_A == np.inf).sum() > 0:
            print("Warning inf values in var_E replace by s-min")
        var_E[var_E == np.inf] = 0
        var_A[var_A == np.inf] = 0

    elif type_output == "PNN":
        pred, logvar = np.split(output, 2, -1)
        var_A = np.exp(logvar)
        var_E = logvar * 0

    else:
        pred = output
        var_A = 0 * pred
        var_E = 0 * pred

    UQ = np.concatenate([var_E[None, :], var_A[None, :]], axis=0)
    return (pred, UQ)


def Ensemble_based_prediction(Inputs, models, ddof, generator=False, type_output=None):
    """Prediction (mu,sigma) of Inputs using Ensemble_based UQ-paradign

    Args:
        model (tf.model): neural network
        n_ech (n_draw): number of dropout drawn
        Inputs (_type_): Inputs of model
        ddof (_type_): ddof
        generator (bool, optional): specify if Inputs is generator or not. Defaults to False.
        type_output : type_output (curently useless)

    Returns:
        _type_: _description_
    """

    if generator:
        pred = []
        var_a = []
        var_e = []
        for Inputs_gen, _ in Inputs:  # by batch do n inf and aggreagate results
            output = []
            for submodel in models:
                output.append(submodel.predict(Inputs_gen))

            pred_, logvar = np.split(np.array(output), 2, -1)
            var_a.append(np.exp(logvar).mean(axis=0))
            var_e.append(pred_.var(axis=0, ddof=ddof))
            pred.append(pred_.mean(axis=0))

        pred = np.concatenate(pred, axis=0)
        var_a = np.concatenate(var_a, axis=0)
        var_e = np.concatenate(var_e, axis=0)
    else:
        output = []
        for submodel in models:
            output.append(submodel.predict(Inputs))
        pred, logvar = np.split(np.array(output), 2, -1)
        var_a = np.exp(logvar).mean(axis=0)
        var_e = np.var(pred, axis=0, ddof=ddof)
        pred = pred.mean(axis=0)
    UQ = np.concatenate([var_a[None, :], var_e[None, :]], axis=0)
    return (pred, UQ)


def get_training_parameters(
    epochs=[100, 100],
    b_s=[64, 32],
    l_r=[0.005, 0.001],
    list_loss=None,
    metrics=None,
    param_loss=None,
    type_output=None,
    generator=False,
    shuffle=True,
    verbose=1,
    sample_w=None,
    callbacks="default",
    **kwargs
):
    if list_loss is None:
        if type_output is None:
            list_loss = ["MSE"]
            metrics = ["MSE"]

        if type_output == "MC_Dropout":
            list_loss = ["MSE", "BNN"]
            metrics = ["MSE", "BNN"]
            param_loss = [2, 0.9]

        if type_output == "Deep_ensemble":
            list_loss = ["MSE", "BNN"]
            metrics = ["MSE", "BNN"]
            param_loss = [2, 0.9]

        if type_output == "EDL":
            list_loss = ["MSE", "EDL", "EDL"]
            metrics = ["MSE", "BNN"]
            param_loss = [4, 1e-2, 10e-2]

    dict_params = {
        "epochs": epochs,
        "b_s": b_s,
        "l_r": l_r,
        "sample_w": sample_w,
        "list_loss": list_loss,
        "metrics": metrics,
        "param_loss": param_loss,
        "generator": generator,
        "shuffle": shuffle,
        "verbose": verbose,
        "callbacks": callbacks,
    }

    for key_arg in kwargs.keys():
        dict_params[key_arg] = kwargs[key_arg]
    return dict_params


def get_params_dict(
    dim_in,
    dim_out=1,
    layers_size=[200, 150, 100],
    regularizer_=(0.0001, 0.0001),
    dp=None,
    name="MLP_UQ",
    type_output="MC_Dropout",
    **kwargs
):

    dict_params = {
        "dim_in": dim_in,
        "dim_out": dim_out,
        "layers_size": layers_size,
        "regularizer_": regularizer_,
        "name": name,
        "n_ech": 5,
        "dp": dp,
        "type_output": type_output,
        "logvar_min": np.log(0.00005),
    }
    if type_output == "MC_Dropout":
        if dp is None:
            dict_params["dp"] = 0.15

    if type_output == "Deep_ensemble":
        dict_params["n_ech"] = 5
        dict_params["k_fold"] = 8
        if dp is None:
            dict_params["dp"] = 0.02

    if type_output == "EDL":
        if dp is None:
            dict_params["dp"] = 0.02

    for key_arg in kwargs.keys():
        dict_params[key_arg] = kwargs[key_arg]

    return dict_params


class default_Generator(tf.keras.utils.Sequence):
    def __init__(
        self, X, y, metamodel, batch=64, shuffle=True, train=True, random_state=None
    ):
        self.X = X
        self.y = y
        self.len_ = len(y)
        self.train = train
        self.random_state = random_state
        self.shuffle = shuffle
        self.batch = batch

        # self.scaler = metamodel.scaler
        self.factory = metamodel.factory
        self._format = metamodel._format
        self.rescale = metamodel.rescale

        if shuffle:
            self.indices = np.arange(self.len_)
            np.random.seed(self.random_state)
            np.random.shuffle(self.indices)

    def load(self, idx):
        idx = idx * self.batch

        seuil_min = max(0, idx - 1)
        seuil_max = min(idx, self.len_ - 1)
        Inputs = self.X[seuil_min:seuil_max]
        Targets = self.y[seuil_min:seuil_max]
        return (Inputs, Targets)

    def __len__(self):
        return self.len_

    def __getitem__(self, idx):
        if self.shuffle:
            idx = self.indices[idx]

        x, y = self.load(idx)
        Inputs, Ouputs, _ = self.factory(x, y)

        len_ = len(Inputs)
        selection = np.zeros(len(Inputs[0])) == 1

        selection[max(0, idx * self.batch) : min(len_, (idx + 1) * self.batch)] = True

        Inputs = apply_mask(Inputs, selection)
        Ouputs = apply_mask(Ouputs, selection)
        return Inputs, Ouputs

    # shuffles the dataset at the end of each epoch
    def on_epoch_end(self):
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indices)


def generate_K_fold_removing_index(
    n_model, k_fold, train, data_drop, random_state=None
):
    """Generate liste of idx to remove for k_fold deep ensemble procedure

    Args:
        n_model (_type_): Number of models
        k_fold (_type_): Number of fold
        train (_type_): train_flag_idx
        data_drop (_type_): % of data drop
        random_state : handle experimental random using seed
    Returns:
        _type_: list_sampletoremove idx of sample to remove of train for each submodel
    """
    list_sampletoremove = []
    if k_fold is not None:
        if k_fold < n_model:
            print("Warning kfold lesser than model number")
        # Drop data using Kfold + random drop ratio to add variability to deep ensemble
        for n_fold, (keep, removed) in enumerate(
            KFold(k_fold, shuffle=True, random_state=random_state).split(train)
        ):
            if data_drop > 0:
                np.random.seed(add_random_state(random_state, n_fold))
                sampletoremove = np.random.choice(
                    keep, int(len(keep) * data_drop), replace=False
                )
                sampletoremove = sorted(np.concatenate([removed, sampletoremove]))
                list_sampletoremove.append(sampletoremove)
            else:
                list_sampletoremove.append([])
    else:
        list_sampletoremove = [[] for i in range(n_model)]
        if data_drop > 0:
            for n, i in enumerate(list_sampletoremove):
                np.random.seed(add_random_state(random_state, n_fold))
                sampletoremove = np.random.choice(
                    np.arange(len(train)),
                    int(len(train) * data_drop),
                    replace=False,
                )
                list_sampletoremove[n] = sampletoremove
    return list_sampletoremove


def generate_train_test(len_, train_ratio=0.92, last_val=True, random_state=None):
    if last_val:
        train = np.arange(len_) < train_ratio * len_
    else:
        np.random.seed(random_state)
        train = np.random.rand(len_) < train_ratio
    test = np.invert(train)
    return (train, test)
