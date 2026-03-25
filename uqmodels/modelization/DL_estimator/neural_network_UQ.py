import copy
import inspect
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from abc import ABC, abstractmethod
import uqmodels.modelization.DL_estimator.loss as uqloss
import uqmodels.processing as uqproc
from uqmodels.modelization.DL_estimator.utils import set_global_determinism
from uqmodels.modelization.UQEstimator import UQEstimator
from uqmodels.modelization.DL_estimator.data_generator import default_Generator
from uqmodels.utils import add_random_state, apply_mask, cut


class UQOutput(ABC):
    """
    Interface for models exposing uncertainty outputs.
    Required by `NN_UQ.basic_inference` to extract aleatoric/epistemic variance.
    """

    @abstractmethod
    def get_uq_outputs(self, y_pred: np.ndarray, type_output: str):
        """
        Parse raw model output into (mean, var_a, var_e).

        Args:
            y_pred: raw model prediction (e.g., [recon] or [mean, logvar] or [γ, ν, α, β])
            type_output: one of 'PNN', 'EDL', 'MC_Dropout', 'Deep_ensemble', None

        Returns:
            tuple: (mean, var_a, var_e) each as np.ndarray or None (if not available)
        """
        pass


class UQModelMixin:
    """
    Mixin to be inherited by TensorFlow models (or wrapped models) that support UQ output parsing.
    Provides default implementation for `get_uq_outputs`.
    """

    def get_uq_outputs(self, y_pred: np.ndarray, type_output: str):
        """
        Default behavior:
        - 'PNN': y_pred = [μ, logσ²] → var_a = exp(logσ²), var_e = 0
        - 'EDL': y_pred = [γ, ν, α, β] → compute Normal-Gamma variance
        - else: y_pred = μ → var_a = var_e = 0
        """
        if type_output == "PNN":
            mu, logvar = np.split(y_pred, 2, axis=-1)
            return mu, np.exp(logvar), np.zeros_like(mu)

        elif type_output == "EDL":
            gamma, nu, alpha, beta = np.split(y_pred, 4, axis=-1)
            alpha = alpha + 1e-5
            var_a = beta / (alpha - 1)
            var_e = beta / (nu * (alpha - 1))
            # replace inf
            var_e[var_e == np.inf] = 0.0
            var_a[var_a == np.inf] = 0.0
            return gamma, var_a, var_e

        else:  # deterministic or None
            return y_pred, np.zeros_like(y_pred), np.zeros_like(y_pred)

class NN_UQ(UQEstimator):
    """
    Neural Network wrapper for Uncertainty Quantification (UQ) on time series.

    This class provides a high-level interface to train and perform inference on
    deep learning models capable of estimating both **aleatoric** (data-inherent)
    and **epistemic** (model-induced) uncertainty. It supports three mainstream
    UQ paradigms:

    - **MC Dropout**: Stochastic inference via repeated forward passes with dropout enabled.
    - **Deep Ensemble**: Averaging predictions from multiple independently trained models.
    - **Epistemic Neural Networks (EDL)**: Explicit probabilistic modeling (e.g., Gamma distribution) to quantify uncertainty.

    For all modes, the interface is unified:
      - ``predict(X)`` returns a tuple ``(y_pred, UQ)``
      - ``UQ`` has shape ``(2, *y_pred.shape[1:])``: ``UQ[0] = var_aleatoric``, ``UQ[1] = var_epistemic``

    The wrapper manages the entire lifecycle:
      - *Data preparation* (rescaling, masking, train/test split, generators)
      - *Model instantiation* (via user-provided initializer)
      - *Training* (multi-stage, multi-loss, early stopping, snapshots)
      - *Serialization* (weights + hyperparameters)

    It is designed for **sequence modeling** (e.g., time-series forecasting), but
    supports any input-output ND array where axes are compatible with the initializer.

    Parameters
    ----------
    model_initializer : callable
        Factory function returning a TensorFlow/Keras model (or dict with keys
        ``'model'`` and optionally ``'encoder'``). It must accept keyword arguments
        defined in `model_parameters`.
    model_parameters : dict
        Arguments passed to ``model_initializer`` (e.g., ``dim_in``, ``dim_out``,
        ``layers_size``, ``dp``, ``random_state``).
    factory_parameters : dict, optional
        Reserved for future extension (currently unused).
    training_parameters : dict
        Training configuration, including epochs, batch sizes, learning rates,
        loss functions, metrics, callbacks, etc. Use ``get_training_parameters``
        helper to generate defaults per ``type_output``.
    type_output : {'MC_Dropout', 'Deep_ensemble', 'EDL', 'PNN', None}, optional
        UQ strategy. Defaults to None → deterministic output.
    rescale : bool, default False
        Whether to apply input/target rescaling (handled by parent `UQEstimator`).
    n_ech : int, default 5
        Number of stochastic draws (MC Dropout) or ensemble members (Deep Ensemble).
    train_ratio : float, default 0.9
        Fraction of data used for training (remaining used for validation).
    var_min : float, default 1e-6
        Minimum variance clamped during inference (prevents numerical instabilities).
    name : str, default "NN"
        Identifier used for saving/loading models.
    random_state : int or None, default None
        Seed for reproducibility (applied where supported by the initializer).

    Notes
    -----
    - The model initializer can return:
        - A single Keras model (used for prediction & encoding if ``model_encode`` not provided)
        - A dict with keys ``'model'`` (prediction model) and optionally ``'encoder'``
          (intermediate representation model). Encoder inference is invoked via ``.encode()``.

    - Uncertainty decomposition:
        - *Aleatoric*: irreducible noise in observations.
        - *Epistemic*: reducible model uncertainty (via diversity: dropout, ensembles, or EDL).

    - For **EDL**, outputs are expected to be split into 4 chunks: ``[γ, ν, α, β]``
      describing a Normal-Gamma distribution; variance formulas are:
      .. math::
          \\sigma^2_A = \\frac{\\beta}{\\alpha - 1}, \\quad
          \\sigma^2_E = \\frac{\\beta}{\\nu (\\alpha - 1)}

    - For **PNN**, outputs are ``[y, \\log \\sigma^2]``; aleatoric variance is
      ``exp(\\log \\sigma^2)``.
    """

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

    def get_uq_from_prediction(self, y_pred: np.ndarray, type_output: str):
        """
        Extract (mean, var_a, var_e) from raw prediction, using model's interface if available.

        Falls back to default parsing if model doesn't implement UQOutput.
        """
        # Try to delegate to model
        if hasattr(self, "model") and isinstance(self.model, (list, tuple)):
            # Deep ensemble: assume all members implement same interface
            model = self.model[0]
        else:
            model = self.model

        if isinstance(model, UQOutput):
            return model.get_uq_outputs(y_pred, type_output)

        # Default behavior (backward compatible)
        return self._default_uq_parse(y_pred, type_output)

    def _default_uq_parse(self, y_pred, type_output):
        """Original logic, kept for backward compatibility."""
        if type_output == "EDL":
            gamma, nu, alpha, beta = np.split(y_pred, 4, axis=-1)
            alpha += 1e-5
            var_a = beta / (alpha - 1)
            var_e = beta / (nu * (alpha - 1))
            var_e[var_e == np.inf] = 0.0
            var_a[var_a == np.inf] = 0.0
            return gamma, var_a, var_e

        elif type_output == "PNN":
            mu, logvar = np.split(y_pred, 2, axis=-1)
            return mu, np.exp(logvar), np.zeros_like(mu)

        else:
            return y_pred, np.zeros_like(y_pred), np.zeros_like(y_pred)
        
    def save(self, path=None, name=None):
        if name is None:
            name = self.name

        if self.type_output == "Deep_ensemble":
            for n, model in enumerate(self.model):
                cur_name = name + "_" + str(n)
                new_path = os.path.join(path, cur_name + '.weights.h5')
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                model.save_weights(new_path)
        else:
            new_path = os.path.join(path, name + '.weights.h5')
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            self.model.save_weights(new_path)

        model_tmp = self.model
        self.model = True
        dict_parameters = self.__dict__
        uqproc.write(path, [name + "_params"], dict_parameters)
        self.model = model_tmp

    def load(self, path, name=None):
        # old_level_info = os.environ["TF_CPP_MIN_LOG_LEVEL"]
        # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        if name is None:
            name = self.name

        dict_parameters = uqproc.read(path, [name + "_params"])
        for attributes, values in dict_parameters.items():
            self.__setattr__(attributes, values)

        self.init_neural_network()
        if self.type_output == "Deep_ensemble":
            for n, model in enumerate(self.model):
                new_path = os.path.join(path, name + "_" + str(n) + '.weights.h5')
                model.load_weights(new_path)
        else:
            new_path = os.path.join(path, name + '.weights.h5')
            self.model.load_weights(new_path)
        # os.environ["TF_CPP_MIN_LOG_LEVEL"] = old_level_info

    def compile(self, step=0, optimizer=None, loss=None, metrics=None, **kwarg):
        if optimizer is None:
            l_r = self.training_parameters["l_r"][step]
            optimizer = tf.keras.optimizers.Nadam(learning_rate=l_r)
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
        """Rebuild model(s) with a new dropout rate and reload previous weights.

        This method relies on `init_neural_network` so that model instantiation
        remains consistent with the initializer contract:
            - model
            - {"model": ..., "encode": ...}
        """
        if not self.initialized:
            raise RuntimeError("Cannot modify dropout before model initialization.")

        if self.type_output == "Deep_ensemble":
            weight_paths = []
            for i, model in enumerate(self.model):
                weight_path = f"{self.name}_{i}"
                model.save_weights(weight_path)
                weight_paths.append(weight_path)
        else:
            self.model.save_weights(self.name)

        self.model_parameters["dp"] = dp
        self.init_neural_network()

        if self.type_output == "Deep_ensemble":
            for model, weight_path in zip(self.model, weight_paths):
                model.load_weights(weight_path)
        else:
            self.model.load_weights(self.name)

    def reset(self):
        del self.model
        self.initialized = False

    def _parse_initializer_output(self, init_output):
        """Normalize initializer output into model and optional encoder.

        Supported initializer outputs:
            - a model object
            - a dict containing at least key "model" and optionally key "encode"

        Args:
            init_output: Output returned by self.model_initializer(**self.model_parameters).

        Returns:
            tuple:
                model: Main prediction model.
                model_encode: Optional encoding model, or None.

        Raises:
            TypeError: If the initializer output format is unsupported.
            KeyError: If a dict output does not contain the required "model" key.
        """
        if isinstance(init_output, dict):
            if "model" not in init_output:
                raise KeyError(
                    "Initializer returned a dict but missing required key 'model'."
                )
            model = init_output["model"]
            model_encode = init_output.get("encoder", None)
        else:
            model = init_output
            model_encode = None

        return model, model_encode


    def init_neural_network(self):
        """Instantiate and store prediction model and optional encoder.

        The initializer can return either:
            - a model object
            - a dict with keys:
                - 'model': main prediction model
                - 'encode': optional encoder model

        For deep ensembles, self.model and self.model_encode are lists of length n_ech.
        For non-ensemble settings, self.model is a single model and self.model_encode
        is either a single encoder model or None.
        """
        self.model_encode = None
        if self.random_state is not None:
            set_global_determinism(seed=self.random_state)

        if self.type_output == "Deep_ensemble":
            self.model = []
            self.model_encode = []

            for _ in range(self.n_ech):
                init_output = self.model_initializer(**self.model_parameters)
                model, model_encode = self._parse_initializer_output(init_output)
                self.model.append(model)
                self.model_encode.append(model_encode)

            # If no encoder was provided for any ensemble member, keep a simpler API.
            if all(enc is None for enc in self.model_encode):
                self.model_encode = None

        else:
            init_output = self.model_initializer(**self.model_parameters)
            self.model, self.model_encode = self._parse_initializer_output(init_output)

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
                        optimizer=tf.keras.optimizers.Nadam(
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
                        batch_size
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
        validation_data=None,
        batch_size=32,
        shuffle=False,
        generator=True,
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

    def Build_generator(self, X, y, batch=32, shuffle=True, train=True):
        return default_Generator(X, y, metamodel=self, batch=batch, shuffle=shuffle, train=train)

    def predict(self, X, type_output=None, generator=None, **kwargs):
        """User API for prediction inference."""
        if type_output is None:
            type_output = self.type_output

        pred, UQ = self.basic_inference(
            X,
            n_ech=self.n_ech,
            type_output=type_output,
            generator=generator,
            inference_mode="prediction",
            **kwargs,
        )

        if self.rescale:
            _, pred = self._format(None, pred, type_transform="inverse_transform")
            _, UQ = self._format(
                None, UQ, type_transform="inverse_transform", mode_UQ=True
            )

        return pred, UQ


    def encode(self, X, type_output=None, generator=None, **kwargs):
        """User API for encoding inference.

        This method reuses the same intermediate inference pipeline as `predict`,
        but requests encoder inference instead of prediction inference.

        Warning:
            Output rescaling is not applied in encoding mode, since latent
            representations are not assumed to belong to the original target space.
        """
        if type_output is None:
            type_output = self.type_output

        pred, UQ = self.basic_inference(
            X,
            n_ech=self.n_ech,
            type_output=type_output,
            generator=generator,
            inference_mode="encoding",
            **kwargs,
        )
        return pred, UQ

    def basic_inference(
        self,
        Inputs,
        n_ech=6,
        type_output="MC_Dropout",
        generator=None,
        test_batch_size=None,
        inference_mode="prediction",
        **kwargs,
    ):
        """Intermediate inference API for prediction or encoding.

        Args:
            Inputs: Raw inputs or iterable input source.
            n_ech (int): Number of stochastic draws when relevant.
            type_output (str | None): Output mode for prediction inference.
            generator (bool | None): Whether to use generator-based inference.
            test_batch_size (int | None): Batch size used for generator inference.
            inference_mode (str): Inference target:
                - "prediction"
                - "encoding"

        Returns:
            tuple[np.ndarray, np.ndarray]:
                pred: Prediction output or latent encoding.
                UQ: Uncertainty tensor.
        """
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

        effective_type_output = self.type_output if type_output is None else type_output
        
        if inference_mode == "prediction":
            inference_model = self.model

        elif inference_mode == "encoding":
            if not hasattr(self, "model_encode") or self.model_encode is None:
                raise AttributeError(
                    "Encoding inference requested but 'self.model_encode' is not defined."
                )
            inference_model = self.model_encode


        else:
            raise ValueError(
                f"Unsupported inference_mode='{inference_mode}'. "
                "Expected 'prediction' or 'encoding'."
            )
        if effective_type_output in ["MC_Dropout", "MC_Dropout_no_PNN"]:
            pred, UQ = Drawn_based_inference(
                Inputs,
                inference_model,
                n_ech,
                ddof=self.ddof,
                generator=generator,
                type_output=effective_type_output,
                uq_parser=self.get_uq_from_prediction,
            )

        elif effective_type_output == "Deep_ensemble":
            pred, UQ = Ensemble_based_inference(
                Inputs,
                inference_model,
                ddof=self.ddof,
                generator=generator,
                type_output=effective_type_output,
                uq_parser=self.get_uq_from_prediction,
            )

        elif effective_type_output in ["EDL", "PNN", "None", None]:
            pred, UQ = Deterministic_inference(
                Inputs,
                inference_model,
                ddof=self.ddof,
                generator=generator,
                type_output=effective_type_output,
                uq_parser=self.get_uq_from_prediction,
            )

        else:
            raise Exception(
                "Unknown type_output: choose 'MC_Dropout', 'MC_Dropout_no_PNN', "
                "'Deep_ensemble', 'EDL', 'PNN', 'None' or None."
            )

        return pred, UQ
    
def Deterministic_inference(Inputs, model, ddof, generator=False, type_output=None,uq_parser=None):
    """Run deterministic inference on a provided model.

    The provided model can be either:
        - a prediction model
        - an encoding model

    In both cases, this function only aggregates the model output and applies
    optional output parsing when requested by `type_output`.

    Args:
        Inputs: Model inputs or generator yielding (x, y).
        model: Inference model exposing predict(...).
        ddof: Unused for deterministic inference, kept for API compatibility.
        generator (bool, optional): Whether Inputs is a generator.
        type_output (str | None, optional): Output interpretation.
            Supported values:
                - "EDL"
                - "PNN"
                - None

    Returns:
        tuple[np.ndarray, np.ndarray]:
            pred: Prediction or latent representation.
            UQ: Array with shape (2, ...) where:
                UQ[0] = epistemic variance
                UQ[1] = aleatoric variance
    """
    if generator:
        output = []
        for Inputs_gen, _ in Inputs:
            output.append(model.predict(Inputs_gen))
        output = np.concatenate(output, axis=0)
    else:
        output = model.predict(Inputs)
    
    if uq_parser is not None:
        mean, var_a, var_e = uq_parser(output, type_output)

    else:
        # default parsing (backward compat)    
        if type_output == "EDL":
            gamma, vu, alpha, beta = np.split(output, 4, axis=-1)
            alpha = alpha + 1e-5
            pred = gamma
            var_A = beta / (alpha - 1)
            var_E = beta / (vu * (alpha - 1))

            if (var_E == np.inf).sum() > 0:
                print("Warning inf values in var_E replaced by 0")
            if (var_A == np.inf).sum() > 0:
                print("Warning inf values in var_A replaced by 0")

            var_E[var_E == np.inf] = 0
            var_A[var_A == np.inf] = 0

        elif type_output == "PNN":
            mu, logvar = np.split(output, 2, axis=-1)
            mean, var_a, var_e = mu, np.exp(logvar), np.zeros_like(mu)

        else:
            mean, var_a, var_e = output, np.zeros_like(output), np.zeros_like(output)

    UQ = np.concatenate([var_e[None, :], var_a[None, :]], axis=0)
    return mean, UQ

def Drawn_based_inference(
    Inputs,
    model,
    n_ech,
    ddof,
    generator=False,
    type_output="MC_Dropout",
    uq_parser=None,
):
    """Run draw-based inference on a provided model.

    Args:
        Inputs: Model inputs or generator yielding (x, y).
        model: Inference model exposing predict(...).
        n_ech (int): Number of stochastic draws.
        ddof (int): Delta degrees of freedom for variance estimation.
        generator (bool, optional): Whether Inputs is a generator.
        type_output (str, optional): Output interpretation.
            Supported values:
                - "MC_Dropout": output is concatenated as [pred, logvar]
                - "MC_Dropout_no_PNN": output is pred only
                - None: deterministic direct output
        uq_parser (callable | None, optional): Function (y_pred, type_output) -> (mean, var_a, var_e).
            If None, uses _default_uq_parse.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            pred: Mean output over draws.
            UQ: Array with shape (2, ...) where:
                UQ[0] = aleatoric variance
                UQ[1] = epistemic variance
    """
    if generator:
        pred = []
        var_a = []
        var_e = []

        for Inputs_gen, _ in Inputs:
            output = []
            for _ in range(n_ech):
                output.append(model.predict(Inputs_gen))

            output = np.array(output)

            # Use custom parser if available, else fallback
            if uq_parser is not None:
                mean_batch, var_a_batch, var_e_batch = uq_parser(output, type_output)
            else:
                if type_output == "MC_Dropout":
                    pred_, logvar = np.split(output, 2, axis=-1)
                    var_a_batch = np.exp(logvar).mean(axis=0)
                else:
                    pred_ = output
                    var_a_batch = np.zeros_like(pred_.mean(axis=0))

                var_e_batch = np.var(pred_, axis=0, ddof=ddof)
                mean_batch = pred_.mean(axis=0)

            pred.append(mean_batch)
            var_a.append(var_a_batch)
            var_e.append(var_e_batch)

        pred = np.concatenate(pred, axis=0)
        var_a = np.concatenate(var_a, axis=0)
        var_e = np.concatenate(var_e, axis=0)

    else:
        output = []
        for _ in range(n_ech):
            output.append(model.predict(Inputs))

        output = np.array(output)

        # Use custom parser if available, else fallback
        if uq_parser is not None:
            mean, var_a, var_e = uq_parser(output, type_output)
        else:
            if type_output == "MC_Dropout":
                pred_, logvar = np.split(output, 2, axis=-1)
                var_a = np.exp(logvar).mean(axis=0)
            else:
                pred_ = output
                var_a = np.zeros_like(pred_.mean(axis=0))

            var_e = np.var(pred_, axis=0, ddof=ddof)
            mean = pred_.mean(axis=0)

    UQ = np.concatenate([var_a[None, :], var_e[None, :]], axis=0)
    return mean, UQ


def Ensemble_based_inference(
    Inputs, models, ddof, generator=False, type_output=None, uq_parser=None
):
    """Run ensemble-based inference on a provided list of models.

    Args:
        Inputs: Model inputs or generator yielding (x, y).
        models (list): List of ensemble members exposing predict(...).
        ddof (int): Delta degrees of freedom for variance estimation.
        generator (bool, optional): Whether Inputs is a generator.
        type_output (str | None, optional): Output interpretation.
            Supported values:
                - "PNN": concatenated output [pred, logvar]
                - None: deterministic direct output
        uq_parser (callable | None, optional): Function (y_pred, type_output) -> (mean, var_a, var_e).
            If None, uses _default_uq_parse.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            pred: Mean output over ensemble members.
            UQ: Array with shape (2, ...) where:
                UQ[0] = aleatoric variance
                UQ[1] = epistemic variance
    """
    if generator:
        pred = []
        var_a = []
        var_e = []

        for Inputs_gen, _ in Inputs:
            output = []
            for submodel in models:
                output.append(submodel.predict(Inputs_gen))

            output = np.array(output)

            # Use custom parser if available, else fallback
            if uq_parser is not None:
                mean_batch, var_a_batch, var_e_batch = uq_parser(output, type_output)
            else:
                if type_output == "PNN":
                    pred_, logvar = np.split(output, 2, axis=-1)
                    var_a_batch = np.exp(logvar).mean(axis=0)
                else:
                    pred_ = output
                    var_a_batch = np.zeros_like(pred_.mean(axis=0))

                var_e_batch = np.var(pred_, axis=0, ddof=ddof)
                mean_batch = pred_.mean(axis=0)

            pred.append(mean_batch)
            var_a.append(var_a_batch)
            var_e.append(var_e_batch)

        pred = np.concatenate(pred, axis=0)
        var_a = np.concatenate(var_a, axis=0)
        var_e = np.concatenate(var_e, axis=0)

    else:
        output = []
        for submodel in models:
            output.append(submodel.predict(Inputs))

        output = np.array(output)

        # Use custom parser if available, else fallback
        if uq_parser is not None:
            mean, var_a, var_e = uq_parser(output, type_output)
        else:
            if type_output == "PNN":
                pred_, logvar = np.split(output, 2, axis=-1)
                var_a = np.exp(logvar).mean(axis=0)
            else:
                pred_ = output
                var_a = np.zeros_like(pred_.mean(axis=0))

            var_e = np.var(pred_, axis=0, ddof=ddof)
            mean = pred_.mean(axis=0)

    UQ = np.concatenate([var_a[None, :], var_e[None, :]], axis=0)
    return mean, UQ

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
                np.random.seed(add_random_state(random_state, n))
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
