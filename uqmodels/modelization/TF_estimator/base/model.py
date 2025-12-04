import tensorflow as tf
import yaml
import os
import copy
from tensorflow import keras
from uqmodels.modelization.TF_estimator.base.train_config import make_optimizer_default,make_callbacks
from uqmodels.modelization.TF_estimator.base.loss import build_MSE_loss
from collections.abc import Mapping

class BaseKModel(keras.Model):
    """
    Mixin / base class that provides:
      - Default keyword arguments for `compile()` and `fit()`
      - Automatic execution of `compile()` right before `fit()` if the model
        has not been compiled yet
      - Merging of default training parameters with those provided at call time
        (call-time kwargs always override defaults)

    This mixin is useful when defining models that should carry
    reusable training configurations and avoid repetitive `compile()`
    and `fit()` boilerplate.

    Parameters
    ----------
    compile_kwargs : dict, optional
        Default keyword arguments passed to `compile()`. These may include
        optimizer, loss function, metrics, etc.
        If omitted, a reasonable default configuration is used.
    
    fit_kwargs : dict, optional
        Default keyword arguments passed to `fit()`. Examples include
        batch_size, epochs, verbose, etc.
        These defaults can be overridden per training call.
    
    Notes
    -----
    - If the model is already compiled, `compile()` will not be called again
      unless explicitly skipped via `skip_compile`.
    - Keyword arguments passed to `fit()` take precedence over stored defaults.
    """

    def __init__(self, 
                 *,
                 name=None,
                 compile_kwargs={},
                 fit_kwargs={},
                 **kwargs):
        
        super().__init__(name=name)
        self._init_keys = list(kwargs.keys())
        self._is_compiled = False
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.build_fit_and_compile_kwargs(compile_kwargs={},fit_kwargs={})

        # --- Base default fit settings (can be extended in subclasses)
        

    def build_fit_and_compile_kwargs(self,compile_kwargs={},fit_kwargs={}):

        if(hasattr(self,'training_params') and ('optimizer' in self.training_params)):
            optimizer = self.training_params['optimizer']
        else:
            optimizer = 'nadam'

        if(hasattr(self,'training_params') and ('loss' in self.training_params)):
            loss = self.training_params['loss']
        else:
            loss = build_MSE_loss(split=1,metric=False)

        if(hasattr(self,'training_params') and ('epochs' in self.training_params)):
            epochs = self.training_params['epochs']
        else:
            epochs =  100
        
        if(hasattr(self,'training_params') and ('batch_size' in self.training_params)):
            batch_size = self.training_params['batch_size']
        else:
            batch_size =  64
         
        self.compile_kwargs = {"optimizer": make_optimizer_default(optimizer),
                               "loss": loss,
                               "metrics": []}
         
        if compile_kwargs:
            self.compile_kwargs.update(compile_kwargs)

        callbacks, configs = make_callbacks()
        self.fit_kwargs = { "epochs": epochs,
                            "batch_size": batch_size,
                            "verbose": "auto",
                            "callbacks":callbacks
                            }
        if fit_kwargs:
            self.fit_kwargs.update(fit_kwargs)

        print(self.compile_kwargs['loss'])

        self.loss_fn  = self.compile_kwargs['loss']


    def _is_model_compiled(self) -> bool:
        """
        Check whether the model is already compiled.

        This method ensures compatibility across TensorFlow / Keras versions.
        
        Returns
        -------
        bool
            True if the model is compiled, False otherwise.
        """
        # Compatibility across TF/Keras versions
        if hasattr(self, "_is_compiled"):
            return bool(self._is_compiled)
        return getattr(self, "compiled_loss", None) is not None


    def fit(self, X, y=None, skip_compile=False, **kwargs):
        """
        Train the model while automatically handling compilation and parameter merging.

        The method:
          1) Runs `compile()` if necessary (unless skip_compile=True)
          2) Merges default fit parameters with call-time parameters
             (call-time parameters override default values)
          3) Delegates execution to `keras.Model.fit()`

        Parameters
        ----------
        X : array-like or tf.data.Dataset
            Training data.
        y : array-like, optional
            Training labels. May be omitted for models such as autoencoders.
        skip_compile : bool, default=False
            If True, does not auto-call `compile()` even if the model
            has not been compiled yet. Useful for manual control.
        **kwargs : dict
            Additional keyword arguments forwarded to `fit()`. These override
            `fit_kwargs`.

        Returns
        -------
        History
            A `keras.callbacks.History` object containing training metrics.
        """

        # 1) Auto-compile if requested and not yet compiled
        if not skip_compile:
            if not self._is_model_compiled():
                compile_kwargs = dict(self.compile_kwargs)
                super().compile(**compile_kwargs)

        fit_kwargs = dict(self.fit_kwargs)
        fit_kwargs.update(kwargs)

        return super().fit(X, y, **fit_kwargs)

    def build_init_config(self):
        """Build a YAML-safe config dict from a model skipping any init key starting with '__'."""

        if not hasattr(self, "_init_keys"):
            raise AttributeError("Model must define `_init_keys` in __init__.")

        init_kwargs = {}
        for key in self._init_keys:

            # skip special/internal attributes (e.g. __class__, __foo__)
            if key.startswith("__"):
                continue

            if not hasattr(self, key):
                continue

            value = getattr(self, key)
            if(isinstance(value, Mapping)):
                value = dict(value)
            init_kwargs[key] = value

        cfg = {
        "class_name": self.__class__.__name__,
        "name": self.name,
        "init_kwargs": init_kwargs}

        # optional hook on the model
        hook = getattr(self, "_extra_init_config", None)
        if callable(hook):
            cfg = hook(cfg)
        return cfg


    def save(self, model_dir: str, **kwargs):
        if not getattr(self, "name", None):
            raise ValueError("`name` must be set to save the model.")
        
        os.makedirs(model_dir, exist_ok=True)

        # build config
        cfg = self.build_init_config()

        # dump YAML
        yaml_path = os.path.join(model_dir, "config.yaml")
        with open(yaml_path, "w") as f:
            yaml.safe_dump(cfg, f)

        # subclass extension hook
        self._extra_save(model_dir)


    @classmethod
    def load(cls, model_dir: str, **override_kwargs):
        """
        Reload a model from weights + YAML config.
        """

        # 1) read YAML config
        yaml_path = os.path.join(model_dir, "config.yaml")
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)

        # optional safety check
        if cfg.get("class_name") != cls.__name__:
            raise ValueError(
                f"Config class_name={cfg.get('class_name')} does not match cls={cls.__name__}"
            )

        init_kwargs = cfg.get("init_kwargs", {}) or {}

        # 2) allow user to override some init params at load time
        init_kwargs.update(override_kwargs)

        init_kwargs = cfg.get("init_kwargs", {})
        compile_kwargs = cfg.get("compile_kwargs", None)
        fit_kwargs = cfg.get("fit_kwargs", None)

        # 3) rebuild instance
        obj = cls(compile_kwargs=compile_kwargs,
                  fit_kwargs=fit_kwargs,
                  **init_kwargs)

        # 5) subclass hook
        obj._extra_load(model_dir)

        return obj

    def _extra_load(self, model_dir: str) -> None:
        """Hook for subclasses."""
        pass

