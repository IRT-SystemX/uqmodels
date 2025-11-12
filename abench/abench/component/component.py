from abc import ABC, abstractmethod
import importlib, inspect, numpy as np
from pathlib import Path
from abench.store.api import read,write

# Encapsulated model format :
class Component(ABC):
    """Abstract Encapsulated Model class :
    Allow generic manipulation of models"""

    def __init__(self, **kwarg):
        """init procedure

        Warning : abench store submodule model in a dict{'initializer':initializer,'paramaters':paramaters}
        In order to build submodule model in the meta model, use such procedure :

        for each submodule do :
        __init__(self,submodule_1= stored_submodule_1) where stored_submodule_1 is provided by abench
        if(submodule_1['parameters'] is None): # If submodule is store as model
            self.submodule_1 = submodule_1["initializer"]
        else:  # If submodule is store as (initializer,parameters)
            self.submodule_1 = submodule_1["initializer"](**submodule_1['parameter'])


        """

        pass

    def _tuning(self, X, y, **kwarg):
        """Tunning procedure

        Args:
            X (array): Inputs
            y (array): Targets
            context (array): Contextual complementary informations
        """
        pass

    def fit(self, X, y, **kwarg):
        """Fitting procedure

        Args:
            X (array): Inputs
            y (array): Targets
            context (array): Additional information
        """
        pass

    def predict(self, X, **kwarg):
        """Predict procedure

        Args:
            X (array): Inputs
             context (array): Contextual complementary information

        Returns:
            output : Encapsulated results format
        """
        output = None
        return output

    def save(self,storing,keys):
        pass
        
    def load(self,storing,keys):
        pass

class GenericComponent(Component):
    """
    A light wrapper around *any* model that exposes a scikit‑learn‑like API.
    """

    # ------------------------------------------------------------------
    # ctor (unchanged except we keep the spec so we can re‑emit it later)
    # ------------------------------------------------------------------
    def __init__(self, model={'initializer': None, 'parameters': None}, **kwargs):
        self._model_spec = {          # remember how the model is built
            'initializer': model['initializer'],
            'parameters' : model['parameters']}

        if model['parameters'] is None:            # model object given directly
            self.model = model['initializer']
        else:                                      # rebuild from (callable, kwargs)
            self.model = model['initializer'](**model['parameters'])

    # ------------------------------------------------------------------
    # learners' API
    # ------------------------------------------------------------------
    def fit(self, X, y, **kwargs):
        self.model.fit(X, y)
        return self

    def predict(self, X, y=None, **kwargs):
        return self.model.predict(X)

    # ------------------------------------------------------------------
    # persistence helpers
    # ------------------------------------------------------------------
    def _export_config(self):
        """
        What we need to resurrect this component later:
          • where the model class lives        (module + qual‑name)
          • the arguments originally passed to it
        """
        init = self._model_spec['initializer']
        if inspect.isclass(init):     # typical case: class object
            class_path = f"{init.__module__}.{init.__name__}"
        else:                         # any callable (factory, fn, partial…)
            class_path = f"{init.__module__}.{init.__name__}"

        return {
            'class_path': class_path,
            'parameters': self._model_spec['parameters']
        }

    # --------------- public save / load -------------------------------
    def save(self, storing, keys):
        """
        Layout on disk / dict:

        <keys>/config   -> metadata dict  (.p)      via write()
        <keys>/model    ->             • if sub‑model has .save() -> delegated tree
                                       • else entire object (.joblib / .p) via write()
        """
        # 1) config
        write(storing, keys + ['config'], self._export_config())

        # 2) model (state)
        if hasattr(self.model, 'save'):
            self.model.save(storing, keys + ['model'])
        else:
            write(storing, keys + ['model'], self.model)
    
    @classmethod
    def load(cls, storing, keys=[]):
        """
        Rebuild *in place* – first the skeleton from config, then the learned state.
        """
        # ----- config -----
        cfg = read(storing, keys + ['config'])
        if cfg is None:
            raise FileNotFoundError("GenericComponent: config file not found")

        module_name, class_name = cfg['class_path'].rsplit('.', 1)
        model_cls = getattr(importlib.import_module(module_name), class_name)
        params    = cfg['parameters'] or {}


        model = {'initializer': model_cls, 'parameters': params}
        component = cls(model)

        # ----- state -----
        if hasattr(component.model, 'load'):
            component.model = component.model.load(storing, keys + ['model'])
        else:
            # entire fitted object was pickled/joblib‑ed
            component.model = read(storing, keys + ['model'])

        return component   # for chaining

    # ------------------------------------------------------------------
    # misc utils
    # ------------------------------------------------------------------
    def get_params(self, deep=True):
        """Expose parameters like scikit‑learn estimators do."""
        return {'model': self._model_spec}