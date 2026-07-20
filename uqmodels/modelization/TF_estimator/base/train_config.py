import os
from copy import deepcopy
from typing import Dict, Any, List, Tuple, Optional
from tensorflow.keras import callbacks

# Call back procedure
import tensorflow as tf
from tensorflow import keras

def make_optimizer_default(
    name: str = "adam",
    task: str = "deterministic",
    learning_rate: float | None = None,
    beta_1: float | None = None,
    beta_2: float | None = None,
    epsilon: float | None = None,
    momentum: float | None = None,
    nesterov: bool | None = None,
    weight_decay: float | None = None,
):
    """
    Factory for optimizers with sensible defaults per optimizer type,
    especially for time series & uncertainty quantification tasks.

    Parameters
    ----------
    name : str
        One of {"adam", "adamw", "nadam", "sgd"}.
    task : str
        "deterministic", "gaussian", "edl" — tunes LR defaults.
    learning_rate, beta_1, beta_2, epsilon, momentum, nesterov, weight_decay :
        If None, optimizer-specific defaults are used.

    Returns
    -------
    keras.optimizers.Optimizer
    """

    # Default LR depending on task
    if learning_rate is None:
        learning_rate = 1e-3 if task == "deterministic" else 3e-4

    # ---- Adam ----
    if name == "adam":
        if beta_1 is None:   beta_1 = 0.9
        if beta_2 is None:   beta_2 = 0.999
        if epsilon is None:  epsilon = 1e-7
        return keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=beta_1, beta_2=beta_2, epsilon=epsilon
        )

    # ---- AdamW ----
    elif name == "adamw":
        if beta_1 is None:   beta_1 = 0.9
        if beta_2 is None:   beta_2 = 0.999
        if epsilon is None:  epsilon = 1e-8   # 🔁 souvent plus stable pour AdamW
        if weight_decay is None: weight_decay = 1e-4
        return keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta_1=beta_1, beta_2=beta_2, epsilon=epsilon
        )

    # ---- Nadam ----
    elif name == "nadam":
        if beta_1 is None:   beta_1 = 0.9
        if beta_2 is None:   beta_2 = 0.999
        if epsilon is None:  epsilon = 1e-7
        return keras.optimizers.Nadam(
            learning_rate=learning_rate,
            beta_1=beta_1, beta_2=beta_2, epsilon=epsilon
        )

    # ---- SGD ----
    elif name == "sgd":
        if momentum is None: momentum = 0.9
        if nesterov is None: nesterov = True
        return keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=nesterov
        )

    else:
        raise ValueError(f"Unknown optimizer '{name}'. Choose: adam, adamw, nadam, sgd.")

# --- Configuration par défaut : dictionnaire de dictionnaires ---
# - enabled: bool pour activer/désactiver un callback
# - class: constructeur du callback
# - params: kwargs passés au constructeur
DEFAULT_CALLBACKS_CONFIG: Dict[str, Dict[str, Any]] = {
    # ---- TES VALEURS PAR DÉFAUTS ----
    "EarlyStopping": {
        "enabled": True,
        "class": callbacks.EarlyStopping,
        "params": {
            "monitor": "loss",
            "min_delta": 1e-4,
            "patience": 60,          # earlystop_patience
            "verbose": 0,
            "mode": "min",
            "restore_best_weights": False,
        },
    },
    "ReduceLROnPlateau": {
        "enabled": True,
        "class": callbacks.ReduceLROnPlateau,
        "params": {
            "monitor": "loss",        # tu utilisais "loss" ici (et non "val_loss")
            "min_delta": 1e-4,
            "factor": 0.3,            # reducelr_factor
            "patience": 30,           # reducelr_patience
            "verbose": 0,
            "mode": "min",
            "cooldown": 0,
            "min_lr": 1e-6,           # reduce_lr_min_lr
        },
    },
    "TerminateOnNaN": {
        "enabled": True,
        "class": callbacks.TerminateOnNaN,
        "params": {},                 # pas de paramètres
    },

    # ---- Optionnels (désactivés par défaut) ----
    "ModelCheckpoint": {
        "enabled": False,
        "class": callbacks.ModelCheckpoint,
        "params": {
            "filepath": os.path.join("checkpoints", "best.keras"),
            "monitor": "val_loss",
            "mode": "min",
            "save_best_only": True,
            "save_weights_only": False,
            "verbose": 0,
        },
    },
    "TensorBoard": {
        "enabled": False,
        "class": callbacks.TensorBoard,
        "params": {
            "log_dir": os.path.join("checkpoints", "tb_logs"),
            "histogram_freq": 0,
            "write_graph": True,
        },
    },
}


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Mise à jour récursive d'un dictionnaire (overrides prime)."""
    out = deepcopy(base)
    for k, v in (overrides or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def make_callbacks(
    config_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    create_dirs: bool = True,
    return_config: bool = False,
) -> Tuple[List[callbacks.Callback], Optional[Dict[str, Dict[str, Any]]]]:
    """
    Construit une liste de callbacks Keras à partir d'une configuration
    (dictionnaire de dictionnaires), en appliquant d'éventuelles surcharges.

    Parameters
    ----------
    config_overrides : dict[str, dict], optional
        Dictionnaire partiel avec les clés des callbacks à modifier.
        Exemple:
            {
              "EarlyStopping": {
                 "enabled": True,
                 "params": {"patience": 80, "restore_best_weights": True}
              },
              "ModelCheckpoint": {
                 "enabled": True,
                 "params": {"filepath": "ckpts/best.keras", "monitor": "val_mae"}
              }
            }
    create_dirs : bool, default=True
        Crée les répertoires nécessaires (filepath/log_dir) si présents.
    return_config : bool, default=False
        Si True, retourne aussi la configuration finale (après merge).

    Returns
    -------
    callbacks_list : list[tf.keras.callbacks.Callback]
        Liste ordonnée des callbacks instanciés.
    final_config : dict[str, dict] or None
        Config résultante (si return_config=True).
    """
    # Ordre stable : tu peux modifier l'ordre ici si tu préfères
    order = ["EarlyStopping", "ReduceLROnPlateau", "TerminateOnNaN",
             "ModelCheckpoint", "TensorBoard"]

    # Merge: défauts -> overrides
    final_config = _deep_update(DEFAULT_CALLBACKS_CONFIG, config_overrides or {})

    # Crée les dossiers si nécessaire
    if create_dirs:
        # ModelCheckpoint
        mc = final_config.get("ModelCheckpoint", {})
        if mc.get("enabled", False):
            fp = mc.get("params", {}).get("filepath")
            if fp:
                os.makedirs(os.path.dirname(fp), exist_ok=True)
        # TensorBoard
        tb = final_config.get("TensorBoard", {})
        if tb.get("enabled", False):
            ld = tb.get("params", {}).get("log_dir")
            if ld:
                os.makedirs(ld, exist_ok=True)

    # Instanciation des callbacks
    cbs: List[callbacks.Callback] = []
    for name in order:
        spec = final_config.get(name)
        if not spec or not spec.get("enabled", False):
            continue
        cls = spec["class"]
        params = spec.get("params", {})
        cbs.append(cls(**params))

    return (cbs, final_config) if return_config else (cbs, None)

def add_callback(callbacks_list: List[tf.keras.callbacks.Callback],   
                 callback: Optional[tf.keras.callbacks.Callback] = None,    
                 callback_class: Optional[type] = None,    
                 callback_params: Optional[Dict[str, Any]] = None,    
                 position: str = "end",
                 index: Optional[int] = None,) -> List[tf.keras.callbacks.Callback]:    
    """    
    Ajoute un callback à une liste existante retournée par `make_callbacks`,    de manière totalement générique.
    
    Paramètres    
    ----------    
    callbacks_list : list of Callback        Sortie de make_callbacks (liste existante à enrichir).
    callback : tf.keras.callbacks.Callback, optional        Callback déjà instancié (alternative à callback_class).
    callback_class : type, optional        Classe du callback à instancier.
    callback_params : dict, optional        Paramètres pour instancier callback_class.
    position : {"start", "end", "index"}, default="end"        Où insérer le callback :        - "start" → au début        - "end" → à la fin        - "index" → utilise l'argument `index`
    index : int, optional        Position spécifique si position="index".
    
    Retour    
    ------    
    list[Callback]        Nouvelle liste enrichie. 
    """
    # --- validation minimale ---    
    if callback is None and callback_class is None:        
        raise ValueError("Provide either 'callback' or 'callback_class'.")
    # --- instanciation si nécessaire ---    
    if callback is None:        
        callback_params = callback_params or {}        
        callback = callback_class(**callback_params)
    # --- insertion ---    
    if position == "start":        
        callbacks_list.insert(0, callback)
    elif position == "end":        
        callbacks_list.append(callback)
    elif position == "index":        
        if index is None:            
            raise ValueError("When using position='index', provide `index`.")        
        callbacks_list.insert(index, callback)
    else:        
        raise ValueError(f"Unknown position: {position}")
    return callbacks_list