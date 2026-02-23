import tensorflow as tf
import numpy as np

import random
from uqmodels.utils import apply_mask,identity

class default_Generator(tf.keras.utils.Sequence):
    def __init__(
        self, X, y, metamodel, batch=64, shuffle=True, train=True, random_state=None
    ):
        """
        Standard batch Sequence generator for supervised learning.
        Builds batches from X and y, applies metamodel preprocessing via factory,
        and returns fixed-shape input/output arrays compatible with Keras training.
        """

        self.X = X
        self.y = y
        self.len_ = len(y)              # nombre d'exemples
        self.train = train
        self.random_state = random_state
        self.shuffle = shuffle
        self.batch = batch

        self.factory = metamodel.factory
        self._format = metamodel._format
        self.rescale = metamodel.rescale

        # indices d'échantillons
        self.indices = np.arange(self.len_)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(self.indices)

    def __len__(self):
        """Nombre de batches par epoch."""
        return int(np.ceil(self.len_ / self.batch))

    def __getitem__(self, idx):
        """Retourne le batch idx (Inputs, Outputs) sous forme de np.ndarray."""
        # idx : index de batch (0, 1, 2, ...)
        start = idx * self.batch
        end = min((idx + 1) * self.batch, self.len_)

        batch_indices = self.indices[start:end]

        # batch brut
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]

        # factory renvoie (X_transformed, y_transformed, mask)
        X_trans, y_trans, _ = self.factory(X_batch, y_batch)

        # on force en np.ndarray pour que Keras / tf.data puissent
        # inférer un output_signature propre
        X_trans = np.asarray(X_trans)
        y_trans = np.asarray(y_trans)

        return X_trans, y_trans

    def on_epoch_end(self):
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(self.indices)

class Folder_Generator(tf.keras.utils.Sequence):
    def __init__(
        self, X, y, metamodel, batch=64, shuffle=True, train=True, random_state=None,
        dtype=np.float32
    ):
        """
        Folder-based Sequence generator producing sliding-window batches for temporal models.
        Extracts past and future context around each batch, applies metamodel formatting,
        and returns masked input/output sequences compatible with Keras training and inference.
        """
        self.X = X
        self.y = y
        self.random_state = random_state
        self.dtype = np.float32
        if X is not None:
            # X est supposé être une liste/tuple de arrays : [X0, X1, ...]
            self.len_ = X[0].shape[0]
        elif y is not None:
            self.len_ = y.shape[0]
        else:
            raise ValueError("Folder_Generator requires at least X or y to be non-None.")

        self.train = train
        self.shuffle = shuffle
        self.batch = batch

        self.factory = metamodel.factory
        self.factory_parameters = metamodel.factory_parameters
        self._format = metamodel._format
        self.rescale = metamodel.rescale

        self.causality_remove = None
        self.model_parameters = metamodel.model_parameters
        self.past_horizon = metamodel.model_parameters["size_window"]
        self.futur_horizon = (
            metamodel.model_parameters["dim_horizon"]
            * metamodel.model_parameters["step"]
        )
        self.size_seq = self.past_horizon + self.futur_horizon + self.batch
        self.size_window_futur = 1

        # nombre de batches
        self.n_batch = int(np.ceil(self.len_ / self.batch))

        # indices de batches (0, 1, ..., n_batch-1) pour le shuffle
        self.indices = np.arange(self.n_batch)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(self.indices)

    def load(self, idx):
        """
        Charge la séquence de données centrée autour du batch idx :
        [idx * batch - past_horizon, idx * batch + futur_horizon]
        """
        idx = idx * self.batch

        idx_min = max(0, idx - self.past_horizon)
        idx_max = max(self.size_seq + idx_min, idx + self.futur_horizon)

        # cas du dernier batch : on peut remonter un peu pour compléter la fenêtre
        if idx > 0:
            idx_min = max(idx_min - max(0, idx_max - self.len_), 0)

        y_batch = None
        if self.y is not None:
            y_batch = self.y[idx_min:idx_max]

        if self.X is None:
            return [None, None], y_batch
        else:
            # Gestion polymorphique de X -> 
            if(type(self.X) is list):
                input = [array[idx_min:idx_max] for array in self.X]
                if len(input)==1:
                    input = input[0]
            else:
                input = self.X[idx_min:idx_max]

            print(input.shape)
            return input, y_batch

    def __len__(self):
        """Nombre de batches par epoch."""
        return self.n_batch

    def __getitem__(self, idx):
        if self.shuffle:
            idx = self.indices[idx]

        x, y = self.load(idx)
    
        Inputs, Outputs, _ = self.factory(x, y, fit_rescale=False)

        if('skip' in self.factory_parameters.keys() and self.factory_parameters['skip']):
            selection = np.ones(len(Inputs[0]), dtype=bool)

        else:
            # TO DO : A replacer coté factory respective.
            selection = aux_compute_sliding_mask(
            n_items=len(Inputs[0]),
            idx=idx,
            batch=self.batch,
            past_horizon=self.past_horizon,
            futur_horizon=self.futur_horizon,
            size_seq=self.size_seq,
            len_=self.len_,
            train=self.train,
            )

        Inputs = apply_mask(Inputs, selection)
        Outputs = apply_mask(Outputs, selection)

        # Hold multi-input case
        if isinstance(Inputs, (list, tuple)):
            Inputs = tuple(np.asarray(xi) for xi in Inputs)
        else:
            Inputs = np.asarray(Inputs,dtype=self.dtype)

        Outputs = np.asarray(Outputs,dtype=self.dtype)

        return Inputs, Outputs

    def on_epoch_end(self):
        """Shuffle des batches à la fin de chaque epoch."""
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(self.indices)


def aux_compute_sliding_mask(
    n_items: int,
    *,
    idx: int,
    batch: int,
    past_horizon: int,
    futur_horizon: int,
    size_seq: int,
    len_: int,
    train: bool,
) -> np.ndarray:
    """
    Build a boolean mask for windowed time-series samples produced by a sliding-window factory.

    This helper is designed to be **retro-compatible** with the legacy inlined masking logic:
    - In training mode, it keeps only indices that have a full past and full future context:
        selection[past_horizon : -futur_horizon] = True
    - In evaluation mode, it attempts to return only the subset of indices associated to the
      batch number `idx`, while handling boundary/padding effects near sequence start/end.

    Parameters
    ----------
    n_items : int
        Number of windowed items produced by the factory (typically len(Inputs[0])).
    idx : int
        Batch index.
    batch : int
        Batch size (in number of windowed items).
    past_horizon : int
        Required past context length (number of steps).
    futur_horizon : int
        Required future horizon length (number of steps).
    size_seq : int
        Minimal segment length used by the legacy computation of idx_max.
    len_ : int
        Total available number of windowed items in the underlying sequence (legacy: self.len_).
    train : bool
        Whether we are in training mode (legacy: self.train).

    Returns
    -------
    np.ndarray, dtype=bool, shape (n_items,)
        Boolean selection mask to apply on Inputs / Outputs.

    Notes
    -----
    - The logic intentionally mirrors the original behavior, including edge-case handling.
    - `idx_min` is computed for completeness/traceability but is not directly used in masking,
      matching the legacy code structure.
    """
    if n_items < 0:
        raise ValueError("n_items must be non-negative.")
    if batch <= 0:
        raise ValueError("batch must be > 0.")
    if past_horizon < 0 or futur_horizon < 0:
        raise ValueError("past_horizon and futur_horizon must be >= 0.")
    if size_seq < 0:
        raise ValueError("size_seq must be >= 0.")
    if len_ < 0:
        raise ValueError("len_ must be >= 0.")
    if idx < 0:
        raise ValueError("idx must be >= 0.")

    selection = np.zeros(n_items, dtype=bool)

    # ---- Train: keep only indices with full past & future -------------------
    if train:
        # Guard against empty slices when horizons are larger than n_items.
        start = min(past_horizon, n_items)
        end = n_items - futur_horizon
        if end < start:
            return selection  # all False
        selection[start:end] = True
        return selection

    # ---- Eval/Test: legacy batch-aware selection ----------------------------
    # Legacy idx_min / idx_max computations (kept for retro-compat behavior).
    idx_min = max(0, idx * batch - past_horizon)
    idx_max = max(
        size_seq + idx_min,
        idx * batch + batch + futur_horizon,
    )

    if idx == 0:
        if batch >= len_:
            selection[:] = True
        else:
            # Equivalent to legacy: selection[: -past_horizon - futur_horizon] = True
            cut = past_horizon + futur_horizon
            if cut == 0:
                selection[:] = True
            else:
                end = n_items - cut
                if end > 0:
                    selection[:end] = True
    else:
        padding_test = max(futur_horizon, idx_max - len_)
        start = padding_test + past_horizon
        if start < n_items:
            selection[start:] = True

    return selection