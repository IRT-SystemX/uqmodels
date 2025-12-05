import tensorflow as tf
import numpy as np
from uqmodels.utils import apply_mask


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
            # X est supposé être une liste [X0, X1]
            return [self.X[0][idx_min:idx_max], self.X[1][idx_min:idx_max]], y_batch

    def __len__(self):
        """Nombre de batches par epoch."""
        return self.n_batch

    def __getitem__(self, idx):
        if self.shuffle:
            idx = self.indices[idx]

        x, y = self.load(idx)

        Inputs, Outputs, _ = self.factory(x, y, fit_rescale=False)

        selection = np.zeros(len(Inputs[0]), dtype=bool)
        idx_min = max(0, idx * self.batch - self.past_horizon)
        idx_max = max(
            self.size_seq + idx_min,
            idx * self.batch + self.batch + self.futur_horizon,
        )

        if self.train:
            selection[self.past_horizon: -self.futur_horizon] = True
        else:
            idx_min = max(0, idx * self.batch - self.past_horizon)
            idx_max = max(
                self.size_seq + idx_min,
                idx * self.batch + self.batch + self.futur_horizon,
            )

            if idx == 0:
                if self.batch >= self.len_:
                    selection[:] = True
                else:
                    selection[: -self.past_horizon - self.futur_horizon] = True
            else:
                padding_test = max(self.futur_horizon, idx_max - self.len_)
                selection[padding_test + self.past_horizon:] = True

        Inputs = apply_mask(Inputs, selection)
        Outputs = apply_mask(Outputs, selection)

        # Hold multi-input case
        if isinstance(Inputs, (list, tuple)):
            Inputs = tuple(np.asarray(xi) for xi in Inputs)
        else:
            Inputs = np.asarray(Inputs, dtype=self.dtype)

        Outputs = np.asarray(Outputs, dtype=self.dtype)

        return Inputs, Outputs

    def on_epoch_end(self):
        """Shuffle des batches à la fin de chaque epoch."""
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(self.indices)
