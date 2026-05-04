"""
Base Autoencoder with hook-based training.

- Provides a generic Keras Model with:
  - metrics (loss, reconstruction_loss)
  - train_step / test_step relying on a single hook: forward_and_losses(data)
  - default reconstruction loss (MSE) for non-sequential data
- Mixins can override _get_reconstruction_loss or forward_and_losses to change behavior.
"""

import os
import json
import yaml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers, metrics
from uqmodels.modelization.TF_estimator.base.model import BaseKModel


class BaseAutoencoder(BaseKModel):
    def __init__(self, *,name='AE',**kwargs):
        super().__init__(name=name,**kwargs)
        # Core models must be defined by subclasses/mixins:
        # self.encoder, self.decoder

        # Trackers
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.total_loss_tracker = metrics.Mean(name="loss")

    # ---------- Metrics list ----------
    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker]

    # ---------- Losses ----------
    def _seq_reduce_axes(self, tensor):
        return tf.range(1, tf.rank(tensor))

    def _get_reconstruction_loss(self, y_true, y_pred):
        """
        Default MSE reconstruction loss for non-sequence tensors:
        reduce-sum over feature dims, then mean over batch.
        """
        #sq = tf.square(y_true - y_pred)
        if(self.loss_fn is None):
            sq = tf.square(y_true - y_pred)
        else:
            sq = self.loss_fn(y_true, y_pred)
        # sum over all non-batch dims
        if (int(sq.shape.rank) == 0):
            return tf.reduce_mean(sq)
        else:
            axes = self._seq_reduce_axes(sq)
            per_example = tf.reduce_mean(sq, axis=axes)
            return tf.reduce_mean(per_example)

    # ---------- Hook to implement ----------
    def forward_and_losses(self, data):
        """
        Must return: total_loss (scalar), logs (dict of tensors)
        Implementations typically:
          - unpack data
          - forward pass (encoder/decoder)
          - compute reconstruction (and other) losses
        """
        # Accepte data = X ou (X, y); on ignore y ici (utile si on compose ensuite un HybridMixin)
        X = data[0] if isinstance(data, (tuple, list)) else data

        # 1) encode
        z = self.encode(X, training=True)

        # 2) decode
        X_recon = self.decode(z,training=True)

        # 3) losses
        recon_loss = self._get_reconstruction_loss(X, X_recon)  # séquentiel ou non -> géré par la classe/mixin parent
        
        logs = {"reconstruction_loss": recon_loss}
        return recon_loss, logs

    # ---------- Keras steps ----------
    def train_step(self, data):

        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        # Cas autoencodeur: si y n'est pas fourni, on prédit la reconstruction de x
        if y is None:
            y = x

        with tf.GradientTape() as tape:
            total_loss, logs = self.forward_and_losses(data)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # update base trackers
        self.total_loss_tracker.update_state(total_loss)
        if "reconstruction_loss" in logs:
            self.reconstruction_loss_tracker.update_state(logs["reconstruction_loss"])

        # update any extra trackers exposed by mixins
        for name, value in logs.items():
            tracker = getattr(self, f"{name}_tracker", None)
            if tracker is not None and name != "reconstruction_loss":
                tracker.update_state(value)

        # build result dict with tracker .result() when available
        result = {"loss": self.total_loss_tracker.result()}
        for name, value in logs.items():
            tracker = getattr(self, f"{name}_tracker", None)
            result[name] = tracker.result() if tracker is not None else value
        return result

    def test_step(self, X):
        data = X

        total_loss, logs = self.forward_and_losses(data)

        self.total_loss_tracker.update_state(total_loss)
        if "reconstruction_loss" in logs:
            self.reconstruction_loss_tracker.update_state(logs["reconstruction_loss"])

        for name, value in logs.items():
            tracker = getattr(self, f"{name}_tracker", None)
            if tracker is not None and name != "reconstruction_loss":
                tracker.update_state(value)

        result = {"loss": self.total_loss_tracker.result()}
        for name, value in logs.items():
            tracker = getattr(self, f"{name}_tracker", None)
            result[name] = tracker.result() if tracker is not None else value
        return result

    # ---------- Convenience ----------
    def encode(self, x, **kwargs):
        return self.encoder(x, **kwargs)

    def decode(self, z, **kwargs):
        return self.decoder(z, **kwargs)

    def predict(self, x, **kwargs):
        return self.decode(self.encode(x, **kwargs), **kwargs)

    def summary(self, print_fn=print):
        print_fn("\nEncoder:")
        self.encoder.summary()
        print_fn("\nDecoder:")
        self.decoder.summary()

    # --- hooks no-op, à chaîner par les mixins / classes dérivées ---
    def _extra_save(self, model_dir: str) -> None:
        pass

    def _extra_load(self, model_dir: str) -> None:
        pass

    # --- save/load compacts ---
    def save(self, model_dir: str, **kwargs) -> None:
            super().save(model_dir, **kwargs)

            enc_path = os.path.join(model_dir,"enc.weights.h5")
            dec_path = os.path.join(model_dir,"dec.weights.h5")
            self.encoder.save_weights(enc_path)
            self.decoder.save_weights(dec_path)
            self._extra_save(model_dir)
            
    @classmethod
    def load(cls, model_dir: str, **override_kwargs):
        obj = super().load(model_dir=model_dir, **override_kwargs)

        if not hasattr(obj, "encoder") or not hasattr(obj, "decoder"):
            raise AttributeError(
                "Loaded object must define `encoder` and `decoder` attributes "
                "before BaseAutoencoder.load can restore their weights."
            )

        # 2) load encoder / decoder weights
        enc_path = os.path.join(model_dir, "enc.weights.h5")
        dec_path = os.path.join(model_dir, "dec.weights.h5")

        if os.path.exists(enc_path):
            obj.encoder.load_weights(enc_path)
        else:
            raise FileNotFoundError(f"Encoder weights not found: {enc_path}")

        if os.path.exists(dec_path):
            obj.decoder.load_weights(dec_path)
        else:
            raise FileNotFoundError(f"Decoder weights not found: {dec_path}")

        # 3) optional subclass hook
        obj._extra_load(model_dir)
        return obj



class Sampling(layers.Layer):
    """z = mean + exp(0.5 * log_var) * eps (broadcast-friendly)."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

"""
Variational mixin:
- Sampling layer that is shape-agnostic: works for (B,D), (B,T,D), (B,T,H,W,D), etc.
- KL divergence reduced over all non-batch dims, then mean over batch.
- Provides helper to compute total loss with KL weighting.
"""

class VariationalMixin:
    def __init__(self, kl_weight: float = 0.0, **kwargs):
        super().__init__(**kwargs)  # plays well in multiple inheritance
        self.beta = tf.Variable(float(kl_weight), trainable=False)

        # optional tracker (created lazily if used by a subclass)
        # self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    # --- helpers ---
    def _kl_axes(self, tensor):
        # sum across all non-batch axes
        return tf.range(1, tf.rank(tensor))

    def _get_kl_loss(self, z_mean, z_log_var):
        """
        Element-wise KL to N(0, I):
            -0.5 * (1 + log_var - mu^2 - exp(log_var))
        Then sum over non-batch dims and mean over batch.
        """
        kl_term = -0.5 * (1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        if hasattr(self, "free_bits_lambda"):
            kl_term = tf.maximum(kl_term, self.free_bits_lambda)
        per_example = tf.reduce_sum(kl_term, axis=self._kl_axes(kl_term))
        return tf.reduce_mean(per_example)

    def _sample_latent(self, z_mean, z_log_var):
        return Sampling()([z_mean, z_log_var])

    def compute_total_with_kl(self, recon_loss, z_mean, z_log_var):
        kl_term = self._get_kl_loss(z_mean, z_log_var)

        if hasattr(self, "free_bits_lambda"):
            kl_term = tf.maximum(kl_term, self.free_bits_lambda)

        total = recon_loss + self.beta * tf.reduce_mean(kl_term)
        return total, kl_term
    
"""
Hybrid mixin:
- Ajoute une tête supervisée (classification ou forecasting) sur le latent.
- Compatible 2D (B,D) ou 3D (B,T,D).
- Ne présuppose pas l'archi: construit une tête par défaut si self.classifier est None.
"""

class BaseVariationalAutoencoder(VariationalMixin, BaseAutoencoder):
    """
    Classe 'confort' pour VAE:
      - suppose par défaut: encoder(X) -> (z_mean, z_log_var, z[, *extras])
      - decoder reçoit par défaut z -> X_recon
      - calcule total = recon + kl_weight * KL
    Si votre archi a des 'extras' (ex: U-Net skips), vous pouvez soit:
      - faire que decoder prenne [z] + extras (ça marchera sans override),
      - soit surcharger _decode().
    """

    def __init__(self,*,kl_weight: float = 1.0, name='VAE',**kwargs):
        super().__init__(name=name, kl_weight=kl_weight,**kwargs)
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    # ---------- metrics ----------
    @property
    def metrics(self):
        # BaseAutoencoder expose (loss, reconstruction_loss); on ajoute KL
        return super().metrics + [self.kl_loss_tracker]

    # ---------- hooks (surchargables au besoin) ----------
    def _encode(self, X, training: bool):
        """Par défaut, on appelle directement l'encoder."""
        z_mean, z_log_var = self.encoder(X, training=training)
        z = self._sample_latent(z_mean, z_log_var)
        return [z_mean, z_log_var, z]

    def _decode(self, z, extras: list, training: bool):
        """
        Par défaut:
          - s'il y a des extras, on les passe au decoder avec z: decoder([z] + extras)
          - sinon: decoder(z)
        Surclassez si votre decoder attend un format différent.
        """
        if extras:
            return self.decoder([z] + extras, training=training)
        return self.decoder(z, training=training)

    # ---------- cœur du calcul des pertes ----------
    def forward_and_losses(self, data):
        # Accepte data = X ou (X, y); on ignore y ici (utile si on compose ensuite un HybridMixin)
        X = data[0] if isinstance(data, (tuple, list)) else data

        # 1) encode
        z_mean, z_log_var, z, *extras = self._encode(X, training=True)

        # 2) decode
        X_recon = self._decode(z, extras, training=True)

        # 3) losses
        recon_loss = self._get_reconstruction_loss(X, X_recon)  # séquentiel ou non -> géré par la classe/mixin parent
        total, kl_loss = self.compute_total_with_kl(recon_loss, z_mean, z_log_var)

        logs = {"reconstruction_loss": recon_loss, "kl_loss": kl_loss}
        return total, logs
    
    def predict(self, X, **kwargs):
        z_mean, z_log_var, z, *extras = self._encode(X, training=True)
        X_recon = self._decode(z, extras, training=True)
        return X_recon


class HybridMixin:
    """
    Mixin supervision (classif/forecast) agnostique des dims de z:
      - z: (B, D) ou (B, T, D)
    Options:
      - target_sequence: True => prédire une séquence (TimeDistributed)
      - pooling: "mean" | "last" | "flatten" (si target_sequence=False et z 3D)
      - supervised_loss: défaut = CategoricalCrossentropy (classification)
      - activation: None => auto (softmax si CCE from_logits=False)
      - from_logits: si True, pas d'activation finale (à gérer dans la loss)
      - sup_weight: poids de la tête supervisée dans la loss totale
    Expose:
      - compute_supervised(y_true, z, training) -> (sup_loss, y_pred)
      - add_supervised_to_total(base_total, sup_loss) -> total
      - metrics: ajoute supervised_loss (+ accuracy auto si CCE & softmax & cible globale)
    """

    def __init__(
        self,
        name: str,
        n_outputs: int,
        sup_weight: float = 1.0,
        supervised_loss: tf.keras.losses.Loss | None = None,
        from_logits: bool = False,
        target_sequence: bool = False,
        pooling: str = "mean",
        activation: str | None = None,   # None => auto
        **kwargs,
    ):
        super().__init__(name=name,**kwargs)
        self.n_outputs = int(n_outputs)
        self.sup_weight = float(sup_weight)
        self.target_sequence = bool(target_sequence)
        self.pooling = pooling
        self.from_logits = bool(from_logits)

        # Déterminer loss par défaut
        if supervised_loss is None:
            supervised_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=self.from_logits)
            if activation is None and not self.from_logits:
                activation = "softmax"
        self.supervised_loss_fn = supervised_loss

        # Déterminer activation par défaut si pas fournie (régression p.ex.)
        if activation is None and not isinstance(self.supervised_loss_fn, tf.keras.losses.CategoricalCrossentropy):
            activation = None
        self._sup_activation = activation

        # Tête (créée à la volée si non posée par la sous-classe)
        self.classifier: tf.keras.Model | None = None

        # Trackers
        self.supervised_loss_tracker = tf.keras.metrics.Mean(name="supervised_loss")

        # Accuracy auto si c'est bien de la classification "probabiliste" non-séquentielle
        self._has_acc = (
            isinstance(self.supervised_loss_fn, tf.keras.losses.CategoricalCrossentropy)
            and not self.from_logits
            and self._sup_activation == "softmax"
            and not self.target_sequence
        )
        if self._has_acc:
            self.supervised_acc_tracker = tf.keras.metrics.CategoricalAccuracy(name="supervised_acc")

    @property
    def metrics(self):
        base = super().metrics
        extra = [self.supervised_loss_tracker]
        if self._has_acc:
            extra.append(self.supervised_acc_tracker)
        # éviter doublons si super() inclut déjà ces trackers
        base_ids = {id(m) for m in base}
        return base + [m for m in extra if id(m) not in base_ids]

    # ----------------- Construction par défaut de la tête ----------------- #
    def _ensure_classifier(self, z_sample, training=True):
        if self.classifier is not None:
            return

        z_shape = tuple(z_sample.shape[1:].as_list())  # sans batch
        rank = len(z_shape)
        if rank not in (1, 2):
            raise ValueError(f"z attendu (B,D) ou (B,T,D), reçu: rank={rank+1}, shape={z_sample.shape}")

        inp = tf.keras.Input(shape=z_shape)
        x = inp

        if self.target_sequence:
            if rank != 2:  # (T,D)
                raise ValueError("target_sequence=True attend z de rang 3 (B,T,D).")
            x = layers.TimeDistributed(layers.Dense(64, activation="relu"))(x)
            out = layers.TimeDistributed(layers.Dense(self.n_outputs, activation=(None if self.from_logits else self._sup_activation)))(x)
        else:
            # cible globale: si z est séquentiel -> pooling configurable
            if rank == 2:  # (T,D)
                if self.pooling == "mean":
                    x = tf.reduce_mean(x, axis=0)       # (T,D) -> (D)
                elif self.pooling == "last":
                    x = x[-1]                            # (D)
                elif self.pooling == "flatten":
                    x = layers.Flatten()(tf.expand_dims(x, 0))[0]  # (T*D)
                else:
                    raise ValueError(f"pooling inconnu: {self.pooling}")
            x = layers.Dense(64, activation="relu")(x)
            out = layers.Dense(self.n_outputs, activation=(None if self.from_logits else self._sup_activation))(x)

        self.classifier = tf.keras.Model(inp, out, name="supervised_head")

    # ------------------------- API de calcul ------------------------- #
    def compute_supervised(self, z, y_true, training=True):
        self._ensure_classifier(z, training=training)
        y_pred = self.classifier(z, training=training)
        sup_loss = self.supervised_loss_fn(y_true, y_pred)
        return sup_loss, y_pred

    def add_supervised_to_total(self, base_total, sup_loss):
        return base_total + self.sup_weight * sup_loss

    def update_supervised_metrics(self, sup_loss, y_true=None, y_pred=None):
        self.supervised_loss_tracker.update_state(sup_loss)
        if hasattr(self, "supervised_acc_tracker") and (y_true is not None) and (y_pred is not None):
            self.supervised_acc_tracker.update_state(y_true, y_pred)

class HybridVAE(HybridMixin, BaseVariationalAutoencoder):
    """
    Ajoute la supervision à la logique VAE 'standard' de BaseVariationalAutoencoder.
    """
    def __init__(name='HybrideVae',
                 n_outputs=2,
                 sup_weight=1.0,
                 supervised_loss=None,
                 from_logits=False,
                 target_sequence=False,
                 pooling="mean",
                 activation=None,
                 **kwargs):
        
        super().__init__(name=name,
                         n_outputs=n_outputs,
                         sup_weight=sup_weight,
                         supervised_loss=supervised_loss,
                         from_logits=from_logits,
                         target_sequence=target_sequence,
                         pooling=pooling,
                         activation=activation,
                         **kwargs)


    def forward_and_losses(self, data):
        X, y = data
        # on réutilise le forward VAE standard
        total_wo_sup, logs = super().forward_and_losses(X)
        # récupérons z (on doit réencoder; si tu veux éviter un second passage,
        # surclasse _encode pour mettre z en cache)
        enc_out = self._encode(X, training=True)
        z = enc_out[2]
        sup_loss, _ = self.compute_supervised(z, y, training=True)
        total = self.add_supervised_to_total(total_wo_sup, sup_loss)
        logs["supervised_loss"] = sup_loss
        return total, logs
    

import tensorflow as tf
class UncertaintyMixin:
    def __init__(self, dist: str = "gauss", var_clip=(-10.0, 5.0), **kwargs):
        super().__init__(**kwargs)
        self._unc_dist = dist
        self._var_clip = var_clip  # clip range for log-variance

    def _split_mu_logvar(self, y_pred):
        # y_pred may be tuple/list (mu, log_var) or dict; adapt if needed
        mu, log_var = y_pred
        log_var = tf.clip_by_value(log_var, self._var_clip[0], self._var_clip[1])
        return mu, log_var

    def _nll_gauss(self, y_true, mu, log_var):
        inv_var = tf.exp(-log_var)
        nll = 0.5 * (tf.math.log(2.0 * tf.constant(3.1415926535)) + log_var + tf.square(y_true - mu) * inv_var)
        # sum over non-batch dims then mean over batch (compatible with SequenceMixin)
        axes = tf.range(1, tf.rank(nll))
        return tf.reduce_mean(tf.reduce_sum(nll, axis=axes))

    def _nll_laplace(self, y_true, mu, log_b):
        b = tf.exp(log_b)
        nll = tf.math.log(2.0*b) + tf.abs(y_true - mu)/b
        axes = tf.range(1, tf.rank(nll))
        return tf.reduce_mean(tf.reduce_sum(nll, axis=axes))

    # Override reconstruction loss to NLL
    def _get_reconstruction_loss(self, y_true, y_pred):
        mu, logv = self._split_mu_logvar(y_pred)
        if self._unc_dist == "gauss":
            return self._nll_gauss(y_true, mu, logv)
        elif self._unc_dist == "laplace":
            return self._nll_laplace(y_true, mu, logv)
        else:
            raise ValueError(f"Unknown dist: {self._unc_dist}")

    # MC Dropout prediction utility (N stochastic passes)
    def predict_mc(self, X, n_samples=20, batch_size=None):
        outs = []
        for _ in range(n_samples):
            # training=True keeps dropout active
            pred = self(X, training=True, batch_size=batch_size)  # assumes model(x) returns (mu, log_var)
            outs.append(pred)
        # stack (N, B, ...) and compute moments on mu and var
        mus = tf.stack([o[0] for o in outs], axis=0)
        logvars = tf.stack([o[1] for o in outs], axis=0)
        aleatoric = tf.reduce_mean(tf.exp(logvars), axis=0)
        epistemic = tf.math.reduce_variance(tf.reduce_mean(mus, axis=0, keepdims=False), axis=0) if False else tf.math.reduce_variance(mus, axis=0)
        # The variant above: epistemic as var over samples of μ; choose appropriate reduction if per-time-step.
        mu_mean = tf.reduce_mean(mus, axis=0)
        total = aleatoric + tf.math.reduce_variance(mus, axis=0)
        return mu_mean, aleatoric, epistemic, total