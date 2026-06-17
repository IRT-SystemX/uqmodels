"""
Minimal Sequence Variational Autoencoder built by composing:
    SequenceMixin + VariationalMixin + BaseAutoencoder

Encoder (B,T,F) -> (z_mean,z_log_var,z) of shape (B,T,D)
Decoder (B,T,D) -> (B,T,F)
Training is handled by BaseAutoencoder via the forward_and_losses hook.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model as KModel
from uqmodels.modelization.TF_estimator.vae.base_vae import BaseAutoencoder,BaseVariationalAutoencoder,HybridMixin


class LSTMEncoder(KModel):
    def __init__(self, dim_seq, dim_in, dim_h, dim_z, name="encoder",variational=True,**kwargs):
        super().__init__(name=name, **kwargs)
        self.dim_seq = dim_seq
        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_z = dim_z

        self.enc_lstm = layers.LSTM(self.dim_h, return_sequences=True, name="enc_lstm")
        self.z_mean_td = layers.TimeDistributed(layers.Dense(self.dim_z), name="z_mean")
        if(self.variational):
            self.z_log_var_td = layers.TimeDistributed(layers.Dense(self.dim_z), name="z_log_var")

        # Optionnel : définir un input_spec pour une validation d'entrée plus stricte
        self.input_spec = layers.InputSpec(shape=(None, self.dim_seq, self.dim_in))

    def call(self, inputs, training=None):
        x = self.enc_lstm(inputs, training=training)
        output = self.z_mean_td(x)
        if(self.variational):
            output = [output]        
            z_log_var = self.z_log_var_td(x)
            output.append(z_log_var)
        return output

    def build(self, input_shape):
        # Force l’init des variables en appelant les couches sur un tenseur fictif
        _ = self.enc_lstm.build((input_shape[0], self.dim_seq, self.dim_in))
        _ = self.z_mean_td.build((input_shape[0], self.dim_seq, self.dim_h))
        if(self.variational):
            _ = self.z_log_var_td.build((input_shape[0], self.dim_seq, self.dim_h))
        super().build(input_shape)

    def get_config(self):
        base = super().get_config()
        base.update({
            "dim_seq": self.dim_seq,
            "dim_in": self.dim_in,
            "dim_h": self.dim_h,
            "dim_z": self.dim_z,
            "name": self.name,
        })
        return base


# --------------------------
# Decoder: (B, T, D) -> (B, T, F)
# --------------------------
class LSTMDecoder(KModel):
    def __init__(self, dim_seq, dim_in, dim_h, dim_z, name="decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dim_seq = dim_seq
        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_z = dim_z

        self.dec_lstm = layers.LSTM(self.dim_h, return_sequences=True, name="dec_lstm")
        self.recon_td = layers.TimeDistributed(layers.Dense(self.dim_in), name="recon_out")
        self.input_spec = layers.InputSpec(shape=(None, self.dim_seq, self.dim_z))

    def call(self, inputs, training=None):
        y = self.dec_lstm(inputs, training=training)
        out = self.recon_td(y)
        return out

    def build(self, input_shape):
        _ = self.dec_lstm.build((input_shape[0], self.dim_seq, self.dim_z))
        _ = self.recon_td.build((input_shape[0], self.dim_seq, self.dim_h))
        super().build(input_shape)

    def get_config(self):
        base = super().get_config()
        base.update({
            "dim_seq": self.dim_seq,
            "dim_in": self.dim_in,
            "dim_h": self.dim_h,
            "dim_z": self.dim_z,
            "name": self.name,
        })
        return base

class SeqVariationalVAE(BaseAutoencoder):
    def __init__(
        self,
        dim_seq: int,
        dim_in: int,
        dim_z: int,
        dim_h: int = 128,
        name: str = "seq_vae",
        **kwargs
    ):
        # Multiple inheritance init order: call BaseAutoencoder and VariationalMixin
        super().__init__(self, name=name, **kwargs)
        self.dim_seq = int(dim_seq)
        self.dim_in = int(dim_in)
        self.dim_z = int(dim_z)
        self.dim_h = int(dim_h)

        self.encoder = LSTMEncoder(dim_seq, dim_in, dim_h, dim_z,variational=False)
        self.decoder = LSTMDecoder(dim_seq, dim_in, dim_h, dim_z)
        # KL tracker (optional)
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

class SeqVariationalVAE(BaseVariationalAutoencoder):
    def __init__(
        self,
        dim_seq: int,
        dim_in: int,
        dim_z: int,
        dim_h: int = 128,
        kl_weight: float = 1.0,
        name: str = "seq_vae",
        **kwargs):
        # Multiple inheritance init order: call BaseAutoencoder and VariationalMixin
        super().__init__(self, name=name,kl_weight=kl_weight, **kwargs)
        self.dim_seq = int(dim_seq)
        self.dim_in = int(dim_in)
        self.dim_z = int(dim_z)
        self.dim_h = int(dim_h)

        self.encoder = LSTMEncoder(dim_seq, dim_in, dim_h, dim_z,variational=True)
        self.decoder = LSTMDecoder(dim_seq, dim_in, dim_h, dim_z)
        # KL tracker (optional)
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    
class HybridSequenceVAE(HybridMixin, BaseVariationalAutoencoder):
    """
    VAE séquentiel hybride (reconstruction + KL + supervision).

    Encoder:  (B, T, F) -> (z_mean, z_log_var, z) de shape (B, T, D)
    Decoder:  (B, T, D) -> (B, T, F)
    Supervision: tête branchée sur z (B, T, D) ou sur un pooling de z selon HybridMixin.

    Paramètres principaux:
      - dim_seq, dim_in, dim_z, dim_h
      - n_outputs: taille de la cible (classes ou régression)
      - sup_weight: poids de la tête supervisée
      - supervised_loss: loss de supervision (défaut: CCE from_logits=False)
      - from_logits: si True, la tête n’a pas d’activation (à gérer dans la loss)
      - target_sequence: True si la cible est séquentielle (B, T, n_outputs)
      - pooling: "mean" | "last" | "flatten" (si cible globale et z est séquentiel)
      - kl_weight: poids du terme KL
    """

    def __init__(
        self,
        dim_seq: int,
        dim_in: int,
        dim_z: int,
        dim_h: int = 128,
        *,
        n_outputs: int,
        sup_weight: float = 1.0,
        supervised_loss=None,
        from_logits: bool = False,
        target_sequence: bool = False,
        pooling: str = "mean",
        kl_weight: float = 1.0,
        name: str = "hybrid_seq_vae",
        **kwargs
    ):
        # Respecter l’ordre d’init pour héritage multiple
        super.__init__(n_outputs=n_outputs,
                       sup_weight=sup_weight,
                       supervised_loss=supervised_loss,
                       from_logits=from_logits,
                       target_sequence=target_sequence,
                       pooling=pooling,
                       kl_weight=kl_weight,
                       name=name,
                       **kwargs)

        self.dim_seq = int(dim_seq)
        self.dim_in = int(dim_in)
        self.dim_z = int(dim_z)
        self.dim_h = int(dim_h)



    @property
    def metrics(self):
        # Base + KL + (HybridMixin ajoute supervised_loss (+ accuracy éventuelle))
        return super().metrics + [self.kl_loss_tracker]

    # -------------- Loop (hook central) -------------- #
    def forward_and_losses(self, data):
        """
        Attend data = (X, y) :
          - X: (B, T, F)
          - y: (B, n_outputs) si cible globale
               ou (B, T, n_outputs) si target_sequence=True
        """
        X, y = data

        z_mean, z_log_var, z = self.encoder(X, training=True)
        X_recon = self.decoder(z, training=True)

        # Reconstruction séquentielle (SequenceMixin)
        recon_loss = self._get_reconstruction_loss(X, X_recon)

        # KL (VariationalMixin)
        total_wo_sup, kl_loss = self.compute_total_with_kl(recon_loss, z_mean, z_log_var)

        # Supervision (HybridMixin)
        sup_loss, y_pred = self.compute_supervised(z, y, training=True)

        total = self.add_supervised_to_total(total_wo_sup, sup_loss)

        logs = {
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
            "supervised_loss": sup_loss,
        }
        return total, logs