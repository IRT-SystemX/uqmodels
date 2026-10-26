"""
Time-series decomposition autoencoder models.

Provides deterministic and variational autoencoders combining structured
level, trend, seasonal, and convolutional residual reconstruction components.
"""

import tensorflow as tf
import math
import inspect
from tensorflow.keras import layers,Model
from tensorflow.keras.layers import Input
from uqmodels.modelization.TF_estimator.vae.base_vae import BaseAutoencoder,VariationalBlock,BaseVariationalAutoencoder,BaseVariationalAutoencoder
from uqmodels.modelization.TF_estimator.layers.convlayers import CNNSubNet
from uqmodels.modelization.TF_estimator.layers.trendseasonlayers import TrendSeasonSubNet
    
def build_timevae_encoder(
    dim_seq: int,
    dim_in: int,
    dim_z: int,
    variational: bool = True,
    cfg_subnet: dict | None = None,
    random_state: int | None = None,
    name: str = "encoder_time",
) -> tf.keras.Model:
    """
    Build a TimeVAE convolutional encoder.

    Architecture
    ------------
    Input (B, dim_seq, dim_in)
        -> CNNSubNet(mode="encoder")
        -> optional VariationalBlock

    Parameters
    ----------
    dim_seq:
        Input sequence length.

    dim_in:
        Number of input features per time step.

    dim_z:
        Latent dimension.

    variational:
        If True, applies VariationalBlock and returns
        ``(z_mean, z_log_var)``.
        Otherwise, returns a deterministic latent representation.

    cfg_subnet:
        Optional configuration dictionary passed to CNNSubNet.

    random_state:
        Random seed propagated to the sub-network.

    name:
        Keras model name.

    Returns
    -------
    tf.keras.Model
        Configured TimeVAE encoder.
    """

    encoder_input = layers.Input(
        shape=(dim_seq, dim_in),
        name="encoder_input",
    )

    if cfg_subnet is None:
        cfg_subnet = CNNSubNet.make_config(
            mode="encoder",
            dim_seq=dim_seq,
            dim_in=dim_in,
            dim_z=dim_z,
            num_channels=1,
            list_filters=(32, 64, 128),
            list_kernels=(3, 3, 3),
            list_strides=(2, 2, 2),
            block="1D",
            type_output=None,
            random_state=random_state,
            name="timevae_encoder_subnet",
        )

    else:
        cfg_subnet = dict(cfg_subnet)

        cfg_subnet.setdefault(
            "mode",
            "encoder",
        )

        cfg_subnet.setdefault(
            "dim_seq",
            dim_seq,
        )

        cfg_subnet.setdefault(
            "dim_in",
            dim_in,
        )

        cfg_subnet.setdefault(
            "dim_z",
            dim_z,
        )

        cfg_subnet.setdefault(
            "random_state",
            random_state,
        )

        cfg_subnet.setdefault(
            "name",
            "timevae_encoder_subnet",
        )

    encoder_subnet = CNNSubNet(
        **cfg_subnet
    )

    x = encoder_subnet(
        encoder_input
    )

    if variational:
        outputs = VariationalBlock(
            dim_z=dim_z,
            name="variational_block",
        )(x)

    else:
        outputs = x

    return tf.keras.Model(
        inputs=encoder_input,
        outputs=outputs,
        name=name,
    )

def build_timevae_decoder(
    dim_seq: int,
    dim_out: int,
    dim_z: int,
    cfg_time_subnet: dict | None = None,
    cfg_cnn_subnet: dict | None = None,
    use_residual_cnn: bool = True,
    random_state: int | None = None,
    name: str = "decoder_time",
) -> tf.keras.Model:

    decoder_input = layers.Input(
        shape=(dim_z,),
        name="decoder_input",
    )

    if cfg_time_subnet is None:
        cfg_time_subnet = TrendSeasonSubNet.make_config(
            dim_seq=dim_seq,
            dim_out=dim_out,
            dim_z=dim_z,
            name="time_subnet",
        )

    components = [
        TrendSeasonSubNet(
            **cfg_time_subnet
        )(decoder_input)
    ]

    if use_residual_cnn:
        if cfg_cnn_subnet is None:
            cfg_cnn_subnet = CNNSubNet.make_config(
                mode="decoder",
                dim_seq=dim_seq,
                dim_out=dim_out,
                dim_z=dim_z,
                block="1D",
                random_state=random_state,
                name="residual_cnn_subnet",
            )

        components.append(
            CNNSubNet(
                **cfg_cnn_subnet
            )(decoder_input)
        )

    outputs = (
        components[0]
        if len(components) == 1
        else layers.Add(
            name="timevae_reconstruction"
        )(components)
    )

    return tf.keras.Model(
        inputs=decoder_input,
        outputs=outputs,
        name=name,
    )

class TimeAE(BaseAutoencoder):
    """
    Deterministic TimeVAE-inspired autoencoder.

    The encoder and decoder are fully configured through dedicated
    builder configuration dictionaries.

    The encoder configuration must explicitly define:
        variational=False
    """

    def __init__(
        self,
        cfg_encoder: dict,
        cfg_decoder: dict,
        name: str = "time_ae",
        **kwargs,
    ):
        frame_locals = inspect.currentframe().f_locals

        explicit_args = {
            key: value
            for key, value in frame_locals.items()
            if key not in ("self", "kwargs", "frame_locals")
        }

        all_init_params = {
            **explicit_args,
            **kwargs,
        }

        super().__init__(**all_init_params)

        self.cfg_encoder = dict(cfg_encoder)
        self.cfg_decoder = dict(cfg_decoder)

        self._validate_encoder_config(
            self.cfg_encoder
        )

        self.encoder = build_timevae_encoder(
            **self.cfg_encoder
        )

        self.decoder = build_timevae_decoder(
            **self.cfg_decoder
        )
    
    @staticmethod
    def make_config(
        dim_seq: int,
        dim_in: int,
        dim_z: int,
        cfg_encoder_subnet: dict,
        cfg_time_subnet: dict,
        cfg_residual_cnn_subnet: dict | None = None,
        use_residual_cnn: bool = True,
        random_state: int | None = None,
        name: str = "time_ae",
        **kwargs,
    ) -> dict:
        """
        Build a TimeAE configuration.

        Notes
        -----
        Use:
        - CNNSubNet.make_config for cfg_encoder_subnet;
        - TrendSeasonSubNet.make_config for cfg_time_subnet;
        - CNNSubNet.make_config for cfg_residual_cnn_subnet.
        """
        return _make_time_autoencoder_config(
            dim_seq=dim_seq,
            dim_in=dim_in,
            dim_z=dim_z,
            cfg_encoder_subnet=cfg_encoder_subnet,
            cfg_time_subnet=cfg_time_subnet,
            cfg_residual_cnn_subnet=cfg_residual_cnn_subnet,
            variational=False,
            use_residual_cnn=use_residual_cnn,
            random_state=random_state,
            name=name,
            **kwargs,
        )


class TimeVAE(BaseVariationalAutoencoder):
    """
    Variational TimeVAE-inspired autoencoder.

    The encoder and decoder are fully configured through dedicated
    builder configuration dictionaries.

    The encoder configuration must explicitly define:
        variational=True
    """

    def __init__(
        self,
        cfg_encoder: dict,
        cfg_decoder: dict,
        kl_weight: float = 1.0,
        name: str = "time_vae",
        **kwargs,
    ):
        frame_locals = inspect.currentframe().f_locals

        explicit_args = {
            key: value
            for key, value in frame_locals.items()
            if key not in ("self", "kwargs", "frame_locals")
        }

        all_init_params = {
            **explicit_args,
            **kwargs,
        }

        super().__init__(**all_init_params)

        self.cfg_encoder = dict(cfg_encoder)
        self.cfg_decoder = dict(cfg_decoder)

        self._validate_encoder_config(
            self.cfg_encoder
        )

        self.encoder = build_timevae_encoder(
            **self.cfg_encoder
        )

        self.decoder = build_timevae_decoder(
            **self.cfg_decoder
        )
    
    @staticmethod
    def make_config(
        dim_seq: int,
        dim_in: int,
        dim_z: int,
        cfg_encoder_subnet: dict,
        cfg_time_subnet: dict,
        cfg_residual_cnn_subnet: dict | None = None,
        use_residual_cnn: bool = True,
        kl_weight: float = 1.0,
        random_state: int | None = None,
        name: str = "time_vae",
        **kwargs,
    ) -> dict:
        """
        Build a TimeVAE configuration.

        Notes
        -----
        Use:
        - CNNSubNet.make_config for cfg_encoder_subnet;
        - TrendSeasonSubNet.make_config for cfg_time_subnet;
        - CNNSubNet.make_config for cfg_residual_cnn_subnet.
        """
        return _make_time_autoencoder_config(
            dim_seq=dim_seq,
            dim_in=dim_in,
            dim_z=dim_z,
            cfg_encoder_subnet=cfg_encoder_subnet,
            cfg_time_subnet=cfg_time_subnet,
            cfg_residual_cnn_subnet=cfg_residual_cnn_subnet,
            variational=True,
            use_residual_cnn=use_residual_cnn,
            kl_weight=kl_weight,
            random_state=random_state,
            name=name,
            **kwargs,
        )

def _make_time_autoencoder_config(
    *,
    dim_seq: int,
    dim_in: int,
    dim_z: int,
    cfg_encoder_subnet: dict,
    cfg_time_subnet: dict,
    cfg_residual_cnn_subnet: dict | None,
    variational: bool,
    use_residual_cnn: bool = True,
    random_state: int | None = None,
    kl_weight: float | None = None,
    name: str,
    **kwargs,
) -> dict:
    """
    Assemble a TimeAE or TimeVAE configuration.

    Build sub-network configurations with:
    - CNNSubNet.make_config for the encoder;
    - TrendSeasonSubNet.make_config for structured reconstruction;
    - CNNSubNet.make_config for the optional residual decoder.
    """
    if use_residual_cnn and cfg_residual_cnn_subnet is None:
        raise ValueError(
            "cfg_residual_cnn_subnet is required when "
            "use_residual_cnn=True."
        )

    config = {
        "cfg_encoder": {
            "dim_seq": dim_seq,
            "dim_in": dim_in,
            "dim_z": dim_z,
            "variational": variational,
            "cfg_subnet": dict(cfg_encoder_subnet),
            "random_state": random_state,
            "name": "encoder_time",
        },
        "cfg_decoder": {
            "dim_seq": dim_seq,
            "dim_out": dim_in,
            "dim_z": dim_z,
            "cfg_time_subnet": dict(cfg_time_subnet),
            "cfg_cnn_subnet": (
                None
                if cfg_residual_cnn_subnet is None
                else dict(cfg_residual_cnn_subnet)
            ),
            "use_residual_cnn": use_residual_cnn,
            "random_state": random_state,
            "name": "decoder_time",
        },
        "name": name,
        **kwargs,
    }

    if kl_weight is not None:
        config["kl_weight"] = kl_weight

    return config