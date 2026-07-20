"""
Trend and seasonal decomposition layers and sub-networks.

Provides reusable level, trend, and seasonal reconstruction components,
together with structured decomposition sub-networks for time-series decoding.
"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras import backend as K

@tf.keras.utils.register_keras_serializable(package="UQModels_layers")
class LevelLayer(layers.Layer):
    """
    Constant level reconstruction component.

    Input
    -----
    Shape: (B, dim_z)

    Output
    ------
    Shape: (B, dim_seq, dim_out)
    """

    def __init__(
        self,
        dim_out: int,
        dim_seq: int,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.dim_out = int(dim_out)
        self.dim_seq = int(dim_seq)

        self.dense_hidden = layers.Dense(
            units=self.dim_out,
            activation="relu",
            name="level_params_hidden",
        )

        self.dense_output = layers.Dense(
            units=self.dim_out,
            activation=None,
            name="level_params",
        )

        self.reshape = layers.Reshape(
            target_shape=(1, self.dim_out),
            name="level_reshape",
        )

    def call(self, inputs):
        x = self.dense_hidden(inputs)
        x = self.dense_output(x)
        x = self.reshape(x)

        return tf.repeat(
            x,
            repeats=self.dim_seq,
            axis=1,
        )

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "dim_out": self.dim_out,
                "dim_seq": self.dim_seq,
            }
        )

        return config

@tf.keras.utils.register_keras_serializable(package="UQModels_layers")
class TrendLayer(layers.Layer):
    """
    Polynomial trend reconstruction block.

    Maps a latent representation to a feature-wise polynomial trend.

    Input
    -----
    Shape: (B, dim_z)

    Output
    ------
    Shape: (B, dim_seq, dim_out)
    """

    def __init__(
        self,
        dim_out: int,
        trend_poly: int,
        dim_seq: int,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.dim_out = int(dim_out)
        self.trend_poly = int(trend_poly)
        self.dim_seq = int(dim_seq)

        self._validate_config()

        self.trend_dense1 = layers.Dense(
            units=self.dim_out * self.trend_poly,
            activation="relu",
            name="trend_params_hidden",
        )

        self.trend_dense2 = layers.Dense(
            units=self.dim_out * self.trend_poly,
            activation=None,
            name="trend_params",
        )

        self.reshape_layer = layers.Reshape(
            target_shape=(
                self.dim_out,
                self.trend_poly,
            ),
            name="trend_reshape",
        )

        lin_space = (
            tf.range(
                self.dim_seq,
                dtype=tf.float32,
            )
            / float(self.dim_seq)
        )

        self.poly_space = tf.stack(
            [
                lin_space ** float(order + 1)
                for order in range(self.trend_poly)
            ],
            axis=0,
        )

    def _validate_config(self) -> None:
        """Validate trend block configuration."""

        if self.dim_out <= 0:
            raise ValueError(
                "dim_out must be strictly positive."
            )

        if self.dim_seq <= 0:
            raise ValueError(
                "dim_seq must be strictly positive."
            )

        if self.trend_poly <= 0:
            raise ValueError(
                "trend_poly must be strictly positive."
            )

    def call(self, inputs):
        """Generate polynomial trend values."""

        trend_params = self.trend_dense1(inputs)
        trend_params = self.trend_dense2(trend_params)
        trend_params = self.reshape_layer(trend_params)

        poly_space = tf.cast(
            self.poly_space,
            trend_params.dtype,
        )

        trend_values = tf.matmul(
            trend_params,
            poly_space,
        )

        return tf.transpose(
            trend_values,
            perm=[0, 2, 1],
        )

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "dim_out": self.dim_out,
                "trend_poly": self.trend_poly,
                "dim_seq": self.dim_seq,
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @staticmethod
    def make_config(
        dim_out: int,
        trend_poly: int,
        dim_seq: int,
        name: str | None = "trend_layer",
    ) -> dict:
        """Build a configuration dictionary for TrendLayer."""

        return {
            "dim_out": dim_out,
            "trend_poly": trend_poly,
            "dim_seq": dim_seq,
            "name": name,
        }
    
@tf.keras.utils.register_keras_serializable(package="UQModels_layers")
class SeasonalLayer(layers.Layer):
    """
    Seasonal reconstruction block.

    Generates one or more periodic feature-wise seasonal components
    from a latent representation.

    Input
    -----
    Shape: (B, dim_z)

    Output
    ------
    Shape: (B, dim_seq, dim_out)
    """

    def __init__(
        self,
        dim_out: int,
        dim_seq: int,
        custom_seas: list[tuple[int, int]],
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.dim_out = int(dim_out)
        self.dim_seq = int(dim_seq)

        self.custom_seas = [
            (int(num_seasons), int(len_per_season))
            for num_seasons, len_per_season in custom_seas
        ]

        self._validate_config()

        self.dense_layers = []
        self.reshape_layers = []

        for idx, (num_seasons, _) in enumerate(
            self.custom_seas
        ):
            self.dense_layers.append(
                layers.Dense(
                    units=self.dim_out * num_seasons,
                    name=f"season_params_{idx}",
                )
            )

            self.reshape_layers.append(
                layers.Reshape(
                    target_shape=(
                        self.dim_out,
                        num_seasons,
                    ),
                    name=f"season_reshape_{idx}",
                )
            )

    def _validate_config(self) -> None:
        """Validate seasonal block configuration."""

        if self.dim_out <= 0:
            raise ValueError(
                "dim_out must be strictly positive."
            )

        if self.dim_seq <= 0:
            raise ValueError(
                "dim_seq must be strictly positive."
            )

        if len(self.custom_seas) == 0:
            raise ValueError(
                "custom_seas must contain at least one seasonal component."
            )

        for num_seasons, len_per_season in self.custom_seas:

            if num_seasons <= 0:
                raise ValueError(
                    "num_seasons must be strictly positive."
                )

            if len_per_season <= 0:
                raise ValueError(
                    "len_per_season must be strictly positive."
                )

    def _get_season_indexes_over_seq(
        self,
        num_seasons: int,
        len_per_season: int,
    ):
        """Return seasonal indexes over the full sequence."""

        season_indexes = tf.repeat(
            tf.range(
                num_seasons,
                dtype=tf.int32,
            ),
            repeats=len_per_season,
        )

        repeats = (
            self.dim_seq
            // (num_seasons * len_per_season)
            + 1
        )

        season_indexes = tf.tile(
            season_indexes,
            [repeats],
        )

        return season_indexes[:self.dim_seq]

    def call(self, inputs):
        """Generate and aggregate seasonal components."""

        seasonal_values = []

        for idx, (
            num_seasons,
            len_per_season,
        ) in enumerate(self.custom_seas):

            params = self.dense_layers[idx](inputs)

            params = self.reshape_layers[idx](params)

            # params:
            # (B, dim_out, num_seasons)

            season_indexes = (
                self._get_season_indexes_over_seq(
                    num_seasons,
                    len_per_season,
                )
            )

            values = tf.gather(
                params,
                season_indexes,
                axis=2,
            )

            # values:
            # (B, dim_out, dim_seq)

            seasonal_values.append(values)

        values = tf.add_n(
            seasonal_values
        )

        return tf.transpose(
            values,
            perm=[0, 2, 1],
        )

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "dim_out": self.dim_out,
                "dim_seq": self.dim_seq,
                "custom_seas": self.custom_seas,
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @staticmethod
    def make_config(
        dim_out: int,
        dim_seq: int,
        custom_seas: list[tuple[int, int]],
        name: str | None = "seasonal_layer",
    ) -> dict:
        """Build a configuration dictionary for SeasonalLayer."""

        return {
            "dim_out": dim_out,
            "dim_seq": dim_seq,
            "custom_seas": list(custom_seas),
            "name": name,
        }
    
@tf.keras.utils.register_keras_serializable(package="UQModels_layers")
class TrendSeasonSubNet(layers.Layer):
    """
    Structured temporal reconstruction sub-network.

    Aggregates interpretable temporal components generated from
    a latent representation:

        latent input
        -> level component
        -> optional trend component
        -> optional seasonal component
        -> additive aggregation

    Input
    -----
    Shape: (B, dim_z)

    Output
    ------
    Shape: (B, dim_seq, dim_out)
    """

    def __init__(
        self,
        dim_seq: int,
        dim_out: int,
        dim_z: int,
        trend_poly: int = 0,
        custom_seas: list[tuple[int, int]] | None = None,
        use_level: bool = True,
        use_trend: bool = True,
        use_seasonal: bool = True,
        cfg_level: dict | None = None,
        cfg_trend: dict | None = None,
        cfg_seasonal: dict | None = None,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.dim_seq = int(dim_seq)
        self.dim_out = int(dim_out)
        self.dim_z = int(dim_z)

        self.trend_poly = int(trend_poly)

        self.custom_seas = (
            None
            if custom_seas is None
            else [
                (int(num_seasons), int(len_per_season))
                for num_seasons, len_per_season in custom_seas
            ]
        )

        self.use_level = bool(use_level)
        self.use_trend = bool(use_trend)
        self.use_seasonal = bool(use_seasonal)

        self.cfg_level = (
            {}
            if cfg_level is None
            else dict(cfg_level)
        )

        self.cfg_trend = (
            {}
            if cfg_trend is None
            else dict(cfg_trend)
        )

        self.cfg_seasonal = (
            {}
            if cfg_seasonal is None
            else dict(cfg_seasonal)
        )

        self._validate_config()
        self._prepare_component_configs()

        self.level_layer = None
        self.trend_layer = None
        self.seasonal_layer = None

        if self.use_level:
            self.level_layer = LevelLayer(
                **self.cfg_level
            )

        if self.use_trend and self.trend_poly > 0:
            self.trend_layer = TrendLayer(
                **self.cfg_trend
            )

        if (
            self.use_seasonal
            and self.custom_seas
        ):
            self.seasonal_layer = SeasonalLayer(
                **self.cfg_seasonal
            )

    def _validate_config(self) -> None:
        """Validate temporal sub-network configuration."""

        if self.dim_seq <= 0:
            raise ValueError(
                "dim_seq must be strictly positive."
            )

        if self.dim_out <= 0:
            raise ValueError(
                "dim_out must be strictly positive."
            )

        if self.dim_z <= 0:
            raise ValueError(
                "dim_z must be strictly positive."
            )

        if self.trend_poly < 0:
            raise ValueError(
                "trend_poly must be greater than or equal to 0."
            )

        if not any(
            (
                self.use_level,
                self.use_trend and self.trend_poly > 0,
                self.use_seasonal and bool(self.custom_seas),
            )
        ):
            raise ValueError(
                "TrendSeasonSubNet must activate at least one temporal component."
            )

    def _prepare_component_configs(self) -> None:
        """Complete component configurations with sub-network defaults."""

        if self.use_level:
            self.cfg_level.setdefault(
                "dim_out",
                self.dim_out,
            )

            self.cfg_level.setdefault(
                "dim_seq",
                self.dim_seq,
            )

            self.cfg_level.setdefault(
                "name",
                "level_component",
            )

        if self.use_trend and self.trend_poly > 0:
            self.cfg_trend.setdefault(
                "dim_out",
                self.dim_out,
            )

            self.cfg_trend.setdefault(
                "dim_seq",
                self.dim_seq,
            )

            self.cfg_trend.setdefault(
                "trend_poly",
                self.trend_poly,
            )

            self.cfg_trend.setdefault(
                "name",
                "trend_component",
            )

        if (
            self.use_seasonal
            and self.custom_seas
        ):
            self.cfg_seasonal.setdefault(
                "dim_out",
                self.dim_out,
            )

            self.cfg_seasonal.setdefault(
                "dim_seq",
                self.dim_seq,
            )

            self.cfg_seasonal.setdefault(
                "custom_seas",
                self.custom_seas,
            )

            self.cfg_seasonal.setdefault(
                "name",
                "seasonal_component",
            )

    def call(
        self,
        inputs,
        training=False,
    ):
        """Generate and aggregate temporal reconstruction components."""

        components = []

        if self.level_layer is not None:
            components.append(
                self.level_layer(inputs)
            )

        if self.trend_layer is not None:
            components.append(
                self.trend_layer(inputs)
            )

        if self.seasonal_layer is not None:
            components.append(
                self.seasonal_layer(inputs)
            )

        if len(components) == 1:
            return components[0]

        return tf.add_n(
            components
        )

    def get_config(self):
        """Return serializable configuration."""

        config = super().get_config()

        config.update(
            {
                "dim_seq": self.dim_seq,
                "dim_out": self.dim_out,
                "dim_z": self.dim_z,
                "trend_poly": self.trend_poly,
                "custom_seas": self.custom_seas,
                "use_level": self.use_level,
                "use_trend": self.use_trend,
                "use_seasonal": self.use_seasonal,
                "cfg_level": self.cfg_level,
                "cfg_trend": self.cfg_trend,
                "cfg_seasonal": self.cfg_seasonal,
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        """Rebuild layer from serialized configuration."""
        return cls(**config)

    @staticmethod
    def make_config(
        dim_seq: int,
        dim_out: int,
        dim_z: int,
        trend_poly: int = 0,
        custom_seas: list[tuple[int, int]] | None = None,
        use_level: bool = True,
        use_trend: bool = True,
        use_seasonal: bool = True,
        cfg_level: dict | None = None,
        cfg_trend: dict | None = None,
        cfg_seasonal: dict | None = None,
        name: str | None = "trendseason_subnet",
    ) -> dict:
        """Build a configuration dictionary for TrendSeasonSubNet."""

        return {
            "dim_seq": dim_seq,
            "dim_out": dim_out,
            "dim_z": dim_z,
            "trend_poly": trend_poly,
            "custom_seas": (
                None
                if custom_seas is None
                else list(custom_seas)
            ),
            "use_level": use_level,
            "use_trend": use_trend,
            "use_seasonal": use_seasonal,
            "cfg_level": (
                {}
                if cfg_level is None
                else dict(cfg_level)
            ),
            "cfg_trend": (
                {}
                if cfg_trend is None
                else dict(cfg_trend)
            ),
            "cfg_seasonal": (
                {}
                if cfg_seasonal is None
                else dict(cfg_seasonal)
            ),
            "name": name,
        }