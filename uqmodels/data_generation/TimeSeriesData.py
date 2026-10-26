from dataclasses import dataclass, field
from typing import Any

import numpy as np

Array = np.ndarray

@dataclass
class TimeSeriesData:
    """
    Central representation of a univariate or multivariate time series.

    All time-dependent arrays use the first axis as the temporal axis.
    Main signal arrays must follow the shape `(n_steps, n_features)`.
    
    | Champ                  | Signification                                               |
    | ---------------------- | ----------------------------------------------------------- |
    | `time`                 | Index temporel ou timestamps                                |
    | `values`               | Valeurs actuellement exposées au pipeline                   |
    | `latent_values`        | État latent du processus génératif                          |
    | `nominal_values`       | Observations nominales avant perturbation                   |
    | `conditional_mean`     | Espérance conditionnelle théorique                          |
    | `conditional_variance` | Variance conditionnelle théorique                           |
    | `contexts`             | Informations externes ou génératives indexées dans le temps |
    | `components`           | Décomposition du processus génératif                        |
    | `masks`                | Localisation des événements, perturbations ou zones valides |
    | `descriptors`          | Caractérisations calculées a posteriori                     |
    | `metadata`             | Informations globales non temporelles                       |
    | `history`              | Traçabilité des opérations appliquées                       |

    """

    # Core temporal structure
    time: Array | None = None
    values: Array | None = None

    # Generative ground truth
    latent_values: Array | None = None
    nominal_values: Array | None = None
    conditional_mean: Array | None = None
    conditional_variance: Array | None = None

    # Structured additional information
    contexts: dict[str, Array] = field(default_factory=dict)
    components: dict[str, Array] = field(default_factory=dict)
    masks: dict[str, Array] = field(default_factory=dict)
    descriptors: dict[str, Array] = field(default_factory=dict)

    # Non-array information
    feature_names: tuple[str, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_steps(self) -> int | None:
        for array in (
            self.values,
            self.latent_values,
            self.nominal_values,
            self.time,
        ):
            if array is not None:
                return len(array)
        return None

    @property
    def n_features(self) -> int | None:
        for array in (
            self.values,
            self.latent_values,
            self.nominal_values,
        ):
            if array is not None:
                return 1 if array.ndim == 1 else array.shape[1]
        return None
        
    def validate(self) -> None:
        n_steps = self.n_steps

        if n_steps is None:
            return

        temporal_arrays = {
            "time": self.time,
            "values": self.values,
            "latent_values": self.latent_values,
            "nominal_values": self.nominal_values,
            "conditional_mean": self.conditional_mean,
            "conditional_variance": self.conditional_variance,
            **self.contexts,
            **self.components,
            **self.masks,
            **self.descriptors,
        }

        invalid = {
            name: array.shape
            for name, array in temporal_arrays.items()
            if array is not None and len(array) != n_steps
        }

        if invalid:
            raise ValueError(
                f"Inconsistent temporal dimensions: {invalid}."
            )