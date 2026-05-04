from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Tuple, Callable, Union,  List,Literal, Sequence
import tensorflow as tf


@dataclass
class TrajectoryResult:
    """
    Result container for a single diffusion trajectory run.

    Attributes
    ----------
    x_hat : tf.Tensor
        Reconstructed / sampled signal, shape (B,T,C).
    y_obs : Optional[tf.Tensor]
        Observed values used for conditioning (if any).
    mask : Optional[tf.Tensor]
        Observation mask used for conditioning (if any).
    collect : Optional[Dict[str, Any]]
        Optional collection payload (reserved for later steps).
    """
    x_hat: tf.Tensor
    y_obs: Optional[tf.Tensor] = None
    mask: Optional[tf.Tensor] = None
    collect: Optional[Dict[str, Any]] = None
    
    def resolve(self, path: str) -> tf.Tensor:
        """
        Resolve a tensor field from this TrajectoryResult using a dotted path.

        Supported paths:
            - "x_hat"
            - "y_obs"
            - "mask"
            - "collect.<key>"
        """
        if path == "x_hat":
            return self.x_hat
        if path == "y_obs":
            if self.y_obs is None:
                raise ValueError("y_obs is None.")
            return self.y_obs
        if path == "mask":
            if self.mask is None:
                raise ValueError("mask is None.")
            return self.mask
        if path.startswith("collect."):
            if self.collect is None:
                raise ValueError("collect is None.")
            key = path.split(".", 1)[1]
            if key not in self.collect or self.collect[key] is None:
                raise ValueError(f"Missing collect key: {key!r}")
            return self.collect[key]
        raise ValueError(f"Unsupported field path: {path!r}")

@dataclass
class CollectSpec:
    """
    Collection configuration for trajectory instrumentation.

    Attributes
    ----------
    enabled : bool
        Enable/disable collection.
    keys : Optional[Sequence[str]]
        Payload keys to keep. None -> keep all keys.
    reduce : {"last"}
        Reduction strategy over time steps. V0 supports "last" only.
    """
    enabled: bool = False
    keys: Optional[Sequence[str]] = None
    reduce: Literal["last"] = "last"

@dataclass
class SweepSpec:
    """
    Sweep specification for running multiple trajectories.

    Attributes
    ----------
    n : int
        Number of runs in the sweep.
    mode : {"none","offset"}
        - "none": single run only (n is ignored / forced to 1)
        - "offset": seeds are generated as base_seed + i
    """
    n: int = 1
    mode: Literal["none", "offset"] = "offset"

@dataclass
class RunConfig:
    """
    Minimal run configuration for orchestrating trajectories.

    Attributes
    ----------
    num_steps : int
        Number of reverse steps.
    projection : str
        Projection mode ("hard" or "soft" if supported).
    seed : Optional[int]
        Base seed (used when sweeping).
    seed_sweep : SweepSpec
        Controls multi-sampling under same condition.
    mask_sweep : SweepSpec
        Controls multi-masking / conditioning variants (V0: simply resample masks).
    collect_spec : Optional[CollectSpec]
        Optional instrumentation readout.
    """
    num_steps: int = 30
    projection: str = "hard"
    seed: Optional[int] = None
    seed_sweep: SweepSpec = field(default_factory=lambda: SweepSpec(n=1, mode="none"))
    mask_sweep: SweepSpec = field(default_factory=lambda: SweepSpec(n=1, mode="none"))
    collect_spec: Optional[CollectSpec] = None
    reducers: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class RunResult:
    """
    Output of run(cfg): raw trajectories + aggregated reductions.
    """
    results: List[TrajectoryResult]
    groups: List[List[TrajectoryResult]]
    reduced: Dict[str, Any]
    
RunConfig,RunResult,CollectSpec,SweepSpec