import numpy as np
from uqmodels.postprocessing.anomaly_processing import _resolve_covariance_mode


def test_resolve_covariance_mode():
    assert _resolve_covariance_mode(100, 5, mode="auto", auto_threshold=0.1) == "EmpiricalCovariance"
    assert _resolve_covariance_mode(10, 5, mode="auto", auto_threshold=0.1) == "LedoitWolf"
    assert _resolve_covariance_mode(100, 5, mode="LedoitWolf", auto_threshold=0.1) == "LedoitWolf"
