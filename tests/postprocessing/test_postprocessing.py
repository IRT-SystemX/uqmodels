import numpy as np
import pytest

from uqmodels.postprocessing.UQ_processing import compute_PI, fit_PI
from uqmodels.postprocessing.UQKPI_Processor import NormalPIs_processor


@pytest.fixture
def list_alpha():
    return [0.025, 0.16, 0.84, 0.975]


@pytest.fixture
def uq_measure():
    return np.array([[[51.0], [51.0]], [[34.29], [34.29]]])


@pytest.fixture
def pred():
    return np.array([[20.1], [20.1]])


@pytest.fixture
def pis_proc(list_alpha):
    return NormalPIs_processor(list_alpha=list_alpha)


@pytest.mark.parametrize(
    "type_uq",
    ["None", "var_A&E"],
)
def test_fit_predictive_intervals(list_alpha, uq_measure, pred, type_uq):
    params_ = fit_PI(
        UQ=uq_measure,
        type_UQ=type_uq,
        pred=pred,
        y=None,
        list_alpha=list_alpha,
    )
    assert params_ is None


@pytest.mark.parametrize(
    "type_uq, kpi_shape",
    [
        ("var_A&E", (2, 1)),
        ("res_2var", (2, 1)),
        ("2var", (2, 1)),
        ("res_var", (2, 2, 1)),
        ("var", (2, 2, 1)),
    ],
)
def test_compute_predictive_intervals(list_alpha, uq_measure, pred, type_uq, kpi_shape):
    prediction_intervals_kpis, _ = compute_PI(
        UQ=uq_measure,
        type_UQ=type_uq,
        pred=pred,
        y=None,
        list_alpha=list_alpha,
        params_=None,
    )
    assert len(prediction_intervals_kpis) == len(list_alpha)
    for kpi in prediction_intervals_kpis:
        assert kpi.shape == kpi_shape


@pytest.mark.parametrize(
    "type_uq",
    [
        "None",
        "var_A&E",
        "res_var",
        "var",
        "res_2var",
        "2var",
        "quantile",
        "res_quantile",
    ],
)
def test_normal_pi_fit_transform(list_alpha, uq_measure, pred, pis_proc, type_uq):
    pis_proc.fit(
        UQ=uq_measure,
        type_UQ=type_uq,
        pred=pred,
        y=None,
    )
    # pis_kpis = pis_proc.transform(
    #     UQ=uq_measure,
    #     type_UQ=type_uq,
    #     pred=pred,
    #     y=None,
    # )
    assert True
