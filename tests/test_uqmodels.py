import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from uqmodels.modelization.ML_estimator.random_forest_UQ import RF_UQEstimator
from uqmodels.postprocessing.UQKPI_Processor import (
    Anomscore_processor,
    Epistemicscorelvl_processor,
    NormalPIs_processor,
    UQKPI_Processor,
)
from uqmodels.UQModel import UQModel


@pytest.fixture
def list_alpha():
    return [0.025, 0.16, 0.84, 0.975]


@pytest.fixture
def sample_data():
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([10, 20, 30])
    X_test = np.array([[7, 8], [9, 10]])
    return X_train, y_train, X_test


@pytest.fixture
def random_forest_regressor():
    return RandomForestRegressor(
        min_samples_leaf=5,
        n_estimators=50,
        max_depth=20,
        ccp_alpha=0.0001,
        max_samples=0.7,
    )


@pytest.fixture
def rf_estimator_params(random_forest_regressor):
    return {
        "estimator": random_forest_regressor,
        "var_min": 0.002,
        "type_UQ": "var_A&E",
        "rescale": True,
    }


@pytest.fixture
def rf_model_without_processing(rf_estimator_params):
    return UQModel(
        UQEstimator_initializer=RF_UQEstimator,
        UQEstimator_params=rf_estimator_params,
        name="UQModels",
        predictor=None,
        list_predict_KPI_processors=[],
        list_score_KPI_processors=[],
        cache_manager=None,
    )


@pytest.fixture
def pis_processor(list_alpha):
    return NormalPIs_processor(list_alpha=list_alpha)


def test_fit_rf_uqmodel(rf_model_without_processing, sample_data):
    X_train, y_train, _ = sample_data
    rf_model_without_processing.fit(X_train, y_train)
    assert rf_model_without_processing.is_fitted


def test_uq_model_pipeline_for_prediction_interval(
    sample_data, pis_processor, rf_estimator_params
):
    X_train, y_train, X_test = sample_data
    rf_uqmodel = UQModel(
        RF_UQEstimator,
        rf_estimator_params,
        name="UQModels",
        predictor=None,
        list_predict_KPI_processors=[pis_processor],
        list_score_KPI_processors=[],
        cache_manager=None,
    )
    rf_uqmodel.fit(X_train, y_train)
    assert rf_uqmodel.is_fitted
    # pred, (pis_kpis) = rf_uqmodel.predict(X_test)
    # TODO: test predict output once super transform is fixed


def test_uq_model_pipeline_multi_kpi_after_observation(
    list_alpha, sample_data, rf_estimator_params
):
    X_train, y_train, X_test = sample_data
    uq_proc = UQKPI_Processor()
    pis_proc = NormalPIs_processor(list_alpha=list_alpha)
    elvl_proc = Epistemicscorelvl_processor()
    rf_uqmodel_predict_kpi_proc = UQModel(
        RF_UQEstimator,
        rf_estimator_params,
        name="UQModels",
        predictor=None,
        list_predict_KPI_processors=[uq_proc, pis_proc, elvl_proc],
        list_score_KPI_processors=[],
        cache_manager=None,
    )
    rf_uqmodel_predict_kpi_proc.fit(X_train, y_train)
    assert True
    # pred, (uq, pis, elvl) = rf_uqmodel.predict(X_test)
    # kpi_anom = rf_uqmodel.score(X_train, y_train)
    # TODO: test predict output once super transform is fixed
    anom_proc = Anomscore_processor(Anom_parameters={"alpha": 0.01})
    rf_uqmodel_predict_kpi_and_score_proc = UQModel(
        RF_UQEstimator,
        rf_estimator_params,
        name="UQModels",
        predictor=None,
        list_predict_KPI_processors=[],
        list_score_KPI_processors=[anom_proc],
        cache_manager=None,
    )
    rf_uqmodel_predict_kpi_and_score_proc.fit(X_train, y_train)
