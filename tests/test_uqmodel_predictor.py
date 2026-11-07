import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor

from uqmodels.modelization.ML_estimator.random_forest_UQ import RF_UQEstimator
from uqmodels.UQModel import UQModel


@pytest.fixture
def sample_data():
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y_train = np.array([10, 20, 30, 40, 50])
    X_test = np.array([[11, 12], [13, 14]])
    return X_train, y_train, X_test


def test_uqmodel_with_predictor(sample_data):
    X_train, y_train, X_test = sample_data
    predictor = RandomForestRegressor(n_estimators=5, random_state=0)
    predictor.fit(X_train, y_train)
    predictor.is_fitted = True
    rf_params = {
        "estimator": RandomForestRegressor(n_estimators=5, random_state=0),
        "var_min": 0.002,
        "type_UQ": "var_A&E",
        "rescale": True,
    }
    model = UQModel(
        UQEstimator_initializer=RF_UQEstimator,
        UQEstimator_params=rf_params,
        name="UQModels",
        predictor=predictor,
        list_predict_KPI_processors=[],
        list_score_KPI_processors=[],
        cache_manager=None,
    )
    model.fit(X_train, y_train)
    pred, _ = model.predict(X_test)
    assert pred.shape[0] == X_test.shape[0]
