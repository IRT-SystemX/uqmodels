import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from uqmodels.UQModel import UQModel
from uqmodels.modelization.ML_estimator.random_forest_UQ import RF_UQEstimator


def test_e2e_uqmodel_rf():
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=float)
    y_train = np.array([10, 20, 30, 40, 50], dtype=float)
    X_test = np.array([[11, 12], [13, 14]], dtype=float)

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
        predictor=None,
        list_predict_KPI_processors=[],
        list_score_KPI_processors=[],
        cache_manager=None,
    )
    model.fit(X_train, y_train)
    pred, _ = model.predict(X_test)
    assert pred.shape[0] == X_test.shape[0]
    assert np.isclose(np.mean(pred), 48.0, atol=1e-6)
