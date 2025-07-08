import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from uqmodels.modelization.ML_estimator.random_forest_UQ import RF_UQEstimator
from uqmodels.modelization.UQEstimator import MeanVarUQEstimator, UQEstimator


@pytest.fixture
def mean_var_estimator():
    return MeanVarUQEstimator(
        name="UQEstimator",
        type_UQ="None",
        rescale=False,
    )


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
def rf_estimator(random_forest_regressor):
    return RF_UQEstimator(
        estimator=random_forest_regressor,
        var_min=0.002,
        type_UQ="var_A&E",
        rescale=True,
    )


@pytest.fixture
def sample_data():
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([10, 20, 30])
    X_test = np.array([[7, 8], [9, 10]])
    return X_train, y_train, X_test


def test_init_base_estimator(mean_var_estimator):
    params = mean_var_estimator.get_params()
    assert mean_var_estimator.name == "UQEstimator"
    assert params["type_UQ"] == "None"
    assert not mean_var_estimator.rescale
    assert not mean_var_estimator.is_fitted


def test_format_estimator(mean_var_estimator, sample_data):
    X_train, y_train, _ = sample_data
    X_transformed, y_transformed = mean_var_estimator._format(
        X_train,
        y_train,
        type_transform="fit_transform",
    )
    assert X_train.shape == X_transformed.shape
    assert y_train.shape == y_transformed.shape


def test_abstract_base_estimator():
    with pytest.raises(TypeError):
        UQEstimator()


def test_fit_predict_rf_estimator(rf_estimator, sample_data):
    X_train, y_train, X_test = sample_data
    rf_estimator.fit(X_train, y_train)
    assert rf_estimator.is_fitted
    pred, uq_measure = rf_estimator.predict(X_test)
    # assert pred.shape == (2, 1)
    assert pred.shape == (2,)
    # assert uq_measure.shape == (2, 2, 1)
    assert uq_measure.shape == (2, 2)
