import os
import pickle

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

import uqmodels.postprocessing.UQ_processing as UQ_Process
import uqmodels.postprocessing.UQKPI_Processor as UQProc
from uqmodels.evaluation.metrics import (
    Generic_metric,
    UQ_absolute_residu_score,
    UQ_average_coverage,
    UQ_dEI,
    UQ_Gaussian_NLL,
    UQ_sharpness,
    rmse,
)
from uqmodels.modelization.ML_estimator.random_forest_UQ import RF_UQEstimator
from uqmodels.preprocessing.Custom_Preprocessor import dict_to_TS_Dataset
from uqmodels.processing import Data_loader, Pipeline
from uqmodels.UQModel import UQModel


@pytest.fixture
def data_loader():
    def load_api(storing, filename, **kwargs):
        with open(os.path.join(storing, filename), "rb") as f:
            return pickle.load(f)

    data_loader = Data_loader(data_loader_api=load_api)
    return data_loader


@pytest.fixture
def synthetic_dataset_multivariate_info():
    storing, filename = "examples/data", "synthetic_dataset_multivariate.p"
    return storing, filename


@pytest.mark.dependency()
def test_data_load_and_preprocessing(data_loader, synthetic_dataset_multivariate_info):
    global X, y, sample_weight, x_split, context, objective, name, train, test, dict_data
    storing, filename = synthetic_dataset_multivariate_info
    dict_data = data_loader.load({"storing": storing, "filename": filename})
    assert isinstance(dict_data, dict)
    assert list(dict_data.keys()) == [
        "X",
        "Y",
        "context",
        "X_name",
        "train",
        "test",
        "X_split",
        "context_name",
        "aux",
        "X_bis",
    ]

    preprocessor = dict_to_TS_Dataset()
    pipeline = Pipeline(data_loader=data_loader, list_processors=[preprocessor])
    list_query = [{"storing": storing, "filename": filename, "name": "Synthetic_data"}]
    dataset_generator = pipeline.transform(list_query)
    X, y, sample_weight, x_split, context, objective, name = next(dataset_generator)
    train = x_split == 1
    test = np.invert(train)
    assert X.shape == (12000, 19)
    assert y.shape == (12000, 3)
    assert name == "Synthetic_data"


@pytest.mark.dependency(depends=["test_data_load_and_preprocessing"])
def test_using_uqestimator_and_processing_fn_or_procesor_object(random_seed):
    global UQEstimator_initializer, UQEstimator_parameters, list_alpha, RF
    RF = RandomForestRegressor(
        min_samples_leaf=5,
        n_estimators=50,
        max_depth=20,
        ccp_alpha=0.0001,
        max_samples=0.7,
        random_state=random_seed,
    )
    UQEstimator_initializer = RF_UQEstimator
    UQEstimator_parameters = {
        "estimator": RF,
        "var_min": 0.002,
        "type_UQ": "var_A&E",
        "rescale": True,
    }
    UQEstimator = UQEstimator_initializer(**UQEstimator_parameters)
    UQEstimator.fit(X[train], y[train])
    pred, UQ = UQEstimator.predict(X)
    assert pred.shape == (12000, 3)
    assert UQEstimator.type_UQ == "var_A&E"
    assert UQ.shape == (2, 12000, 3)

    list_alpha = [0.025, 0.16, 0.84, 0.975]
    params_ = UQ_Process.fit_PI(
        UQ=UQ[:, train],
        type_UQ=UQEstimator.type_UQ,
        pred=pred[train],
        y=None,
        list_alpha=list_alpha,
    )
    PIs_KPIs, _ = UQ_Process.compute_PI(
        UQ=UQ,
        type_UQ=UQEstimator.type_UQ,
        pred=pred,
        y=None,
        list_alpha=list_alpha,
        params_=params_,
    )
    value1 = np.round(
        (np.array(PIs_KPIs)[:, test] > y[test]).mean(axis=(1, 2)) * 100, 1
    )
    value2 = np.array(list_alpha) * 100
    value1_true = [1.4, 11.1, 89.6, 98.7]
    value2_true = [2.5, 16.0, 84.0, 97.5]
    for value, value_true in zip(value1, value1_true):
        assert abs(value - value_true) / value_true < 1.0e-3
    for value, value_true in zip(value2, value2_true):
        assert abs(value - value_true) / value_true < 1.0e-3

    PIs_proc = UQProc.NormalPIs_processor(
        KPI_parameters={"list_alpha": [0.025, 0.16, 0.84, 0.975]}
    )
    params_ = PIs_proc.fit(
        UQ=UQ[:, train], type_UQ=UQEstimator.type_UQ, pred=pred[train], y=None
    )
    PIs_KPIs = PIs_proc.transform(UQ=UQ, type_UQ=UQEstimator.type_UQ, pred=pred, y=None)

    value1 = np.round(
        (np.array(PIs_KPIs)[:, test] > y[test]).mean(axis=(1, 2)) * 100, 1
    )
    value2 = np.array(list_alpha) * 100
    value1_true = [1.4, 11.1, 89.6, 98.7]
    value2_true = [2.5, 16.0, 84.0, 97.5]
    for value, value_true in zip(value1, value1_true):
        assert abs(value - value_true) / value_true < 1.0e-3
    for value, value_true in zip(value2, value2_true):
        assert abs(value - value_true) / value_true < 1.0e-3


@pytest.mark.dependency(
    depends=["test_using_uqestimator_and_processing_fn_or_procesor_object"]
)
def test_uqmodel_pipeline_for_predictive_interval():
    global RF_UQModel
    PIs_proc = UQProc.NormalPIs_processor(
        KPI_parameters={"list_alpha": [0.025, 0.16, 0.84, 0.975]}
    )
    RF_UQModel = UQModel(
        UQEstimator_initializer,
        UQEstimator_parameters,
        name="UQModels",
        predictor=None,
        list_predict_KPI_processors=[PIs_proc],
        list_score_KPI_processors=[],
        cache_manager=None,
    )
    RF_UQModel.fit(X[train], y[train])
    pred, PIs_KPIs = RF_UQModel.predict(X)

    value1 = np.round(
        (np.array(PIs_KPIs)[:, test] > y[test]).mean(axis=(1, 2)) * 100, 1
    )
    value2 = np.array(list_alpha) * 100
    value1_true = [1.4, 11.1, 89.6, 98.7]
    value2_true = [2.5, 16.0, 84.0, 97.5]
    for value, value_true in zip(value1, value1_true):
        assert abs(value - value_true) / value_true < 1.0e-3
    for value, value_true in zip(value2, value2_true):
        assert abs(value - value_true) / value_true < 1.0e-3


@pytest.mark.dependency(depends=["test_uqmodel_pipeline_for_predictive_interval"])
def test_using_uqmodel_pipeline_for_multikpi_at_inference_and_after_observation():
    global PIs, Elvl, pred, UQ
    UQEstimator_initializer = RF_UQEstimator
    UQEstimator_parameters = {
        "estimator": RF,
        "var_min": 0.002,
        "type_UQ": "var_A&E",
        "rescale": True,
    }

    UQ_proc = UQProc.UQKPI_Processor()
    PIs_proc = UQProc.NormalPIs_processor(
        KPI_parameters={"list_alpha": [0.025, 0.16, 0.84, 0.975]}
    )
    Elvl_proc = UQProc.Epistemicscorelvl_processor()
    Anom_proc = UQProc.Anomscore_processor(KPI_parameters={"beta": 0.01})
    RF_UQModel = UQModel(
        UQEstimator_initializer,
        UQEstimator_parameters,
        name="UQModels",
        predictor=None,
        list_predict_KPI_processors=[UQ_proc, PIs_proc, Elvl_proc],
        list_score_KPI_processors=[Anom_proc],
        cache_manager=None,
    )

    RF_UQModel.fit(X[train], y[train])
    pred, pred_tuple = RF_UQModel.predict(X)
    UQ, PIs, Elvl = pred_tuple
    KPI_anom = RF_UQModel.score(X, y)
    assert KPI_anom.shape == (12000, 3)
    assert abs(KPI_anom.mean() - 0.0018677646364309293) < 1.0e-2
    assert abs(KPI_anom.std() - 0.927062229244036) < 1.0e-2


@pytest.mark.dependency(
    depends=[
        "test_using_uqmodel_pipeline_for_multikpi_at_inference_and_after_observation"
    ]
)
def test_visualisation():
    global output
    output = pred, UQ

    value1 = (PIs[0] < y).mean(axis=0) * 100
    value2 = (PIs[1] < y).mean(axis=0) * 100
    value3 = [
        np.quantile(Elvl, q) for q in [0.50, 0.80, 0.90, 0.95, 0.975, 0.99, 0.999]
    ]
    value1_true = [99.14166667, 99.29166667, 99.175]
    value2_true = [90.66666667, 91.65, 91.225]
    value3_true = [0.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0]

    for value, value_true in zip(value1, value1_true):
        assert abs(value - value_true) / value_true < 1.0e-3
    for value, value_true in zip(value2, value2_true):
        assert abs(value - value_true) / value_true < 1.0e-3
    for value, value_true in zip(value3, value3_true):
        if value_true != 0.0:
            assert abs(value - value_true) / value_true < 1.0e-3
        else:
            assert abs(value - value_true) < 1.0e-3


@pytest.mark.dependency(depends=["test_vizualisation"])
def test_vizualisation_zoom():
    list_ctx_constraint = None
    list_metrics = [
        Generic_metric(
            rmse,
            "Root mean square",
            mask=None,
            list_ctx_constraint=list_ctx_constraint,
            reduce=True,
        ),
        Generic_metric(
            UQ_average_coverage,
            "Coverage",
            mask=None,
            list_ctx_constraint=list_ctx_constraint,
            reduce=True,
            type_UQ=RF_UQModel.type_UQ,
        ),
        Generic_metric(
            UQ_sharpness,
            "Sharpness",
            mask=None,
            list_ctx_constraint=list_ctx_constraint,
            reduce=True,
            type_UQ=RF_UQModel.type_UQ,
        ),
        Generic_metric(
            UQ_Gaussian_NLL,
            "NLL",
            mask=None,
            list_ctx_constraint=list_ctx_constraint,
            reduce=True,
            type_UQ=RF_UQModel.type_UQ,
        ),
        Generic_metric(
            UQ_dEI,
            "Epistemic_indicator",
            mask=None,
            list_ctx_constraint=list_ctx_constraint,
            reduce=True,
            type_UQ=RF_UQModel.type_UQ,
        ),
        Generic_metric(
            UQ_absolute_residu_score,
            "Anom_score",
            mask=None,
            list_ctx_constraint=list_ctx_constraint,
            reduce=True,
            type_UQ=RF_UQModel.type_UQ,
        ),
    ]

    list_values_true = [
        [
            0.37124350149689417,
            0.4480467860435305,
            1.8450807070535749,
            0.6940056644126921,
        ],
        [
            0.9904583333333333,
            0.9750833333333334,
            0.21666666666666667,
            0.923076923076923,
        ],
        [1.8598728206796806, 1.90356430573253, 1.9559838941646657, 2.3872022102146575],
        [
            -0.3903074483727687,
            -0.5242357014273294,
            -4.731548977303201,
            -0.9967148371561111,
        ],
        [
            -0.6541647733401145,
            -0.6216329676223857,
            -0.5912611664708028,
            -0.44176811900483176,
        ],
        [0.5149445117824841, 0.7803380323573322, 12.637093775274602, 1.387240972738234],
    ]

    for metrics, values_true in zip(list_metrics, list_values_true):
        mask_anom = dict_data["aux"]["anom"] > 0.5
        mask_anom_after_anom = (
            np.roll(dict_data["aux"]["anom"], 1) + np.roll(dict_data["aux"]["anom"], 2)
        ) > 0.5
        perf = metrics.compute(
            y,
            output,
            [train, test, test & mask_anom, test & mask_anom_after_anom],
            context=None,
        )

        values = np.round(perf, 3)

        for value, value_true in zip(values, values_true):
            assert abs(value - value_true) / value_true < 5.0e-2
