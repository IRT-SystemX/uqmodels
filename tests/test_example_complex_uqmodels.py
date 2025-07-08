import glob
import os
import pickle

import numpy as np
import pytest

import uqmodels.visualization.visualization as visu
from uqmodels.evaluation.evaluation import evaluate
from uqmodels.evaluation.metrics import Generic_metric, UQ_average_coverage, rmse
from uqmodels.preprocessing.Custom_Preprocessor import dict_to_TS_Dataset
from uqmodels.processing import Data_loader, Pipeline


# Loop over all files and delete them one by one
def delete_folder(path):
    new_path = os.path.join(path, "*")
    for file in glob.glob(new_path):
        os.remove(file)
    os.rmdir(path)


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
    global X, y, sample_weight, x_split, context, objective, name, train, test, dict_data, standards_evaluation_check_procedure

    def standards_evaluation_check_procedure(
        UQModels,
        pred,
        UQ,
        PIs,
        Elvl,
        KPI_anom,
        values_true,
        name_in_dict,
        val_sensibility=2.0,
    ):
        list_metrics = [
            Generic_metric(
                rmse,
                "Root mean square",
                mask=None,
                list_ctx_constraint=None,
                reduce=False,
                type_UQ="var_A&E",
            ),
            Generic_metric(
                UQ_average_coverage,
                "UQ_average_coverage",
                mask=None,
                list_ctx_constraint=None,
                reduce=False,
                type_UQ="var_A&E",
            ),
        ]

        output = pred, UQ
        metrics_val = evaluate(
            y, output, list_metrics, list_sets=[train, test], verbose=True
        )

        f_obs = np.arange(len(train))[-500:]
        list_percent = UQModels.list_predict_KPI_processors[2].KPI_parameters[
            "list_percent"
        ]
        visu.uncertainty_plot(
            y,
            output,
            context=None,
            size=(20, 8),
            f_obs=f_obs,
            name="UQplot",
            mode_res=False,
            dim=np.arange(y.shape[1]),
            type_UQ=UQModels.type_UQ,
            list_percent=list_percent,
            confidence_lvl=Elvl,
            show_plot=False,
        )
        visu.plot_anom_matrice(
            score=KPI_anom,
            true_label=dict_data["aux"]["anom"][:, None],
            f_obs=f_obs,
            show_plot=False,
        )

        list_alpha = UQModels.list_predict_KPI_processors[1].KPI_parameters[
            "list_alpha"
        ]
        empirique = [(pi > y).mean() for pi in PIs]
        for value, value_true in zip(list_alpha, values_true[name_in_dict]["target"]):
            assert abs((value - value_true) / value_true) < val_sensibility
        for value, value_true in zip(empirique, values_true[name_in_dict]["empirique"]):
            if value_true != 0.0:
                assert abs((value - value_true) / value_true) < val_sensibility
            else:
                assert abs(value - value_true) < val_sensibility

        list_percent = UQModels.list_predict_KPI_processors[2].KPI_parameters[
            "list_percent"
        ]
        empirique_elvl = [np.quantile(Elvl, q) for q in list_percent]
        for value, value_true in zip(
            list_percent, values_true[name_in_dict]["target_elvl"]
        ):
            assert abs((value - value_true) / value_true) < val_sensibility
        for value, value_true in zip(
            empirique_elvl, values_true[name_in_dict]["empirique_elvl"]
        ):
            if value_true != 0.0:
                assert abs((value - value_true) / value_true) < val_sensibility
            else:
                assert abs(value - value_true) < val_sensibility

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
def test_rf_uq(random_seed=0):
    from uqmodels.custom_UQModel import UQModel, UQModel_KPI
    from uqmodels.modelization.ML_estimator.random_forest_UQ import RF_UQEstimator
    from uqmodels.modelization.ML_estimator.random_forest_UQ import (
        get_params_dict as get_params_dict_rf,
    )

    values_true = {
        "RF_UQ": {
            "target": [0.025, 0.16, 0.84, 0.975],
            "empirique": [
                0.00675,
                0.07738888888888888,
                0.9259722222222222,
                0.9932777777777778,
            ],
            "target_elvl": [0.5, 0.8, 0.95, 0.98, 0.995, 1],
            "empirique_elvl": [0.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    }

    RF_UQ_paramaters = get_params_dict_rf()
    name = "RF_UQ"
    RF_UQModel = UQModel_KPI(
        RF_UQEstimator,
        RF_UQ_paramaters,
        name=name,
        beta=0.005,
        reduc_filter_KPI=None,
        reduc_filter_pred=None,
        random_state=random_seed,
    )

    # Test save load procedure
    RF_UQModel.save("tests/model")
    del RF_UQModel
    RF_UQModel = UQModel()
    RF_UQModel.load("tests/model/" + name)

    RF_UQModel.fit(X[train], y[train])
    (pred, UQ), KPI_anom, PIs, Elvl = RF_UQModel.score(X, y)
    standards_evaluation_check_procedure(
        RF_UQModel,
        pred,
        UQ,
        PIs,
        Elvl,
        KPI_anom,
        values_true,
        name_in_dict="RF_UQ",
        val_sensibility=2.0e-1,
    )
    del RF_UQModel
    delete_folder("tests/model/" + name)


@pytest.mark.dependency(depends=["test_data_load_and_preprocessing"])
def test_mlp_uq_mc_dropout(random_seed=0):
    from uqmodels.custom_UQModel import UQModel, UQModel_KPI
    from uqmodels.modelization.DL_estimator.neural_network_UQ import NN_UQ
    from uqmodels.modelization.DL_estimator.neural_network_UQ import (
        get_params_dict as get_params_dict_mlp_uq,
    )
    from uqmodels.modelization.DL_estimator.neural_network_UQ import (
        get_training_parameters,
    )
    from uqmodels.modelization.DL_estimator.neural_network_UQ import (
        get_UQEstimator_parameters as get_UQEstimator_parameters,
    )
    from uqmodels.modelization.DL_estimator.neural_network_UQ import mlp

    values_true = {
        "MC_Dropout": {
            "target": [0.025, 0.16, 0.84, 0.975],
            "empirique": [
                0.025694444444444443,
                0.13719444444444445,
                0.83625,
                0.9696944444444444,
            ],
            "target_elvl": [0.5, 0.8, 0.95, 0.98, 0.995, 1],
            "empirique_elvl": [0.0, 1.0, 2.0, 4.0, 4.0, 6.0],
        },
        "Deep_ensemble": {
            "target": [0.025, 0.16, 0.84, 0.975],
            "empirique": [
                0.03841666666666667,
                0.17866666666666667,
                0.84275,
                0.9677222222222223,
            ],
            "target_elvl": [0.5, 0.8, 0.95, 0.98, 0.995, 1],
            "empirique_elvl": [0.0, 1.0, 2.0, 3.0, 5.0, 6.0],
        },
        "EDL": {
            "target": [0.025, 0.16, 0.84, 0.975],
            "empirique": [0.012777777777777779, 0.098, 0.85875, 0.9786111111111111],
            "target_elvl": [0.5, 0.8, 0.95, 0.98, 0.995, 1],
            "empirique_elvl": [0.0, 1.0, 2.0, 3.0, 4.0, 6.0],
        },
    }

    for type_output in ["MC_Dropout", "Deep_ensemble", "EDL"]:
        name = "MLP_" + type_output
        mlp_params = get_params_dict_mlp_uq(
            X.shape[1], y.shape[1], type_output=type_output
        )
        training_params = get_training_parameters(
            epochs=[5], b_s=[128], l_r=[0.0005], type_output=type_output
        )

        UQEstimator_parameters = get_UQEstimator_parameters(
            model_parameters=mlp_params,
            training_parameters=training_params,
            type_output=type_output,
            rescale=True,
            model_initializer=mlp,
        )

        MLP_UQModel = UQModel_KPI(
            UQEstimator_initializer=NN_UQ,
            UQEstimator_params=UQEstimator_parameters,
            name=name,
            beta=0.005,
            reduc_filter_KPI=None,
            reduc_filter_pred=None,
            random_state=random_seed,
        )

        MLP_UQModel.fit(X[train], y[train], verbose=1)

        # Test save load procedure
        MLP_UQModel.save("tests/model")
        del MLP_UQModel
        MLP_UQModel = UQModel()
        MLP_UQModel.load("tests/model/" + name)
        (pred, UQ), KPI_anom, PIs, Elvl = MLP_UQModel.score(X, y)
        standards_evaluation_check_procedure(
            MLP_UQModel,
            pred,
            UQ,
            PIs,
            Elvl,
            KPI_anom,
            values_true,
            name_in_dict=type_output,
            val_sensibility=2.0,
        )
        del MLP_UQModel
        delete_folder("tests/model/" + name)


@pytest.mark.dependency(depends=["test_data_load_and_preprocessing"])
def test_lstm_ed_uq_as_uqestimator(random_seed=0):
    from uqmodels.custom_UQModel import UQModel, UQModel_KPI
    from uqmodels.modelization.DL_estimator.lstm_ed import Lstm_ED_UQ
    from uqmodels.modelization.DL_estimator.lstm_ed import (
        get_params_dict as get_params_dict_lstm,
    )
    from uqmodels.modelization.DL_estimator.lstm_ed import (
        get_training_parameters,
        get_UQEstimator_parameters,
    )

    values_true = {
        "LSTM_ED": {
            "target": [0.025, 0.16, 0.84, 0.975],
            "empirique": [
                0.012277777777777778,
                0.12983333333333333,
                0.9458888888888889,
                0.9935833333333334,
            ],
            "target_elvl": [0.5, 0.8, 0.95, 0.98, 0.995, 1],
            "empirique_elvl": [0.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    }

    name = "LSTM_ED_MC_Dropout"
    type_output = "MC_Dropout"
    model_params = get_params_dict_lstm(
        dim_ctx=X.shape[-1],
        dim_dyn=y.shape[-1],
        dim_target=y.shape[-1],
        size_window=20,
        n_windows=5,
        dim_horizon=5,
        dim_chan=1,
        type_output="MC_Dropout",
    )
    training_param = get_training_parameters(
        epochs=[5],
        b_s=[128],
        l_r=[0.0005],
        list_loss=["MSE", "BNN"],
        metrics=["MSE", "BNN"],
        param_loss=[2, 0.9],
    )
    UQEstimator_parameters = get_UQEstimator_parameters(
        model_parameters=model_params,
        training_parameters=training_param,
        type_output=type_output,
        factory_parameters={"factory_lag_st": 0, "factory_lag_lt": 0},
        rescale=False,
    )

    UQEstimator_initializer = Lstm_ED_UQ
    UQEstimator_parameters = UQEstimator_parameters

    DEEP_UQModel = UQModel_KPI(
        UQEstimator_initializer,
        UQEstimator_parameters,
        name=name,
        predictor=None,
        beta=0.001,
        reduc_filter_KPI=[1, 1, 1, 1, 1],
        reduc_filter_pred=[1, 0, 0, 0, 0],
        random_state=random_seed,
    )

    DEEP_UQModel._init_UQEstimator((X[train], y[train]), y[train])

    # Afactory function turn sequence in Window with mutli-horizon prediction
    # Manipulate sequence and apply factory on each batch : Less space memory / more time consuming
    if True:  # For test purpose run both mode
        DEEP_UQModel.UQEstimator.training_parameters["generator"] = True
        DEEP_UQModel.fit([X[train], y[train]], y[train], verbose=1)
        pred, (UQ, PIs, Elvl) = DEEP_UQModel.predict((X, y))
        (pred, UQ), KPI_anom, PIs, Elvl = DEEP_UQModel.score((X, y), y)

    # Apply factory on all sub-sequences, then learning on batch representation : lot of space memory / less time consuming
    if True:
        DEEP_UQModel.UQEstimator.training_parameters["generator"] = False
        DEEP_UQModel._init_UQEstimator((X[train], y[train]), y[train])
        DEEP_UQModel.fit([X[train], y[train]], y[train], verbose=1)
        if True:  # Test save load procedure
            DEEP_UQModel.save("tests/model")
            del DEEP_UQModel
            DEEP_UQModel = UQModel()
            DEEP_UQModel.load("tests/model/" + name)
        pred, (UQ, PIs, Elvl) = DEEP_UQModel.predict((X, y))
        (pred, UQ), KPI_anom, PIs, Elvl = DEEP_UQModel.score((X, y), y)
    # Inference procedure that provide prediction plus the specified UQKPIs
    standards_evaluation_check_procedure(
        DEEP_UQModel,
        pred,
        UQ,
        PIs,
        Elvl,
        KPI_anom,
        values_true,
        name_in_dict="LSTM_ED",
        val_sensibility=1.5,
    )
    del DEEP_UQModel
    delete_folder("tests/model/" + name)


@pytest.mark.dependency(depends=["test_data_load_and_preprocessing"])
def test_transformer_ed_uq_as_uqestimator(random_seed=0):
    from uqmodels.custom_UQModel import UQModel, UQModel_KPI
    from uqmodels.modelization.DL_estimator.transformer_ed import Transformer_ED_UQ
    from uqmodels.modelization.DL_estimator.transformer_ed import (
        get_params_dict as get_params_dict_transformer,
    )
    from uqmodels.modelization.DL_estimator.transformer_ed import (
        get_training_parameters,
        get_UQEstimator_parameters,
    )

    # to actualise
    values_true = {
        "Transformer_ED": {
            "target": [0.025, 0.16, 0.84, 0.975],
            "empirique": [
                0.012277777777777778,
                0.12983333333333333,
                0.9458888888888889,
                0.9935833333333334,
            ],
            "target_elvl": [0.5, 0.8, 0.95, 0.98, 0.995, 1],
            "empirique_elvl": [0.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    }

    type_output = "MC_Dropout"
    name = "Transformer_ED_MC_Dropout"
    model_params = get_params_dict_transformer(
        dim_ctx=X.shape[-1],
        dim_dyn=y.shape[-1],
        dim_target=y.shape[-1],
        size_window=40,
        n_windows=10,
        dim_horizon=5,
    )
    training_param = get_training_parameters(
        epochs=[5],
        b_s=[128],
        l_r=[0.0005],
        list_loss=["MSE", "BNN"],
        metrics=["MSE", "BNN"],
        param_loss=[2, 0.9],
    )
    UQEstimator_initializer = Transformer_ED_UQ
    UQEstimator_parameters = get_UQEstimator_parameters(
        model_parameters=model_params,
        training_parameters=training_param,
        type_output=type_output,
        factory_parameters={"factory_lag_st": 0, "factory_lag_lt": 0},
        rescale=False,
    )

    # We can also create a more complexe UQmodel that handle several UQKPI_Processor to build at inference (UQMesure, Predictive interval and model unreliability score) and after observation (Anomaly score)

    # Specification of the UQestimator
    UQEstimator_initializer = UQEstimator_initializer
    UQEstimator_parameters = UQEstimator_parameters

    # Instanciation of the UQmodel modeling pipeline
    DEEP_UQModel = UQModel_KPI(
        UQEstimator_initializer,
        UQEstimator_parameters,
        name=name,
        predictor=None,
        beta=0.001,
        reduc_filter_KPI=[1, 1, 1, 1, 1],
        reduc_filter_pred=[1, 0, 0, 0, 0],
        random_state=random_seed,
    )

    # Fit procedure

    # Apply factory at begin on all sub-sequences, then learning using folded representation : lot of space memory / less time consuming
    DEEP_UQModel._init_UQEstimator((X[train], y[train]), y[train])

    # Afactory function turn sequence in Window with mutli-horizon prediction
    # Manipulate sequence and apply factory on each batch : Less space memory / more time consuming
    if True:  # For test purpose run both mode
        DEEP_UQModel.UQEstimator.training_parameters["generator"] = True
        DEEP_UQModel.fit([X[train], y[train]], y[train], verbose=1)
        pred, (UQ, PIs, Elvl) = DEEP_UQModel.predict((X, y))
        (pred, UQ), KPI_anom, PIs, Elvl = DEEP_UQModel.score((X, y), y)

    # Apply factory on all sub-sequences, then learning on batch representation : lot of space memory / less time consuming
    if True:
        DEEP_UQModel.UQEstimator.training_parameters["generator"] = False
        DEEP_UQModel.fit([X[train], y[train]], y[train], verbose=1)
        if (
            False
        ):  # Test save load procedure : issues laading procedure run but doesn't load an operational model
            DEEP_UQModel.save("tests/model")
            DEEP_UQModel = UQModel()
            DEEP_UQModel.load("tests/model/" + name)
    pred, (UQ, PIs, Elvl) = DEEP_UQModel.predict((X, y))
    (pred, UQ), KPI_anom, PIs, Elvl = DEEP_UQModel.score((X, y), y)

    standards_evaluation_check_procedure(
        DEEP_UQModel,
        pred,
        UQ,
        PIs,
        Elvl,
        KPI_anom,
        values_true,
        name_in_dict="Transformer_ED",
        val_sensibility=1.5,
    )
    del DEEP_UQModel
    # delete_folder('tests/model/'+name)
