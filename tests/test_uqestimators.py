import numpy as np
import pytest
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from copy import deepcopy

from uqmodels.UQModel import UQModel
import uqmodels.modelization.UQEstimator as UQ_esti
import uqmodels.data_generation.Gen_two_dimension_uncertainty_data as gen
import uqmodels.modelization.ML_estimator.baseline as baseline
from uqmodels.modelization.ML_estimator.random_forest_UQ import RF_UQEstimator,get_params_dict
import uqmodels.evaluation.base_metrics as base_metrics
import uqmodels.visualization.visualization as visu
import uqmodels.postprocessing.UQKPI_Processor as UQProc

@pytest.fixture
def sample_data():
    dict_data = gen.generate_default()

    X = dict_data['X']
    y = dict_data['Y'][:,None]
    train = dict_data['train'] 
    test = dict_data['test']
    grid_sample = dict_data['aux']['grid_X']
    grid_target = dict_data['aux']['grid_Y']
    keep = dict_data['aux']['keep']
    pseudo_var = dict_data['aux']['pseudo_var']

    var_max = np.quantile(pseudo_var,0.98)
    var_min = np.quantile(pseudo_var,0.2)
    return X, y, train, test,pseudo_var

def test_variante_uq_estimators(sample_data):
    # Data generation :
    X,y,train,test,pseudo_var =sample_data
    y = y.reshape(-1,1)

    def apply_metrics(y,y_pred,y_pred_lower,y_pred_upper,train,test):
        if y_pred is not None:
            base_metrics.mean_squared_error(y[train],y_pred[train])
            base_metrics.mean_squared_error(y[test],y_pred[test])

        if y_pred_lower is not None:
            base_metrics.average_coverage(y[train],y_pred_lower[train],y_pred_upper[train])
            base_metrics.average_coverage(y[test],y_pred_lower[test],y_pred_upper[test])
            base_metrics.ace(y[train],y_pred_lower[train],y_pred_upper[train],alpha)
            base_metrics.ace(y[test],y_pred_lower[test],y_pred_upper[test],alpha)
            base_metrics.sharpness(y_pred_lower[train],y_pred_upper[train])
            base_metrics.sharpness(y_pred_lower[test],y_pred_upper[test])


    # base regressor
    model_gbr = GradientBoostingRegressor()
    model_rf = RandomForestRegressor()

    # Target miscoverage :
    alpha = 0.05



    alpha= 0.1
    # PI model based on Gaussian process regression
    list_UQEstimator_initialisers = []
    list_UQEstimator_parameters = []
    list_params_tunning=[]


    # PIs models based on ML regression UQ (with sigma and 2sigma gaussian hypothesis)
    for type_UQ in ['var']:
        list_UQEstimator_initialisers.append(UQ_esti.MeanVarUQEstimator)
        list_UQEstimator_parameters.append({'estimator':model_gbr,'estimator_var':model_rf,'type_UQ':type_UQ})
        
    # Quantile base predictor
    list_quantile_estimators = []
    list_quantile_estimators.append(GradientBoostingRegressor(random_state=0, loss='quantile', alpha=0.05))
    list_quantile_estimators.append(GradientBoostingRegressor(random_state=0, loss='quantile', alpha=0.5))
    list_quantile_estimators.append(GradientBoostingRegressor(random_state=0, loss='quantile', alpha=0.95))

    for type_UQ in ['quantile','res_quantile']:
        list_UQEstimator_initialisers.append(UQ_esti.QuantileUQEstimator)
        list_UQEstimator_parameters.append({'list_estimators':list_quantile_estimators,
                                    'list_alpha':[0.05,0.5,0.95],
                                    'type_UQ':type_UQ})


    # PI model based on Gaussian process regression
    list_UQEstimator_initialisers.append(baseline.GPR_UQEstimator)
    list_UQEstimator_parameters.append({'drop_ratio':0.95})  
        
    # PIs models based on ML regression UQ (with sigma and 2sigma gaussian hypothesis)
    for type_UQ in ['var','2var','res_var','res_2var']:
        estimator = deepcopy(model_gbr)
        print(estimator)
        list_UQEstimator_initialisers.append(baseline.REGML_UQEstimator)
        list_UQEstimator_parameters.append({'estimator':estimator,
                                            'estimator_var':deepcopy(estimator),
                                            'type_UQ':type_UQ})


    # PI model based on Grandient boosting quantile regression
    for type_UQ in ['quantile','res_quantile']:
        estimator=deepcopy(model_gbr)
        list_UQEstimator_initialisers.append(baseline.GBRQ_UQEstimator)
        list_UQEstimator_parameters.append({'list_alpha':[0.05,0.5,0.95],
                                            'type_UQ':type_UQ})


    # PIs models based on RF_UQ UQ (with sigma, 2sigma and empirique hypothesis)
    for type_UQ in ['var','2var','quantile','var_A&E','res_var','res_2var','res_quantile']:
        dict_params = get_params_dict(beta=0.1,var_min=0.0001,n_estimators=75,max_depth=12,type_UQ=type_UQ)
        list_UQEstimator_initialisers.append(RF_UQEstimator)
        list_UQEstimator_parameters.append(dict_params)

    #############################################################################
    # Loop for train and make inference on each config
    for intialiser,parameters in zip(list_UQEstimator_initialisers,list_UQEstimator_parameters):
        estimator = intialiser(**parameters)
        print(estimator.name,estimator.type_UQ)
        
        pred_model_train = 0
        pred_model = 0
        
        if(estimator.type_UQ is not False):
            if('res' in estimator.type_UQ):
                model_gbr.fit(X[train],y[train])
                pred_model = np.squeeze(model_gbr.predict(X))
                pred_model_train =  np.squeeze(pred_model[train])

        
        y_train = (np.squeeze(y[train])-pred_model_train).reshape(y[train].shape)
        estimator.fit(X[train], y_train)
        
        (pred,UQ) = estimator.predict(X)
        # Ensure good shape
        pred = (np.squeeze(pred) + pred_model).reshape(pred.shape)
        # Call to processing UQ:var or 2var to PIs
        
        PIs_processor = UQProc.NormalPIs_processor(KPI_parameters={"list_alpha": [0.05,0.95]})
        Anom_processor = UQProc.Anomscore_processor(KPI_parameters={"beta": 0.005,"d": 2})
        
        if(estimator.type_UQ in ['var','res_var']):
            UQ_train = UQ[train]
        elif(estimator.type_UQ in ['res_quantile','quantile','var_A&E','res_2var','2var']):
            UQ_train = UQ[:,train]
        
        # Execution of fit and transform procedure of PI_KPI_processor
        PIs_params_ = PIs_processor.fit(UQ=UQ_train,
                                        type_UQ=estimator.type_UQ,pred=pred[train],
                                        type_UQ_params=estimator.type_UQ_params,y=None)
        (y_pred_lower,y_pred_upper) = PIs_processor.transform(UQ=UQ,type_UQ=estimator.type_UQ,
                                                            pred=pred,type_UQ_params=estimator.type_UQ_params, 
                                                            y=None)
        
        # Execution of fit and predict of Anom_KPI_processor
        
        Anom_params_ = Anom_processor.fit(UQ=UQ_train,
                                        type_UQ=estimator.type_UQ,
                                        pred=pred[train],y=y[train],
                                        type_UQ_params=estimator.type_UQ_params)
        
        anom_score = Anom_processor.transform(UQ=UQ,
                                            type_UQ=estimator.type_UQ,
                                            pred=pred,y=y,
                                            type_UQ_params=estimator.type_UQ_params)
        
        apply_metrics(y,pred,y_pred_lower,y_pred_upper,train,test)

        f_obs= np.arange(len(y))[0:500]
        visu.plot_pi(
            y,
            pred,
            y_pred_lower,
            y_pred_upper,
            mode_res=False,
            f_obs = f_obs,
            name = 'Prediction with uncertainty',
            size = (15, 6))

        visu.plot_pi(
            y,
            pred,
            y_pred_lower,
            y_pred_upper,
            mode_res=True,
            f_obs = f_obs,
            name = 'Residuals with uncertainty',
            size = (15, 6))
        
        visu.plot_anom_matrice(score=anom_score,true_label=None,f_obs=f_obs,vmin=-4,vmax=4)
        del(estimator)
        del(anom_score)

    # set of KPI parameters to test
    dict_NormalPIs_processor_0 = {'list_alpha':[0.025,0.16,0.84,0.975],'with_epistemic':True}
    dict_NormalPIs_processor_1 = {'list_alpha':[0.025,0.975],'with_epistemic':False}

    Epistemicscorelvl_0 = {'mode':"score",'list_percent':[0.50, 0.80, 0.95, 0.98, 0.995, 1]}
    Epistemicscorelvl_1 = {'mode':"levels",'list_percent':[0.10, 0.50, 0.90]}

    dict_Anom1_processor_0 = {'beta':0.1,'mode':'score','fusion':False,'type_fusion':'mahalanobis','type_norm':'Nsigma_local','q_var':2}
    dict_Anom2_processor_0 = {'beta':0.1,'mode':'born','fusion':True,'type_fusion':'mean','type_norm':'Nsigma_local','q_var':2}

    dict_Anom1_processor_1 = {'beta':0.05,'mode':'score','fusion':False,'type_fusion':'mahalanobis','type_norm':'chebyshev_local','q_var':0.1}
    dict_Anom2_processor_1 = {'beta':0.05,'mode':'born','fusion':True,'type_fusion':'mean','type_norm':'chebyshev_global','q_var':0.1}

    dict_Anom1_processor_2 = {'beta':0.2,'mode':'score','fusion':False,'type_fusion':'mahalanobis','type_norm':'Cantelli_local','q_var':0.1}
    dict_Anom2_processor_2 = {'beta':0.2,'mode':'born','fusion':True,'type_fusion':'mean','type_norm':'Cantelli_global','q_var':0.1}

    dict_Anom1_processor_3 = {'beta':0.005,'mode':'score','fusion':False,'type_fusion':'mahalanobis','type_norm':'quantiles_local','q_var':0.1}
    dict_Anom2_processor_3 = {'beta':0.005,'mode':'born','fusion':True,'type_fusion':'mean','type_norm':'quantiles_global','q_var':0.1,'q_Eratio':2}

    dict_Anom1_processor_4 = {'beta':0.0005,'mode':'score','fusion':False,'type_fusion':'mahalanobis','type_norm':'Chi2','q_var':0.1,'q_var_e':0}
    dict_Anom2_processor_4 = {'beta':0.0005,'mode':'born','fusion':True,'type_fusion':'mean','type_norm':'Chi2','q_var':0.1,'k_var_e':0}

    list_KPI_parameters = [[dict_NormalPIs_processor_0,Epistemicscorelvl_0,dict_Anom1_processor_0,dict_Anom2_processor_0],
                        [dict_NormalPIs_processor_1,Epistemicscorelvl_1,dict_Anom1_processor_1,dict_Anom2_processor_1],
                        [dict_NormalPIs_processor_0,Epistemicscorelvl_1,dict_Anom1_processor_2,dict_Anom2_processor_2],
                        [dict_NormalPIs_processor_1,Epistemicscorelvl_0,dict_Anom1_processor_3,dict_Anom2_processor_3],
                        [dict_NormalPIs_processor_0,Epistemicscorelvl_0,dict_Anom1_processor_4,dict_Anom2_processor_4]]

    # We can also create a more complexe UQmodel that handle several UQKPI_Processor to build at inference (UQMesure, Predictive interval and model unreliability score) and after observation (Anomaly score)
    # Specification of the UQestimator


    for n,(KPI_params_Normal,KPI_params_Epistemic,KPI_params_Anom1,KPI_params_Anom2) in enumerate(list_KPI_parameters):
        UQEstimator_initializer = RF_UQEstimator
        RF =  RandomForestRegressor(min_samples_leaf=5,n_estimators=50,max_depth=10,ccp_alpha=0.0005,max_samples=0.7)
        UQEstimator_parameters = {'estimator':RF,'var_min':0.002,'type_UQ':'var_A&E','rescale':True,'random_state':0}

        # Instanciation of PostProcesseur that provide UQ measure.
        UQ_proc = UQProc.UQKPI_Processor(KPI_parameters={'pred_and_UQ':n==0})

        # PostProcesseur that compute Predictive intervals
        PIs_proc = UQProc.NormalPIs_processor(KPI_parameters=KPI_params_Normal)
        # PostProcesseur  that compute an epistemics lvl score
        Elvl_proc = UQProc.Epistemicscorelvl_processor(KPI_parameters=KPI_params_Epistemic)
        # PostProcesseur Instanciation that compute an epistemics lvl score
        Anom_proc1 = UQProc.Anomscore_processor(KPI_parameters=KPI_params_Anom1)
        Anom_proc2 = UQProc.Anomscore_processor(KPI_parameters=KPI_params_Anom2)
        # Instanciation of the UQmodel modeling pipeline
        RF_UQModel = UQModel(UQEstimator_initializer,
                            UQEstimator_parameters,
                            name='UQModels',
                            predictor=None,
                            list_predict_KPI_processors=[UQ_proc,PIs_proc,Elvl_proc],
                            list_score_KPI_processors=[Anom_proc1,Anom_proc2],
                            cache_manager=None)

        # Fit procedure
        RF_UQModel.fit(X[train],y[train])
        # Inference procedure that provide prediction plus the specified UQKPIs
        pred,(UQ,PIs,Elvl) = RF_UQModel.predict(X)
        # Score procedure that provide ANom-KPI
        KPI_anom1,KPI_anom2 = RF_UQModel.score(X,y)
        print('Done run ',n)