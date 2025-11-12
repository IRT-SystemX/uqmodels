# Tunning regressor parameter
import sys
sys.path.insert(0, "../../")
import numpy as np
import pickle
import os
import warnings
import sys
import abench
import numpy as np
import pandas as pd
from functools import partial

Path_to_store = 'result'
Path_data = 'data/Dataset_Synthetic'

# Specification of data : 
from src.data_loader import MyABTimeSeriesFromDict,MyABCvDataExperiment
path = Path_data

#Tunning dataset
tunning_set = MyABTimeSeriesFromDict(path=path,selected_split_indexes=[0])

# Data experimental plan : list of trainset associated to list list of testsets generate according a Times series splits configuration 
ABcv_experiment_TS = MyABCvDataExperiment(path=path)

# Specification of Regressor modules (subpart of component)
from src.regressor import Regressor,get_tunning_parameters # Wrapper and get_tunning parameters
from src.regressor import LinearRegression,Ridge,RandomForestRegressor,GradientBoostingRegressor # Candidates
dict_Regressor={ 'LinearRegressor':{'module':Regressor,'parameters':{'estimator':LinearRegression()},
                                    'grid_params':get_tunning_parameters('LinearRegressor')},
                 'RidgeRegressor':{'module':Regressor(estimator=Ridge()),'parameters':None,
                                   'grid_params':get_tunning_parameters('RidgeRegressor')},
                 'RandomForestRegressor':{'module':Regressor(estimator=RandomForestRegressor()),'parameters':None,
                                    'grid_params':get_tunning_parameters('RandomForestRegressor')},
                 'GradientBoostingRegressor':{'module':Regressor(estimator=GradientBoostingRegressor()),'parameters':None,
                                              'grid_params':get_tunning_parameters('GradientBoostingRegressor')}}

# Specification of Preprocessor modules (subpart of component)
from src.preprocessor import Preprocessor # Wrapper and get_tunning parameters
dict_Preprocessor={'none':{'module':Preprocessor('none'),'parameters':None,},
                   'standard':{'module':Preprocessor('standard'),'parameters':None,}}

# Specification of the Component candidate list :
exp_design=[]
subexp_design=[{'name':'LinearRegressor','regressor':'LinearRegressor','preprocessor':'none'}]
exp_design.append(subexp_design)
subexp_design=[{'name':'RidgeRegressor','regressor':'RidgeRegressor','preprocessor':'standard'}]
exp_design.append(subexp_design)
subexp_design=[{'name':'RandomForestRegressor','regressor':'RandomForestRegressor','preprocessor':'none'}]
exp_design.append(subexp_design)
subexp_design=[{'name':'GradientBoostingRegressor','regressor':'GradientBoostingRegressor','preprocessor':'standard'}]
exp_design.append(subexp_design)

from src.component import ComponentRegressor
dict_exp={'Component': ComponentRegressor,
          'tuning_scheme' : {'regressor':{'set':tunning_set,'kwargs':{'n_esti':10,'folds':5}}},
          'regressor': dict_Regressor,
          'preprocessor': dict_Preprocessor,
          'exp_design':exp_design}

# Metric Specification 
from src.metric import MetricsMae,MetricsMse
list_metrics=[MetricsMae(),MetricsMse()]     

from abench.benchmark.benchmark import benchmark
storing = Path_to_store
storing = benchmark(storing=storing,
                    ABDataExperiment=ABcv_experiment_TS,
                    dict_exp=dict_exp,
                    list_metrics=list_metrics,
                    verbose=True)