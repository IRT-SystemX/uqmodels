from sklearn.model_selection import RandomizedSearchCV
from abench.store.store import write,read
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import json

class Regressor:
    def __init__(self,estimator=LinearRegression()):
        self.estimator = estimator
        if(hasattr(estimator,'name')):
            self.name = estimator.name
        else:
            self.name = 'regressor'

    def _tuning(self, X, y, params=None,n_esti=100,folds=4,random_state=0):
        """Random_search with sequential k-split

        Args:
            X (array): Features
            y (array): Target
            params (str array, optional): parameter_grid. Defaults to None.
            n_esti (int, optional): Number of grid try . Defaults to 100.
            folds (int, optional): Number of sequential fold. Defaults to 4.
            verbose (int, optional): [description]. Defaults to 0.
        """

        if not(params is None):
            tscv = TimeSeriesSplit(n_splits=folds)
            random_search = RandomizedSearchCV(
                self.estimator,
                param_distributions=params,
                n_iter = n_esti,
                scoring="neg_mean_squared_error",
                n_jobs=8,
                cv=tscv.split(X),
                random_state=random_state,
                verbose=0)
            random_search.fit(X, y)
            self.estimator = random_search.best_estimator_
            
    def fit(self, X, y, **kwarg):
        self.estimator.fit(X, y, **kwarg)

    def predict(self, X, **kwarg):
        output = self.estimator.predict(X, **kwarg)
        return output

    def save(self,storing,keys):
        write(storing,keys,self)
        
    def load(self,storing,keys):
        regressor = read(storing,keys)
        self.estimator = regressor.estimator
        self.name = regressor.name
        return(self)
    
    def get_params(self):
        params = {'estimator':self.estimator}
        return params

def get_tunning_parameters(name):
    if(name=='LinearRegressor'):
        params = {"fit_intercept":[True,False]}
    elif(name=='RidgeRegressor'):
        params = {'alpha':[0.001,0.01,0.1,1,10], "fit_intercept" : [True,False]}
    elif(name=='GradientBoostingRegressor'):
        with open("model_parameters/grid_param_gbr.json") as json_file:
            params = json.load(json_file)
            # Processing to update entries with integers where needed
            for key, entry in params.items():
                if (type(entry) is float) and (params[key] == int(entry)):
                    params[key] = int(entry)
            # Model with best hp configuration
    elif(name=='RandomForestRegressor'):
        with open("model_parameters/grid_param_rf.json") as json_file:
            params = json.load(json_file)
            # Processing to update entries with integers where needed
            for key, entry in params.items():
                if (type(entry) is float) and (params[key] == int(entry)):
                    params[key] = int(entry)
            # Model with best hp configuration
    else:
        raise(ValueError(name,'not in [LinearRegressor,RidgeRegressor,GradientBoostingRegressor,RandomForestRegressor]'))