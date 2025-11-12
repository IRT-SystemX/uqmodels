import abench
from abench.component.component import Component

import numpy as np
import pandas as pd

from src.preprocessor import Preprocessor
from src.regressor import Regressor

from abench.component.component import Component
class ComponentRegressor(Component):
    def __init__(self,regressor={'initializer':Regressor(),'parameters':None},preprocessor={'initializer':Preprocessor(),'parameters':None},**kwargs):
        if(regressor['parameters'] is None): # If subpart is store as model
            self.regressor = regressor["initializer"]
        else:  # If subpart is store as (initializer,parameters)
            self.regressor = regressor["initializer"](**regressor['parameters'])
        if(preprocessor['parameters'] is None): # If subpart is store as model
            self.preprocessor = preprocessor["initializer"]
        else:  # If subpart is store as (initializer,parameters)
            self.preprocessor = preprocessor["initializer"](**preprocessor['parameters'])
    
    def fit(self,X,y,**kwargs):
        X,y = self.preprocessor.fit_transform(X,y)
        self.regressor.fit(X,y)
        
    def predict(self,X,y,**kwargs):
        X,y = self.preprocessor.transform(X,y)
        output = self.regressor.predict(X)
        return(output)
        
    def save(self,storing,keys):
        keys_regressor = np.copy(keys).tolist()
        keys_regressor.append('regressor')
        self.regressor.save(storing,keys_regressor)
        keys_preprocessor = np.copy(keys).tolist()
        keys_preprocessor.append('preprocessor')
        self.preprocessor.save(storing,keys_preprocessor)
        
    def load(self,storing,keys):
        keys_regressor = np.copy(keys).tolist().append('regressor')
        self.regressor.load(storing,keys_regressor)
        keys_preprocessor = np.copy(keys).tolist().append('preprocessor')
        self.preprocessor.load(storing,keys_preprocessor)
    
    def get_params(self):
        params = {'estimator':self.estimator}
        return params