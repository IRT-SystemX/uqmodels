from abench.store.store import write,read
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import numpy as np


# Submodule preprocessor
class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, scaler_type='none'):
        self.scaler_type = scaler_type
        self.scaler = self._get_scaler(scaler_type)
        self.name = scaler_type
        
    def _get_scaler(self, scaler_type):
        scalers = {
            'none': None,
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'maxabs': MaxAbsScaler()
        }
        if scaler_type not in scalers:
            raise ValueError(f"Scaler '{scaler_type}' is not supported. Choose from {list(scalers.keys())}")
        return scalers[scaler_type]

    def fit(self, X, y=None):
        if(self.scaler is not None):
            self.scaler.fit(X)
        return self

    def transform(self, X, y=None):
        if(self.scaler is not None):
            X = self.scaler.transform(X)
        return X,y

    def save(self,storing,keys):
        write(storing,keys,self)
        
    def load(self,storing,keys):
        preprocessor = read(storing,keys)
        self.scaler_type = preprocessor.scaler_type
        self.scaler = preprocessor.scaler
        self.name = preprocessor.name
        return(self)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X,y)
