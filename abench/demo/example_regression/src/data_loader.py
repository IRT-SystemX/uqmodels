import abench
import numpy as np
import pandas as pd
import pickle
from functools import partial
from abench.data_loader.data_loader import ABLoader,ABLoaderGenericArray
from abench.data_loader.data_loader import ABDataExperiment,ABCvDataExperiment
from sklearn.model_selection import PredefinedSplit, KFold, TimeSeriesSplit

# Class Load that take a path and generate a iterable (X,y),Contexte,Metadata 
class MyABTimeSeriesFromDict(ABLoaderGenericArray):
    def __init__(self,
                 path='data/Dataset_Synthetic', 
                 selected_split_indexes=[0],
                 with_context=True,
                 with_metadata=True):
        """Read preprocessed  dataset stored in a dict with :
        """

        self.path = path
        dict_dataset = pickle.load(open(self.path,'rb'))
        name = 'syn'
        super().__init__(X=dict_dataset['X'],
                         y=dict_dataset['Y'],
                         context=dict_dataset['context'],
                         split=dict_dataset['X_split'],
                         selected_split_indexes=selected_split_indexes,
                         with_context=with_context,
                         with_metadata=with_metadata,
                         name=name)


# Class Experiment : Iterable providing pair of (Training_set,List_of_Test_set)
class MyABCvDataExperiment(ABCvDataExperiment):
    def __init__(self,
                 path='data/Dataset_Synthetic', 
                 sk_split=TimeSeriesSplit(5),
                 split=np.arange(6),
                 name='crossval_experiment'):
        
        ABloader_fn = partial(MyABTimeSeriesFromDict,path)
        super().__init__(ABloader_fn=ABloader_fn,
                         sk_split=sk_split,
                         split=split,
                         name='TS_CV_Experiment')
        