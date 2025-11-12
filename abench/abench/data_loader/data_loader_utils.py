import pickle
import numpy as np
from abench.utils import Extract_dict
from sklearn.model_selection import PredefinedSplit, KFold, TimeSeriesSplit

############# OLD STUF TO DELETE #########""

class TimeSeries_from_dict:
    def __init__(self, path_file, **kwargs):
        """Read preprocessed  dataset stored in a dict with :
        'X' : Input
        'Y' : Target
        'context' : Additional information
        'train' : Boolean array for train split.
        'X_split' : Additional split information for cross-validation.
        """
        self.path_file = path_file

    def process(self, **kwargs):
        """Load dict at path_fill"""
        self.dict_dataset = pickle.load(open(self.path_file, "rb"))

    def get_data(self, **kwargs):
        """Provide dataset as list of array : [X,y,context,train,test,X_split]"""
        X = self.dict_dataset["X"]
        y = self.dict_dataset["Y"].reshape(len(X), -1)
        context = self.dict_dataset["context"]
        train = self.dict_dataset["train"]
        test = np.invert(train)
        X_split = self.dict_dataset["X_split"]
        return (X, y, context, train, test, X_split)

    def split_train_test(self, split=None):
        """Provide Train and Test data using predifine split or condition"""
        X = self.dict_dataset["X"]
        y = self.dict_dataset["Y"]
        context = self.dict_dataset["context"]
        X_split = self.dict_dataset["X_split"]
        if split == None:
            train = self.dict_dataset["train"]
            test = np.invert(train)
        else:
            train = X_split <= 1
            test = np.invert(train)

        return (X[train], X[test], y[train], y[test], context[train], context[test])

    def split_fit_calib(self, **kwargs):
        """None"""
        return ()

def dataset_generator_from_stored_dict(list_file):
    """Produce data_generator (iterable [X, y, context, objective, train, X_split]) from pickle stored dict (link)"""

    def load_data(file):
        dict_dataset = pickle.load(open(file, "rb"))
        list_str = ["X", "Y", "context", "objective", "train", "X_split"]
        X, y, context, objective, train, X_split = Extract_dict(dict_dataset, list_str)
        return (X, y, context, objective, train, X_split)

    dataset_generator = []
    for file in list_file:
        for i in list_file:
            load_data(file)
            yield
    return dataset_generator

# Encapsulated data from array :
def dataset_generator_from_array(
    X,
    y,
    context=None,
    objective=None,
    sk_split=TimeSeriesSplit(5),
    remove_from_train=None,
):
    """Produce data_generator (iterable [X, y, context, objective, train, X_split]) from arrays

    Args:
        X (array): Inputs.
        y (array or None): Targets.
        context (array or None): Additional information.
        objective (array or None): Ground truth (Unsupervised task).
        sk_split (split strategy): Sklearn split strategy."""

    def select_or_none(array, sample):
        if array is None:
            return None
        else:
            return array[sample]

    if remove_from_train is None:
        remove_from_train = np.zeros(len(X))

    dataset_generator = []

    for train_index, test_index in sk_split.split(X):
        train = np.zeros(len(X))
        train[train_index] = 1
        train[(train == 1) & (remove_from_train == 1)] = -1

        sample_cv = np.concatenate([train_index, test_index])
        sample_cv.sort()

        dataset_generator.append(
            [select_or_none(e, sample_cv) for e in [X, y, train, context, objective]]
        )
    return dataset_generator

