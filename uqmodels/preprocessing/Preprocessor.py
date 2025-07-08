#####################################################################################
# Source link to Preprocessor class :
# Preprocessing pipeline can combine several Preprocessing.
# We suggest to split : Raw_data -> (Raw_data_preprossing) -> Clean_data -> (ML-Preprocessor_Porcessing) -> ML-Dataset
# Then we can produce from a same clean_data several ML-Dataset
# Cache mecanism aim to avoid to do same preprocessing calculation

import copy

from uqmodels.preprocessing.structure import Structure
from uqmodels.processing import Processor


class Preprocessor(Processor):
    def __init__(
        self, name="formaliser", cache=None, structure=None, update_query=None, **kwargs
    ):
        """Preprocessor class (inherit from Processor) : that aim to preprocess data in a (fit/transform) scheme
        and hold a cache manager functionality to save/load object

        Args:
            name (str, optional): Name of processor. Defaults to 'processor'.
            cache (Cache_manager or None, optional): Cache manager. Defaults to None : no save/load procedure
            structure (obj or None): structure that contains specification about how data has to be structured
                after preprocessing
            update_query (function, optional): Function to update query due to Processor application if needed.
                Defaults to default_update_query : no update/
        """
        super().__init__(
            name=name, cache=cache, structure=None, update_query=update_query, **kwargs
        )

        if structure is None:
            self.structure = Structure("Data")
        else:
            self.structure = structure

        for key in kwargs:
            self.structure.__setattr__(key, kwargs[key])

        self.is_fitted = False

    def default_update_query(self, query, name):
        if isinstance(query, dict):
            new_query = query.copy()
            if "processing" in query.keys():
                new_query["processing"].append(name)
            else:
                new_query["processing"] = [name]
        else:
            new_query = super().default_update_query(query)
        return new_query

    def get(self, keys, default_value=None):
        """Get obj from structure using structure.get

        Args:
            keys (_type_): key or list of keys related to attributes to get
            default_value (_type_, optional): default_value if no attribute. Defaults to None.
        """
        self.structure.get(keys, default_value)

    def set(self, key, obj):
        """Set ogj in structure using structure.get

        Args:
            keys (_type_): key or list of keys related to attributes to get

            obj (_type_): _description_
        """
        self.structure.set(key, obj)

    def fit(self, data=None, query={}, save_preprocessor=False):
        """Fit Preprocessing using data

        Args:
            data (obj, optional): data. Defaults to None.
            query: dict_query that generated the data
            save_formaliser (bool, optional): boolean flag that inform if we have to save preprocessor or not
        """
        # Fit formaliser using train_data
        super().fit()
        if save_preprocessor:
            new_query = copy.copy(query)
            query["name"] = self.name
            self.save(new_query)

    def transform(self, data=None, query={}):
        """Apply Preprocessor to data
        Args:
            data (obj, optional): data. Defaults to None.
            query: dict_query that generated the data

        Return
            data : Preprocessed data
        """
        query = self.update_query(query)
        if self.cache is not None:
            self.cache.save(query, data)
        super().transform(data)
        return data

    def fit_transform(self, data=None, query={}):
        """Fit Processor and apply it on data

        Args:
            data (obj, optional): data. Defaults to None.
            query: dict_query that generated the data.

        Return
            data : Preprocessed data
        """

        self.fit(data, query)
        data = self.transform(data, query)
        return data

    def update_query(self, query={}):
        """Apply the update_query_function provided at init to update query
        Args:
            query (dict): dict_query that generated the data.

        Returns:
            new_query: updated query
        """
        if self._update_query is None:
            new_query = self.default_update_query(query, self.name)
        else:
            new_query = self._update_query(query, self.name)
        return new_query

    def use_cache(self, query={}):
        """Use_cache manager to check if there is cache link to data already processed

        Args:
            query (dict): dict_query that generated the data.

        Raises:
            FileNotFoundError: cache Not Found error caught by method that called use_case

        Returns:
            data: if file is found else error
        """
        try:
            data = super().use_cache(query)

        except (FileNotFoundError, NotADirectoryError):
            raise FileNotFoundError()

        return data

    def save(self, query={}, object=None, name="data"):
        """Save method to store object at query+name location using cache_manager

        Args:
            query (dict, optional): dict_query that generated the data.
            object (obj, optional): object to store. Defaults to None.
            name (_type_, optional): filename of obj to store. Defaults to None.
        """
        super().save(query, object, name)

    def load(self, query={}, name="data"):
        """Load method to load Preprocessor at query+name location using cache_manager and use it parameters

        Args:
            query (dict, optional): query_paramaters. Defaults to None.
            name (_type_, optional): filename of obj to load. Defaults to None.
        """
        # Load fitted formaliser
        object = super().load(query, name)
        return object


# GENERIC_Preprocessor


def fit_default(self, data, query={}, structure=None):
    """fit function that done nothing

    Args:
        data (obj): data
        query (dict): dict_query that generated the data.
        structure (structure obj, optional): structure object that provide all meta information about data.
    """


def transform_default(self, data, query={}, structure=None):
    """Transform+ function that done nothing

    Args:
        data (obj): data
        query (dict): dict_query that generated the data.
        structure (structure obj, optional): structure object that provide all meta information about data.
    """
    return data


# Default Preprocessor :


class Generic_Preprocessor(Preprocessor):
    def __init__(
        self,
        name="Generic_preprocessor",
        cache=None,
        structure=None,
        update_query=None,
        fit_function=fit_default,
        transform_function=transform_default,
        **kwargs
    ):
        """Preprocessor class (inherit from Processor) : that aim to preprocess data in a (fit/transform) scheme and
        hold a cache manager functionality to save/load object

        Args:
            name (str, optional): Name of processor. Defaults to 'processor'.
            cache (Cache_manager or None, optional): Cache manager. Defaults to None : no save/load procedure
            structure (obj or None): structure that contains specification about how data has to be
                structured after preprocessing
            update_query (function, optional): Function to update query due to Processor application if needed.
                Defaults to default_update_query : no update/
            fit_function = function to apply in fit procedure. Defaults to fit_default that does nothing.
            transform_function = function to apply in tranform procedure.
                Defaults to transform_default that does nothing.

        """

        super().__init__(
            name=name,
            cache=cache,
            structure=structure,
            update_query=update_query,
            **kwargs
        )

        self.fit_function = fit_function
        self.transform_function = transform_function

    def fit(self, data, query={}):
        """Apply fit_function on data with query as query and self.structure as metadata
            if query has an "source" attribute:
                try to access to corrrespoding substructure by structure.get_structure(query[source])

        Args:
            data (obj, optional): data. Defaults to None.
            query: dict_query that generated the data
            save_formaliser (bool, optional): boolean flag that inform if we have to save preprocessor or not
        """
        structure = self.structure
        if "source" in query.keys():
            structure = self.structure.get_structure(query["source"])

        self.fit_function(self, data, query, structure)
        return super().fit(data)

    def transform(self, data, query={}, **kwarg):
        """Apply transform_function on data with query as query and self.structure as metadata
            if query has an "source" attribute:
                try to access to corrrespoding substructure by structure.get_structure(query[source])
        Args:
            data (obj, optional): data. Defaults to None.
            query: dict_query that generated the data

        Return
            data : Preprocessed data
        """
        structure = self.structure
        if "source" in query.keys():
            structure = self.structure.get_structure(query["source"])

        data = self.transform_function(self, data, query, structure)
        data = super().transform(data, query)
        return data
