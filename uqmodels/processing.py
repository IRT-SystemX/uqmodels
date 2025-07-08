#######################################################################################################
# Source link to base functionality for data processing

import json
import os
import pickle
from abc import ABC
from pathlib import Path

import numpy
import pandas as pd
from sklearn.base import BaseEstimator

from uqmodels.preprocessing.structure import Structure

# Exeption ###################################


class EstimatorNotFitted(Exception):
    pass


# Data Loader ##################################


def split_path(path):
    """Split path into list of folder name using path.split iterativly.

    Args:
        path (str): path
    """
    folder_name = []
    while path is not "":
        path, tail = os.path.split(path)
        print(path)
        folder_name.append(tail)
    return folder_name[::-1]


def to_list(obj):
    """Put obj in list or do nothing if obj is already a list"""
    if not isinstance(obj, list):
        return [obj]
    else:
        return obj


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, Structure):
            return obj.toJSON()
        elif callable(obj):
            return obj.__name__
        elif hasattr(obj, "isoformat"):
            return obj.isoformat()
        else:
            try:
                return super(MyEncoder, self).default(obj)
            except BaseException:
                return obj.__name__


# Naive read and load API ##################################


def write_function(values, filename):
    """Auxiliar write function for file manipulation that hold csv/ pickle /json

    Args:
        values(obj): values to store
        filename (str): name of file

    Raises:
        FileNotFoundError: Error raise when file is not found
    """
    if isinstance(values, pd.core.frame.DataFrame):
        write_type = "pandas"
    elif str(filename).find(".json") > -1:
        write_type = "json"
    else:
        write_type = "pickle"

    if write_type == "pickle":
        filename_p = Path((str(filename) + ".p").replace(".p.p", ".p"))
        pickle.dump(values, open(filename_p, "wb"))

    elif write_type == "json":
        with open(filename, "w") as fp:
            for key in values.keys():
                if not isinstance(values[key], str):
                    pass
            json.dump(values, fp, cls=MyEncoder)

    elif write_type == "pandas":
        filename_csv = Path((str(filename) + ".csv").replace(".csv.csv", ".csv"))
        file = open(filename_csv, "w")
        values.to_csv(file)
        file.close()


def read_function(filename):
    """Auxiliar read function for file manipulation that hold csv/ pickle /json

    Args:
        filename (str): name of file

    Raises:
        FileNotFoundError: _description_

    Returns:
        values: values loaded
    """
    values = None
    filename_csv = Path((str(filename) + ".csv").replace(".csv.csv", ".csv"))
    filename_p = Path((str(filename) + ".p").replace(".p.p", ".p"))
    filename_json = Path((str(filename) + ".json").replace(".json.json", ".json"))
    flag_csv = False
    flag_p = False
    flag_json = False

    read_type = "None"
    if filename_csv.is_file():
        read_type = "pandas"
        flag_csv = True

    if filename_p.is_file():
        read_type = "pickle"
        flag_p = True

    if filename_json.is_file():
        read_type = "json"
        flag_json = True

    if (flag_csv & flag_p) & flag_json:
        print(
            "warning csv/pickle/json with same name : "
            + filename
            + ".p priority to pickle file"
        )

    if read_type == "pickle":
        file = open(filename_p, "rb")
        values = pickle.load(file)
        file.close()

    elif read_type == "json":
        file = open(filename_json, "r")
        values = json.load(file)
        file.close()

    elif read_type == "pandas":
        values = pd.read_csv(open(filename_csv, "rb"))

    elif Path(filename).is_dir():
        values = str(filename)

    else:
        raise FileNotFoundError("Warning: not found", str(filename))

    return values


def write(storing, keys, values, **kwargs):
    """Write API for file management

    Args:
        storing (str): global path of values to read
        keys (list of str): local path as list of folder + filename as last key
        values (obj): values to write

    """
    if isinstance(storing, dict):
        mode = "dict"
    elif isinstance(storing, str):
        mode = "file"
    else:
        print("storing have to be a 'dict' or 'str path_file'", type(storing))

    if mode == "dict":
        sub_dict = storing
        if isinstance(keys, str):
            keys = split_path(keys)

        for k in keys[:-1]:
            if k not in list(sub_dict.keys()):
                sub_dict[k] = {}
            sub_dict = sub_dict[k]
        sub_dict[keys[-1]] = values

    elif mode == "file":
        full_path = storing
        if isinstance(keys, str):
            keys = split_path(keys)

        for k in keys[:-1]:
            full_path = os.path.join(full_path, k)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
        full_path = os.path.join(full_path, keys[-1])
        filename = Path(full_path)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        write_function(values, filename)


def read(storing, keys, **kwargs):
    """Read API for file management

    Args:
        storing (str): global path of values to read
        keys (list of str): local path as list of folder + filename as last key
    """

    if isinstance(storing, dict):
        mode = "dict"
    elif isinstance(storing, str):
        mode = "file"
    else:
        print("storing have to be a 'dict' or 'str path_file'")

    if mode == "dict":
        sub_dict = storing
        if isinstance(keys, str):
            return sub_dict[keys]
        else:
            for n, k in enumerate(keys):
                if k in list(sub_dict.keys()):
                    sub_dict = sub_dict[k]
                    if n + 1 == len(keys):
                        return sub_dict
                else:
                    return None

    elif mode == "file":
        if not isinstance(storing, str):
            print("ERROR : storing is not a path")
        full_path = storing
        if isinstance(keys, str):
            full_path = os.path.join(full_path, keys)
        else:
            full_path = os.path.join(full_path, *keys)
        filename = Path(full_path)
        return read_function(filename)

    else:
        print("mode have to be 'dict' or 'file'")


# Data Loader ##################################


class Data_loader(ABC):
    def __init__(self, data_loader_api=read):
        """Data loader object : aim to instanciate a generic data loader that handle a query to return selected data.

        Args:
            data_loader_api (_type_, optional): API of loading . Defaults to read form store.py
        """
        self.data_loader_api = data_loader_api  # API loader

    def load(self, dict_query):
        """load form a dict_query that will be provide to the data_loader_api function

        Args:
            dict_query (dict): query as a dict that contains argument of the self.data_loader_api

        Raises:
            FileNotFoundError: error if file not found

        Returns:
            selected_data: selected_data loaded by the data_loader_api function from the dict_query
        """
        # Load from data storage using data_link (or API)
        data = self.data_loader_api(**dict_query)
        # Select data from query if can't not be done by data provider.
        selected_data = data
        if data is None:
            raise FileNotFoundError("Erreur query", dict_query)
        return selected_data


# Cache ##################################


def set_query_to_cache_query(filename, with_arborescence=False, storing=None):
    """Generate function that translate query to a cache API query
        Only compatible with write/read uqmodels.processing API
    Args:
        filename (_type_): Name of file to store data & processor
        with_arborescence (bool, optional): store in file arborescence. Defaults to False.
    returns:
        query_to_cache_query : function for cache manager object
    """

    def query_to_cache_query(query, filename=filename, values=None):
        cache_query = {"storing": storing, "keys": []}

        if "storing" in query.keys():
            cache_query["storing"] = query["storing"]

        if values is not None:
            cache_query["values"] = values

        elif "values" in query.keys():
            cache_query["values"] = query["values"]

        # Generation of storage location

        if "keys" in query.keys():
            cache_query["keys"] = query["keys"]

        if "name" in query.keys():
            cache_query["keys"].append(query["name"])

        if "processing" in query.keys():
            for name in query["processing"]:
                cache_query["keys"].append(name)

        if filename is not None:
            cache_query["keys"].append(filename)

        if not (with_arborescence):
            new_keys = cache_query["keys"][0]
            for key in cache_query["keys"][1:]:
                new_keys = new_keys + "_" + key
            cache_query["keys"] = new_keys

        if cache_query["keys"] == []:
            cache_query["keys"] = ["data"]

        return cache_query

    return query_to_cache_query


class Cache_manager:
    def __init__(
        self,
        save_API=write,
        load_API=read,
        storing="",
        default_filename="",
        query_to_cache_query=None,
    ):
        """Cache manager object : aim to save/load results or estimators using provided save_API & load_API
        and query_to_cache_query to transform query into cache_manager query

        Args:
            save_API (_type_, optional): save API. Defaults to write.
            load_API (_type_, optional): load API. Defaults to read.
            storing (str, optional): storing path. Defaults to 'data'.
            query_to_cache_query (function, optional): function that create cache manager query from query.
                Defaults to default_query_to_cache_query.
        """
        self.storing = storing
        self.default_filename = default_filename
        self.save_API = save_API
        self.load_API = load_API
        self.query_to_cache_query = query_to_cache_query

    def load(self, query, filename=None):
        """load obj at query + filname location using save_API

        Args:
            query (dict): query to interpret by query_to_cache_query
            filename (str, optional): filename of object if not provided by query. Defaults to None.

        Raises:
            FileNotFoundError: _description_
            FileNotFoundError: _description_

        Returns:
            _type_: _description_
        """
        if type(query is dict):
            if "storing" not in query.keys():
                query["storing"] = self.storing

        if filename is None:
            filename = self.default_filename

        if self.query_to_cache_query is None:
            new_query = self.default_query_to_cache_query(query, filename)
        else:
            new_query = self.query_to_cache_query(query, filename)
        try:
            obj = self.load_API(**new_query)

        except (FileNotFoundError, NotADirectoryError):
            raise FileNotFoundError(new_query["storing"], new_query["keys"])

        if obj is None:
            raise FileNotFoundError(new_query["storing"], new_query["keys"])

        return obj

    def save(self, query, obj, filename=None, verbose=False):
        """save obj at query + filename location using load_API

        Args:
            query (dict): query to interpret by query_to_cache_query
            obj (obj): object to store by save_API
            filename (_type_, optional): filename of object if not provided by query. Defaults to None.
        """
        # Save data link to a query on cache_link using it's unique id-query
        # print('save', query, filename)

        if query is not {}:

            # Use by default storing and filename
            if type(query is dict):
                if "storing" not in query.keys():
                    query["storing"] = self.storing

            if filename is None:
                filename = self.default_filename

            if self.query_to_cache_query is None:
                new_query = self.default_query_to_cache_query(
                    query, filename, values=obj
                )
            else:
                new_query = self.query_to_cache_query(query, filename, values=obj)

            if verbose:
                print("save", new_query["keys"], type(new_query["values"]))
            # Replace name_of_file if specified

            self.save_API(**new_query)
        else:
            print("skip save")

    def default_query_to_cache_query(self, query, filename=None, values=None):
        """default_query_to_keys : function that create a Save/Load query for the cache_manager load/save api_function
        from query

        Args:
            query (_type_): query to interpret
            filename (str, optional): default_storage_filename. Defaults to 'object.p'.
            values (obj or None, optional): obj to store by load_API/ if None then query_to_save_API

        Returns:
            dict_new_q: dict_parameters to provide to save_API or load_API
        """
        dict_new_q = {"storing": "", "keys": []}

        if values is not None:
            dict_new_q["values"] = values

        if "storing" in query.keys():
            dict_new_q["storing"] = query["storing"]
        else:
            print("warning : Incompatible query for default_query_to_new_query")
        if "keys" in query.keys():
            dict_new_q["keys"] = query["keys"]

        if "name" in query.keys():
            dict_new_q["keys"].append(query["name"])

        if "source" in query.keys():
            dict_new_q["keys"].append(query["source"])

        if filename is not None:
            dict_new_q["keys"].append(filename)

        if dict_new_q["keys"] == []:
            print("warning : Incompatible query for default_query_to_new_query")

        return dict_new_q


# Processor ##################################


class Processor:
    def __init__(self, name="processor", cache=None, update_query=None, **kwargs):
        """Processor class : that aim to process data in a (fit/transform) scheme and hold a cache manager
        functionality to save/load object

        Args:
            name (str, optional): Name of processor. Defaults to 'processor'.
            cache (Cache_manager or None, optional): Cache manager. Defaults to None : no save/load procedure
            update_query (function, optional): Function to update query due to Processor application if needed.
                Defaults to default_update_query : no update/
        """
        self.name = name
        self.cache = cache
        self._update_query = update_query
        for key_arg in kwargs.keys():
            setattr(self, key_arg, kwargs[key_arg])
        self.is_fitted = False

    def default_update_query(self, query):
        type_ = type(query)
        if type_ == dict:
            pass
        if type_ == str:
            pass
        else:
            print(
                "Warning type of query not hold by update_query : define your own function of query update : "
                'replace by str "data_"'
            )
            query = "data_"
        return query

    def fit(self, data=None):
        """Fit Processor using data

        Args:
            data (obj, optional): data. Defaults to None.
        """
        # Fit processor using train_data
        self.is_fitted = True

    def transform(self, data=None):
        """Apply Processor to data

        Args:
            data (obj, optional): data. Defaults to None.
        """
        return data

    def fit_transform(self, data=None):
        """Fit Processor and apply it on data

        Args:
            data (obj, optional): data. Defaults to None.
        """
        self.fit(data)
        data = self.transform(data)
        return data

    def save(self, query=None, object=None, name=None):
        """Save method to store object at queery+name location using cache_manager

        Args:
            query (dict, optional): query_paramaters. Defaults to None.
            object (obj, optional): object to store. Defaults to None.
            name (_type_, optional): filename of obj to store. Defaults to None.
        """
        if self.cache is not None:
            if object is not None:
                self.cache.save(query, object, name)
            else:
                self.cache.save(query, self, self.name)

    def load(self, query=None, name=None):
        """Load method to load Processor at query+name location using cache_manager

        Args:
            query (dict, optional): query_paramaters. Defaults to None.
            name (_type_, optional): filename of obj to load. Defaults to None.
        """
        loaded_processor = None

        if self.cache is None:
            raise FileNotFoundError

        else:
            try:
                if name is None:
                    name = self.name
                loaded_processor = self.cache.load(query, name)
                for property, value in vars(loaded_processor).items():
                    self.__setattr__(property, value)
                self.is_fitted = True
            except BaseException:
                raise FileNotFoundError

        return loaded_processor

    def update_query(self, query):
        """Apply the update_query_function provided at init to update query
        Args:
            query (dict): query

        Returns:
            new_query: updated query
        """
        if self._update_query is None:
            new_query = self.default_update_query(query)
        else:
            new_query = self._update_query(query)
        return new_query

    def use_cache(self, query):
        """Use_cache manager to check if there is cache link to data already processed

        Args:
            query (dict): query

        Raises:
            FileNotFoundError: cache Not Found error caught by method that called use_case

        Returns:
            data: data if
        """
        query = self.update_query(query)
        if self.cache is None:
            raise FileNotFoundError
        else:
            try:
                data = self.cache.load(query)
                return data

            except (FileNotFoundError, NotADirectoryError):
                raise FileNotFoundError


# Pipeline ##################################


class Pipeline(BaseEstimator):
    # Preprocessing Pipeline : Load, Formalisation and Cache management for each query (i.e each similar data source)
    def __init__(
        self, data_loader=None, list_processors=[], verbose=False, skip_cache=False
    ):
        """Pipeline object aim to apply a processing pipeline that include data_loader and sequence of
        Processing with cache management procedure for each Processor

        Args:
            data_loader (Data_loader): Data_loader that use a load_API to load data
            list_processors (list of Processor): List of Processor to fit and apply
            verbose (bool, optional): use verbose mode or not. Defaults to False.
        """
        self.skip_cache = False
        self.data_loader = data_loader
        self.list_processors = list_processors
        self.verbose = verbose

    def fit(self, query, save_processor=False):
        """Apply a fiting procedure to the pipeline with data_loader and a list of processor to a query/querries_list
            if no data_loader provide data instead of "query"

        Args:
            query_or_list (_type_): Query or List of query to provide to the data_loader

        """
        if self.verbose:
            print("Fiting procedure")

        for q in to_list(query):
            if self.verbose:
                print("For ", q)

            if self.data_loader is None:
                data = q

            else:
                data = self.data_loader.load(q)

            if self.verbose:
                print("Data loaded")

            for n, processor in enumerate(self.list_processors):
                try:
                    processor.load(q)
                    if self.verbose:
                        print("Skip fit " + processor.name)

                except BaseException:
                    if self.verbose:
                        print("Fit " + processor.name)
                        if hasattr(processor, "transform"):  # Processor
                            processor.fit(data, q, save_processor)
                        else:  # Estimator
                            X, y = data[0], data[1]
                            processor.fit(X, y)

                # Update data using processor
                data = processor.transform(data, q)
                # Update query
                q = processor.update_query(q)

    def transform(self, query_or_list):
        """Apply a pipeline with data_loader and a list of processor to a query o a list of query provide a generator
            If query not found in cache, Load data and tranform data using fitted processor
            If no data_loader provide data instead of "query_or_list"

        Args:
            query_or_list (query): Query or List of query to provide to the data_loader

        Returns:
            a Data gerenator that provide returned by the last Processor of the piepline

        Yields:
            generator : generator that will apply pipeline to each querries provided.
        """

        if self.verbose:
            print("Transform procedure")

        for q in to_list(query_or_list):
            # Generate the id_querry

            if self.verbose:
                print("Work on ", str(q["source"]))
            # Look if there is data in some cache-step
            strat_marker = len(self.list_processors)

            # For each formalizer (in reverse order)
            for n, processor in enumerate(self.list_processors[::-1]):
                new_q = q
                try:
                    # Update query upon actual processor : strat_marker - n - 1
                    # Doesn't consider himself, since it udpate_query is done in use_cache
                    for proc in self.list_processors[: strat_marker - n - 1]:
                        if hasattr(proc, "update_query"):
                            new_q = proc.update_query(new_q)

                    if hasattr(processor, "use_cache"):
                        data = processor.use_cache(new_q)
                        print("Recover data from cache :", new_q)
                        break

                except FileNotFoundError:
                    # Marker of pipeline start :
                    strat_marker += -1

            if strat_marker == 0:
                print(
                    "Pipeline : No cache data found : load data and execute the whole pipeline"
                )
                # If no cache found load data
                if self.data_loader is None:
                    data = q

                else:
                    data = self.data_loader.load(q)

                new_q = q
            # Apply chain of processor
            for n, processor in enumerate(self.list_processors):
                if n < strat_marker:
                    # Pass step upsteam to the cache
                    if self.verbose:
                        print("Skip " + processor.name)
                else:
                    if hasattr(processor, "transform"):
                        if self.verbose:
                            print("Tranform " + processor.name)
                        data = processor.transform(data, new_q)
                        new_q = processor.update_query(new_q)

                    elif hasattr(processor, "predict"):
                        if len(data) == 2:
                            X, y = data
                        if len(data) > 2:
                            X, y = data[0], data[1]
                        data = processor.predict(X, y)

            yield data

    def fit_transform(self, query_or_list, save_processor=False):
        """Fit tranform procedure : apply sucessively fit then transform

        Args:
            query_or_list (_type_): query or list of query to apply
            save_processor (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        self.fit(query_or_list, save_processor)
        data_processed = self.transform(query_or_list)
        return data_processed
