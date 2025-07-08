import numpy as np

from uqmodels.preprocessing.structure import str_to_date
from uqmodels.processing import Data_loader, read


class TS_csv_Data_loader(Data_loader):
    def __init__(self, data_loader_api=read):
        """Data loader object : aim to instanciate a generic data loader that handle a query to return selected data"""
        super().__init__(data_loader_api)  # Default API

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
        data_selection = self.data_loader_api(**dict_query)
        if data_selection is None:
            print("Erreur query :" + dict_query)
            raise FileNotFoundError()

        columns_selection = None
        if "columns_selection" in dict_query.keys():
            columns_selection = dict_query["columns_selection"]

        ind_begin, ind_end = None, None
        if "date_selection" in dict_query.keys():
            str_begin, str_end = dict_query["date_selection"]
            ind_begin = str_to_date(str_begin)
            ind_end = str_to_date(str_end)

        if "index_selection" in dict_query.keys():
            ind_begin, ind_end = dict_query["index_selection"]

        # Select data from query if can't not be done by data provider.

        ind_mask_begin = np.full(len(data_selection), True, dtype=bool)
        if ind_begin is not None:
            ind_mask_begin = data_selection.index >= ind_begin

        ind_mask_end = np.full(len(data_selection), True, dtype=bool)
        if ind_end is not None:
            ind_mask_end = data_selection.index <= ind_end

        ind_selection = ind_mask_begin & ind_mask_end

        if columns_selection is not None:
            data_selection = data_selection[columns_selection]
        data_selection = data_selection.loc[ind_selection]
        return data_selection
