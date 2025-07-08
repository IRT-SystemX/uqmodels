import numpy as np
import pytest

from uqmodels.preprocessing.structure import Structure
from uqmodels.processing import MyEncoder, to_list


@pytest.fixture
def data_list_2d():
    return [[0, 1], [2, 3]]


# def test_Data_loader(data_loader, synthetic_dataset_multivariate_info):
#     dirname, filename = synthetic_dataset_multivariate_info
#     dict_data = data_loader.load(
#         dict_query={
#             "dirname": dirname,
#             "filename": filename,
#         }
#     )
#     assert isinstance(dict_data, dict)
#     assert dict_data.keys()
#     assert dict_data.values()


# def test_Pipeline(data_loader, synthetic_dataset_multivariate_info):
#     dirname, filename = synthetic_dataset_multivariate_info
#     preprocessor = dict_to_TS_Dataset()
#     pipeline = Pipeline(data_loader=data_loader, list_processors=[preprocessor])
#     assert pipeline

#     list_query = [{"dirname": dirname, "filename": filename, "name": "Synthetic_data"}]
#     dataset_generator = pipeline.transform(list_query)
#     X, y, sample_weight, x_split, context, objective, name = next(dataset_generator)
#     train = x_split == 1
#     test = np.invert(train)
#     assert X.shape
#     assert y.shape
#     assert isinstance(name, str)


def test_to_list(data_list_2d):
    assert to_list(1) == [1]
    assert to_list(1.0) == [1.0]
    assert to_list("abc") == ["abc"]
    assert to_list(data_list_2d) == data_list_2d


def test_MyEncoder(data_list_2d):
    encoder = MyEncoder()
    assert encoder.default(np.int16(1)) == 1
    assert encoder.default(np.float32(1)) == 1.0
    assert encoder.default(np.array(data_list_2d)) == data_list_2d
    assert encoder.default(Structure("Data")) == Structure("Data").toJSON()
    assert encoder.default(print) == print.__name__
