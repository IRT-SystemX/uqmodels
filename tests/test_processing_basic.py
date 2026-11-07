import json
import numpy as np
import pandas as pd
from uqmodels.processing import split_path, to_list, MyEncoder


def test_myencoder():
    enc = MyEncoder()
    data = {"a": np.int64(1), "b": np.array([1, 2]), "c": 2.5}
    s = json.dumps(data, cls=MyEncoder)
    assert "1" in s
