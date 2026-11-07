import json
import tempfile
from pathlib import Path
from uqmodels.processing import read, write_function


def test_read_write_file_mode(tmp_path):
    data = {"a": 1, "b": [2, 3]}
    f = tmp_path / "out.json"
    write_function(data, f)
    got = read(str(tmp_path), ["out.json"])
    assert got["a"] == 1
