from uqmodels.processing import read


def test_read_dict_mode():
    storing = {"a": {"b": 42}}
    assert read(storing, ["a", "b"]) == 42
    assert read(storing, "a") == {"b": 42}
