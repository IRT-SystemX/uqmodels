import numpy as np
import pytest
from uqmodels.component import BaseComponent, BaseBlock, ComponentOutput, PipelineComponent


class DummyBlock(BaseBlock):
    def transform(self, data, context=None, **kwargs):
        return data


from sklearn.linear_model import LinearRegression


class DummyModel(LinearRegression):
    pass


def test_base_component_defaults():
    class Impl(BaseComponent):
        def predict(self, X, context=None, **kwargs):
            return X

    c = Impl()
    assert c.fit(np.array([1, 2])).get_params()["class_name"] == "Impl"
    assert c.predict(np.array([1, 2])).tolist() == [1, 2]
    with pytest.raises(NotImplementedError):
        c.score(np.array([1, 2]))
    with pytest.raises(NotImplementedError):
        c.save("x")
    with pytest.raises(NotImplementedError):
        Impl.load("x")


def test_base_block_transform_and_fit_transform():
    b = DummyBlock()
    data = np.array([1, 2, 3])
    assert b.fit(data).get_params()["class_name"] == "DummyBlock"
    assert b.transform(data).tolist() == [1, 2, 3]
    assert b.fit_transform(data).tolist() == [1, 2, 3]


def test_component_output_to_dict():
    out = ComponentOutput(prediction=1, uncertainty=2, metadata={"a": 3})
    d = out.to_dict()
    assert d["prediction"] == 1
    assert d["uncertainty"] == 2
    assert d["metadata"] == {"a": 3}


def test_pipeline_component_fit_predict():
    model = DummyModel()
    pre = DummyBlock()
    post = DummyBlock()
    comp = PipelineComponent(model=model, preprocessors=[pre], postprocessors=[post])
    X = np.array([[1], [2], [3]])
    y = np.array([10, 20, 30])
    comp.fit(X, y)
    assert hasattr(model, "coef_")
    out = comp.predict(X)
    assert isinstance(out, ComponentOutput)
    assert out.raw_output.shape[0] == X.shape[0]
    assert out.metadata["component_type"] == "PipelineComponent"


def test_pipeline_component_no_model():
    comp = PipelineComponent(model=None)
    X = np.array([1, 2, 3])
    out = comp.predict(X)
    assert out.raw_output.tolist() == [1, 2, 3]
