from abc import ABC, abstractmethod
from typing import Any, Optional, Dict

class BaseComponent(ABC):
    """
    Base abstraction for AI components.

    A component orchestrates a set of blocks with harmonized interfaces.
    It is not tied to uncertainty quantification.
    """

    def fit(
        self,
        X: Any,
        y: Optional[Any] = None,
        context: Optional[Any] = None,
        **kwargs,
    ) -> "BaseComponent":
        """
        Fit the component or its internal trainable blocks.
        """
        return self

    @abstractmethod
    def predict(
        self,
        X: Any,
        context: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """
        Produce component outputs.
        """
        raise NotImplementedError

    def score(
        self,
        X: Any,
        y: Optional[Any] = None,
        context: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """
        Optional scoring method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement score()."
        )

    def get_params(self) -> Dict[str, Any]:
        """
        Return serializable component parameters.
        """
        return {
            "class_name": self.__class__.__name__,
        }

    def save(self, path: str) -> None:
        """
        Optional persistence interface.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement save()."
        )

    @classmethod
    def load(cls, path: str) -> "BaseComponent":
        """
        Optional loading interface.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement load()."
        )
    
from typing import Any, Optional, Dict


class BaseBlock(ABC):
    """
    Base abstraction for reusable blocks inside a component.

    A block can be a model, processor, scorer, thresholding module,
    postprocessor, formatter, or adapter.
    """

    def fit(
        self,
        data: Any = None,
        y: Optional[Any] = None,
        context: Optional[Any] = None,
        **kwargs,
    ) -> "BaseBlock":
        return self

    def transform(
        self,
        data: Any,
        context: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        return data

    def fit_transform(
        self,
        data: Any,
        y: Optional[Any] = None,
        context: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        self.fit(data=data, y=y, context=context, **kwargs)
        return self.transform(data=data, context=context, **kwargs)

    def get_params(self) -> Dict[str, Any]:
        return {
            "class_name": self.__class__.__name__,
        }
    
from dataclasses import dataclass, field
from typing import Any, Optional, Dict


@dataclass
class ComponentOutput:
    """
    Standard internal output object for benchmark-compatible components.

    This object should not force legacy UQModel to change its public output.
    """
    prediction: Optional[Any] = None
    reconstruction: Optional[Any] = None
    uncertainty: Optional[Any] = None
    score: Optional[Any] = None
    y_pred: Optional[Any] = None
    decision_margin: Optional[Any] = None
    raw_output: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction": self.prediction,
            "reconstruction": self.reconstruction,
            "uncertainty": self.uncertainty,
            "score": self.score,
            "y_pred": self.y_pred,
            "decision_margin": self.decision_margin,
            "raw_output": self.raw_output,
            "metadata": self.metadata,
        }
    
class PipelineComponent(BaseComponent):
    """
    Generic component built as an ordered assembly of interface-compatible blocks.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        preprocessors: Optional[List[BaseBlock]] = None,
        postprocessors: Optional[List[BaseBlock]] = None,
        output_formatter: Optional[BaseBlock] = None,
    ):
        self.model = model
        self.preprocessors = preprocessors or []
        self.postprocessors = postprocessors or []
        self.output_formatter = output_formatter

    def fit(
        self,
        X: Any,
        y: Optional[Any] = None,
        context: Optional[Any] = None,
        **kwargs,
    ) -> "PipelineComponent":
        X_proc = X

        for processor in self.preprocessors:
            X_proc = processor.fit_transform(
                data=X_proc,
                y=y,
                context=context,
                **kwargs,
            )

        if self.model is not None and hasattr(self.model, "fit"):
            self.model.fit(X_proc, y, **kwargs)

        return self

    def predict(
        self,
        X: Any,
        context: Optional[Any] = None,
        **kwargs,
    ) -> ComponentOutput:
        X_proc = X

        for processor in self.preprocessors:
            X_proc = processor.transform(
                data=X_proc,
                context=context,
                **kwargs,
            )

        if self.model is None:
            raw_output = X_proc
        else:
            raw_output = self.model.predict(X_proc, **kwargs)

        output = ComponentOutput(
            prediction=raw_output,
            raw_output=raw_output,
            metadata={
                "component_type": self.__class__.__name__,
            },
        )

        for processor in self.postprocessors:
            output = processor.transform(
                data=output,
                context=context,
                **kwargs,
            )

        if self.output_formatter is not None:
            return self.output_formatter.transform(
                data=output,
                context=context,
                **kwargs,
            )

        return output

    def get_params(self) -> Dict[str, Any]:
        return {
            "class_name": self.__class__.__name__,
            "model": self.model.__class__.__name__ if self.model is not None else None,
            "preprocessors": [
                p.get_params() if hasattr(p, "get_params") else str(p)
                for p in self.preprocessors
            ],
            "postprocessors": [
                p.get_params() if hasattr(p, "get_params") else str(p)
                for p in self.postprocessors
            ],
            "output_formatter": (
                self.output_formatter.get_params()
                if self.output_formatter is not None
                and hasattr(self.output_formatter, "get_params")
                else None
            ),
        }