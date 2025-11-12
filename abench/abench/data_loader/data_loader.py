from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Tuple, Dict,Any
from copy import deepcopy

class ABLoader(ABC):
    """
    Abstract base class representing a unified interface for train/test data loading,
    including optional context and metadata. Used in experimental pipelines to 
    standardize data retrieval per iteration.
    """

    def __init__(
        self,
        metadata: Optional[dict] = None,
        with_context: bool = True,
        with_metadata: bool = True,
        name = None
        ):
        """
        Initializes the ABLoader with core components.

        Args:
            dataloader (Iterator): Primary data iterator (required).
            contextloader (Iterator, optional): Iterator providing contextual information (e.g. demographics).
            metadata (dict, optional): Metadata dict, should include at least 'name' and 'target_arg'.
            with_context (bool): Whether to include context in iteration output.
            with_metadata (bool): Whether to include metadata in iteration output.
            name (str): Name of the Dataloader
        """

        self.metadata = metadata or {}
        self.metadata.setdefault('name', name)
        self.metadata.setdefault('target_arg', {})
        self.with_context = with_context
        self.with_metadata = with_metadata

    def get_setname(self) -> Optional[str]:
        """Returns the dataset name defined in metadata (if any)."""
        return self.metadata['name']

    def get_target_arg(self) -> dict:
        """Returns the target argument dictionary from metadata."""
        return self.metadata['target_arg']

    @abstractmethod
    def __iter__(self):
        """
        Abstract method to iterate over the dataset.
        Subclasses must implement logic to yield:
        - data only
        - or (data, context)
        - or (data, context, metadata)
        depending on internal flags or configuration.
        """
        pass
    def to_dict(self) -> dict:
        """
        Serializes metadata and configuration (excluding actual iterators).
        """
        return self.__dict__

    @classmethod
    def from_dict(cls, dict_arg: dict):
        """
        Creates an ABLoader from metadata and optional iterators.

        Args:
            data (dict): Dictionary with keys 'name', 'target_arg', etc.
            dataloader (Iterator): Required data loader.
            contextloader (Iterator): Optional context loader.
        """
        return cls(**dict_arg)

class ABLoaderGeneric(ABLoader):
    """
    Abstract base class representing a unified interface for train/test data loading,
    including optional context and metadata. Used in experimental pipelines to 
    standardize data retrieval per iteration.
    """

    def __init__(
        self,
        dataloader: Iterator,
        contextloader: Optional[Iterator] = None,
        metadata: Optional[dict] = None,
        with_context: bool = True,
        with_metadata: bool = True,
        name = None,
    ):
        """
        Initializes the ABLoader with core components.

        Args:
            dataloader (Iterator or list): Primary data iterator (required).
            contextloader (Iterator or list, optional): Iterator providing contextual information (e.g. demographics).
            metadata (dict, optional): Metadata dict, should include at least 'name' and 'target_arg'.
            with_context (bool): Whether to include context in iteration output.
            with_metadata (bool): Whether to include metadata in iteration output.
            name (str): Name of the Dataloader
        """
        if(isinstance(dataloader,list)):
            dataloader = iter(dataloader)
            
        if(isinstance(contextloader,list)):
            contextloader = iter(contextloader)


        self.dataloader = dataloader
        self.contextloader = contextloader
        self.metadata = metadata or {}

        self.metadata.setdefault('name', name)
        self.metadata.setdefault('target_arg', {})

        self.with_context = with_context
        self.with_metadata = with_metadata

    def get_setname(self) -> Optional[str]:
        """Returns the dataset name defined in metadata (if any)."""
        return self.metadata['name']

    def get_target_arg(self) -> dict:
        """Returns the target argument dictionary from metadata."""
        return self.metadata['target_arg']

    def __iter__(self):
        """
        Iterates over the dataset, optionally including context and metadata.
        Yields:
            A single object, tuple of (data, context), or (data, context, metadata).
        """
        for data in self.dataloader:
            output = [data]
            if self.with_context and self.contextloader is not None:
                output.append(next(self.contextloader))
            if self.with_metadata:
                output.append(self.metadata)
            yield output[0] if len(output) == 1 else tuple(output)

class ABLoaderGenericArray(ABLoaderGeneric):
    """
    ABLoaderThat take X,y,Context as ndarray and hold a split array for subselection
    """

    def __init__(self,
                 X,
                 y,
                 context,
                 split,
                 selected_split_indexes,
                 with_context=True,
                 with_metadata=True,
                 metadata={},
                 name='set'):
        
        mask_selection = [i in selected_split_indexes for i in split]
        X = X[mask_selection]
        y = y[mask_selection]
        context = context[mask_selection]
        dataloader = iter([(X,y)])
        contextloader = iter([context])
        name = name+'_'+str(selected_split_indexes)
        metadata = metadata
        super().__init__(dataloader,contextloader,metadata,with_context,with_metadata,name)


class ABDataExperiment(ABC):
    """
    Abstract base class for managing a cross-validation experimental setup,
    where each training set is paired with one or more test sets.

    Iterating over the object yields (train_loader, test_loader_list) pairs.
    """

    def __init__(
        self,
        ABtrainloader_list: List,
        ABtestloader_sets_list: Optional[List[List]] = None,
        name: str = 'crossval_experiment'
    ):
        """
        Initializes the cross-validation experiment.

        Args:
            ABtrainloader_list (List): List of ABLoader instances used for training.
            ABtestloader_sets_list (List[List], optional): List of test set groups per train set.
            name (str): Name of the experiment.
        """
        self._train_loaders = ABtrainloader_list
        self._test_loader_sets = ABtestloader_sets_list or [[] for _ in ABtrainloader_list]
        self.name = name

    def __iter__(self) -> Iterator[Tuple[Any, List[Any]]]:
        """
        Yields (train_loader, list_of_test_loaders) pairs for each experiment.

        Raises:
            ValueError: if test loader set list has fewer entries than train loaders.
        """
        for i, train_loader in enumerate(self._train_loaders):
            try:
                test_loader_set = self._test_loader_sets[i]
            except IndexError:
                raise ValueError("ABtestloader_sets_list has fewer entries than ABtrainloader_list.")
            yield train_loader, test_loader_set
    
    def get_experiment_plan(self) -> Dict[str, List[str]]:
        """
        Returns a structured dictionary describing the experiment plan.

        Example:
            {
                "TrainSet1": ["TestSetA", "TestSetB"],
                "TrainSet2": ["TestSetC"]
            }

        Returns:
            dict: Mapping from train set names to test set name lists.
        """
        dict_plan={}
        ABtrainloader_list = deepcopy(self._train_loaders)
        ABtestloader_sets_list = iter(deepcopy(self._test_loader_sets))
        for train_loader in ABtrainloader_list:
            test_loader_sets = next(ABtestloader_sets_list)
            trainset_name = train_loader.get_setname() 
            testset_name_list = [test_loader.get_setname() for test_loader in test_loader_sets]
            dict_plan[trainset_name]=testset_name_list
        return(dict_plan)   
    
    def to_dict(self) -> dict:
        """
        Serializes the experiment plan structure (only names and links).
        """
        plan = self.get_experiment_plan()
        return {
            'name': self.name,
            'experiment_plan': plan
        }
    
    def check_set_names(self):
        for A,B in self:
            print('Train',A.get_setname())
            for b in B:
                print('Test',b.get_setname())

    @classmethod
    def from_dict(cls, data: dict):
        """
        Stub method — needs to be implemented in concrete subclass.
        """
        raise NotImplementedError("Use a concrete subclass to load from dict.")
    




from typing import Callable, Any, Sequence
import numpy as np
from sklearn.model_selection import BaseCrossValidator

class ABCvDataExperiment(ABDataExperiment):
    """
    Concrete implementation of ABDataExperiment for cross-validation scenarios.

    This class automates the construction of a cross-validation experiment using
    a provided data loader function and a scikit-learn compatible splitter.

    Each fold generates one training ABLoader and one test ABLoader wrapped in a list
    (to match expected interface from ABDataExperiment).

    Attributes:
        name (str): Name of the cross-validation experiment.
    """

    def __init__(
        self,
        ABloader_fn: Callable[[Sequence[int]], Any],
        sk_split: BaseCrossValidator,
        split: Sequence = None,
        name: str = 'crossval_experiment'
    ):
        """
        Initializes the cross-validation experiment using a splitter and loader function.

        Args:
            ABloader_fn (Callable[[Sequence[int]], ABLoader]):
                A function that takes an index list (train or test indices) and returns an ABLoader.
            
            sk_split (BaseCrossValidator):
                A cross-validator from scikit-learn (e.g., KFold, StratifiedKFold).
            
            split (Sequence, optional):
                Sequence of elements to split. Default is `np.arange(6)` if not provided.
            
            name (str):
                Name of the experiment (used for identification/logging).
        """
        ABtrainloader_list = []
        ABtestloader_sets_list = []
        for train_idx, test_idx in sk_split.split(split):
            ABtrainloader_list.append(ABloader_fn(train_idx))
            ABtestloader_sets_list.append([ABloader_fn(test_idx)])  # List for compatibility

        super().__init__(
            ABtrainloader_list=ABtrainloader_list,
            ABtestloader_sets_list=ABtestloader_sets_list,
            name=name)
        

