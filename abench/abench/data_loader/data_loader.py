from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Tuple, Dict,Any
from copy import deepcopy
from abench.store.api import store_ABDataExperiment,get_ABDataExperiment,store_ABloader,get_ABloader
   

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
    
    def save(self,storing,set_name):
        store_ABloader(storing,set_name,ABloader=self)

    @classmethod
    def load(cls,storing,set_name):
        ABloader = get_ABloader(storing,set_name)
        return(ABloader)



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

class ABLoaderAggregate(ABLoader):
    """
    Aggregate multiple ABLoader instances into a single iterable loader.
    """

    def __init__(
        self,
        ABloaders,
        name=None,
        metadata=None,
    ):
        if not ABloaders:
            raise ValueError("ABloaders must contain at least one loader.")

        super().__init__(
            metadata=metadata,
            with_context=True,
            with_metadata=True,
            name=name,
        )

        self.ABloaders = list(ABloaders)

    def __iter__(self):
        """Iterate sequentially over all underlying loaders."""
        for ABloader in self.ABloaders:
            yield from ABloader

    def get_set_names(self):
        """Return the ordered list of underlying dataset names."""
        set_names = []

        for ABloader in self.ABloaders:
            set_name = ABloader.get_setname()

            if set_name is None:
                raise ValueError(
                    "Each aggregated ABLoader must have a valid set name."
                )

            set_names.append(set_name)

        return set_names

    def get_target_arg(self):
        """Return the common target configuration."""
        target_args = [
            ABloader.get_target_arg()
            for ABloader in self.ABloaders
        ]

        reference = target_args[0]

        if not all(target_arg == reference for target_arg in target_args[1:]):
            raise ValueError(
                "All aggregated ABLoaders must expose the same target_arg."
            )

        return reference

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

    def save(self,storing):
        store_ABDataExperiment(storing,ABDataExperiment=self)

    @classmethod
    def load(cls,storing,name):
        ABDataExperiment = get_ABDataExperiment(storing,name)
        return(ABDataExperiment)

    @classmethod
    def from_dict(cls, data: dict):
        """
        Stub method — needs to be implemented in concrete subclass.
        """
        raise NotImplementedError("Use a concrete subclass to load from dict.")
    
    def merge(self, other: "ABDataExperiment") -> None:
        """
        Merge another experiment into the current one.

        Existing train sets are matched by name and their test sets are merged
        without duplicates. New train sets are appended with their test sets.

        Parameters
        ----------
        other : ABDataExperiment
            Experiment to merge.
        """
        current_plan = self.get_experiment_plan()

        for other_train_loader, other_test_loaders in other:
            train_name = other_train_loader.get_setname()

            if train_name in current_plan:
                idx = next(
                    i for i, train_loader in enumerate(self._train_loaders)
                    if train_loader.get_setname() == train_name
                )

                existing_test_names = set(current_plan[train_name])
                for test_loader in other_test_loaders:
                    test_name = test_loader.get_setname()
                    if test_name not in existing_test_names:
                        self._test_loader_sets[idx].append(test_loader)
                        existing_test_names.add(test_name)
            else:
                self._train_loaders.append(other_train_loader)
                self._test_loader_sets.append(list(other_test_loaders))
                current_plan[train_name] = [
                    test_loader.get_setname() for test_loader in other_test_loaders
                ]




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
        

class ABConfiguredDataExperiment(ABDataExperiment):
    """
    Build an ABDataExperiment from a configuration dictionary defining
    training sets and named evaluation sets.

    Expected configuration format
    -----------------------------
    {
        "train_set_1": [
            {
                "name": "test_simple_renamed",
                "set_name": "test_simple",
            },
            {
                "name": "healthy_vs_altered",
                "set_name": ["healthy", "altered"],
            },
        ]
    }
    """

    def __init__(
        self,
        storing,
        experiment_config,
        name="configured_experiment",
    ):
        ABtrainloader_list = []
        ABtestloader_sets_list = []

        for train_set_name, test_configs in experiment_config.items():

            # Load training set
            ABtrainloader = get_ABloader(
                storing=storing,
                set_name=train_set_name,
            )

            if ABtrainloader is None:
                raise ValueError(
                    f"Training ABLoader '{train_set_name}' not found."
                )

            ABtrainloader_list.append(ABtrainloader)

            # Build associated test loaders
            test_loaders = []

            for test_config in test_configs:

                evaluation_name = test_config["name"]
                set_name = test_config["set_name"]

                # Simple ABLoader
                if isinstance(set_name, str):

                    ABloader = get_ABloader(
                        storing=storing,
                        set_name=set_name,
                    )

                    if ABloader is None:
                        raise ValueError(
                            f"ABLoader '{set_name}' not found."
                        )

                    # Rename logical evaluation set
                    ABloader.metadata["name"] = evaluation_name

                # Aggregated ABLoader
                elif isinstance(set_name, (list, tuple)):

                    ABloaders = [
                        get_ABloader(
                            storing=storing,
                            set_name=current_set_name,
                        )
                        for current_set_name in set_name
                    ]

                    missing_sets = [
                        current_set_name
                        for current_set_name, loader
                        in zip(set_name, ABloaders)
                        if loader is None
                    ]

                    if missing_sets:
                        raise ValueError(
                            f"Missing ABLoaders: {missing_sets}"
                        )

                    ABloader = ABLoaderAggregate(
                        ABloaders=ABloaders,
                        name=evaluation_name,
                    )

                else:
                    raise TypeError(
                        "'set_name' must be a string or a list of strings."
                    )

                test_loaders.append(ABloader)

            ABtestloader_sets_list.append(test_loaders)

        super().__init__(
            ABtrainloader_list=ABtrainloader_list,
            ABtestloader_sets_list=ABtestloader_sets_list,
            name=name,
        )

def generate_cross_dataset_config(
    train_root,
    healthy_root,
    perturbation_root,
    cv_suffixes,
    perturbations,
    healthy_alias="healthy",
):
    """
    Generate cross-dataset evaluation configurations.

    Naming convention
    -----------------
    Train:
        <train_root><cv_suffix>

    Healthy reference:
        <healthy_root><cv_suffix>

    Perturbed dataset:
        <perturbation_root><perturbation_variant><cv_suffix>

    Parameters
    ----------
    train_root : str
        Root name of training datasets.

    healthy_root : str
        Root name of healthy reference datasets.

    perturbation_root : str
        Root name of perturbed datasets.

    cv_suffixes : list[str]
        Cross-validation suffixes, e.g.
        ["_set_1", "_set_2", "_set_3"].

    perturbations : dict[str, str]
        Mapping between perturbation aliases and name variants.

        Example:
        {
            "const_low": "_const_low",
            "noise": "_noise",
        }

    healthy_alias : str, default="healthy"
        Alias used in logical evaluation names.

    Returns
    -------
    dict
        Experiment configuration compatible with ABConfiguredDataExperiment.
    """
    config = {}

    for cv_suffix in cv_suffixes:

        train_set = f"{train_root}{cv_suffix}"
        healthy_set = f"{healthy_root}{cv_suffix}"

        config[train_set] = []

        for perturbation_alias, perturbation_variant in perturbations.items():

            perturbed_set = (
                f"{perturbation_root}"
                f"{perturbation_variant}"
                f"{cv_suffix}"
            )

            config[train_set].append(
                {
                    "name": f"{healthy_alias}_vs_{perturbation_alias}{cv_suffix}",
                    "set_name": [
                        healthy_set,
                        perturbed_set,
                    ],
                }
            )

    return config