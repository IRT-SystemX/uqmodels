
======================================================

## Introduction

'Abench” is a library under development aiming to facilitate and partially automatize execution of machine learning benchmark using standards pipeline et wrapper preconfigured.


## Brief summary of functionalities

UQmodels a pour objectif de robustifier, standardiser et d'automatiser le process de benchmark de ML-Composants (ML-Models et modules annexes) sur la base d'un pipeline d'exécution agnostique associé à des wrappers exploitant du code spécifique au cas d'usage.

L'objectif est de produire des résultats de comportement de composant (Métriques de performances et de ML-Trustworthy) en exploitant des méta data pour produire des métriques conditionelles, qui incorpores des mesures de variabilité d'expériences, et des schémas d'évaluations prédéfinis.


Fonctionalité minimal :
- DataLoader fournissant les données.
- Data-Experiment basé sur la combinaison de DataLoader pour former des couples(Trainset,Evalsets).
- Apprentissage de composant ayant une API standardisé Fit + Predict
- Stockage de modèle (sur la base de l'API Save) et de sortie de modèle par expérience.
- API d'accès pour l'ensemble des objets stockées (Data/Modèle/Sortie/Métriques).
- Exécution de metrics wraperisé sur des grilles conditionelles 
- Pipeline d'éxécution automatisé du benchmark faisant appelle de manière chainé à l'ensemble de ces fonctionalités.
- Fonction de visualisation d'aggrégation de résultats.


Abench est pensé pour facilité la conception d'un projet ML de benchmarking de ML-Componsant sous la forme d'un folder contenant : 

- Philosophie of code production : 
	- Data Folder : raw data stored as a tree-folder 
	- Src Folder :  Task-Depends.
		- Descriptor.py that produce meta data uppon raw data.
		- Perturbator.py that produce altered data for evaluation purpose. 
		- Dataloader.py that load data for component training and inference
		- Component.py that define the automated pipeline done by preprocessing,ML-models and post processing.
		- Metrics.py that define conditional metrics for evaluation purpose.
	- Benchmark.py (+ .slurm) : Execution of the experiment using Abench Loop.
	- Analyse.iypnb : Notebook for analyse metrics.

	- Results : Folder created by abench to store Dataloader, Modeles, Outputs,

Storage of all experiments ouputs (Model, Model output, Data).


## Details of functionalities

![UQmodels general schema](assets/.png)

1. Dataloader : UQEstimator

2. Component : 

3. Modeling Pipeline : UQModel

5. Evaluation/Visualisationction,

Agnostics benchmark
# Examples (Air Liquide Demand Forecast)

# Quickstart

Agnostic benchmark module implement an benchmark loop that aim to facilitate models comparison :

Agnostics benchmark requiere to specify :
- Model encapsultor paradigme (implicite task specification)
- Metrics compatible with the outputs of model encapsulation
- Visualisation compatible with the outputs of model encapsulation

# Architecture Overview


## Model encapsulators

`abench.benchmark.Component` is the canvas of model wrappers. An object instance can be constructed by submodule.

`abench.benchmark.Component` implements three methods:

* A `init` method that initialise Encaspulated model from the submodules elements
```python
# submodules : components of model to encaspulate
Component(submodules)
```

* A `fit` method that fits the model for a dedicated task
```python
# X_train and y_train are the full training dataset used to train some submodule of the model
# context_train are additional informal that may be used during training by some submodule of the model
# The splitter passed as argument to ConformalPredictor assigns data 
# to the fit and calibration sets based on the provided splitting strategy
Component.fit(X_train, y_train, context_train, **kwarg)
```
* a `predict` method that fits the model for a dedicated task

```python
# X_train and y_train are the full training dataset used to train some submodule of the model
# context_train are additional informal that may be used during training by some submodule of the model
# predict method return a object 'Output' that contain model result
output = Component.predict(X, y, context, **kwarg)
```

```python
# Abstract encapsulator class :

class Component(ABC):
    """Abstract Encapsulated Model class :
    Allow generic manipulation of models"""

    def __init__(self, submodule_1=None, submodule_n=None, **kwarg):
        self.submodule_1 = submodule_1
        self.submodule_n = submodule_n

    def fit(self, X, y, context=None, **kwarg):
        """Fitting procedure

        Args:
            X (array): Inputs
            y (array): Targets
            context (array): Additional information
        """
        pass
        pass

    def predict(self, X, context=None, **kwarg):
        """Predict procedure

        Args:
            X (array): Inputs
             context (array): Contextual complementary information

        Returns:
            output : Encapsulated results format
        """
        output = None
        return output

```

# Metrics encapsulators

`abench.benchmark.Encapsulated_metrics` is the canvas of metrics wrappers. Encapsulated_metrics manipulated output of Component

`abench.benchmark.Component` implements three methods:

* A `init` method that initialise Encaspulated metric paramaters and name
```python
encapsulated_metrics()
```

* A `compute` method that compute metrics using Encaspulated_model output and additional information
```python
#output (array): Model results
#y (array): Targets
#sets (array list): Sub-set (train,test)
#context (array): Additional information
#objective (array) : ground truth for unsupervised task evaluation
metric_performance = Component.predict(X, y, context=none, ojective=none,**kwarg)
```

```python
# Encapsulated metrics class :
class Encapsulated_metrics(ABC):
    """Abstract Encapsulated Metrics class :
    Allow generic manipulation of metrics with output specifyied format"""

    def __init__(self):
        self.name = "metrics"

    def compute(self, output, y, sets, context, **kwarg):
        """Compute metrics

        Args:
            output (array): Model results
            y (array): Targets
            sets (array list): Sub-set (train,test)
            context (array): Additional information
        """
        pass

```

To do visualisation + Split benchmark strategy