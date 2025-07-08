
<div align="center">
    <img src="_static/Logo_ConfianceAI.png" width="20%" alt="ConfianceAI Logo" />
    <h1 style="font-size: large; font-weight: bold;">UQMODELS</h1>
</div><div align="center">
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.10-efefef">
    </a>
	<a href="#">
        <img src="https://img.shields.io/badge/Python-3.11-efefef">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/Licence-Apache%202.0-blue">
    </a>
	<a href="_static/pylint/pylint.txt">
        <img src="_static/pylint/pylint.svg" alt="Pylint Score">
    </a>
    <a href="_static/flake8/index.html">
        <img src="_static/flake8/flake8.svg" alt="Flake8 Report">
    </a>
	<a href="_static/coverage/index.html">
        <img src="_static/coverage/coverage.svg" alt="Coverage report">
    </a>
</div>
<br>


`UQMODELS` for time series is a python library that seeks to put into practice Uncertainty Quantification (UQ) based on ML/Deep learning models for the analysis of numerical data (Regression and time series). UQModels is inspired by scikit-learn for the creation of ML datascience processing chains incorporating uncertainty quantification mechanisms.

The main objective of these chains is to provide functionality that enhance confidence in model prediction by computing additional UQ-KPI that give insight about ML-modeling uncertainty throught for example : Predictive intervales/Margin of error that catch irreducible variability around an observation, or even local Model unreliability link to the impact of the lack of observation due to data-representativeness issues.

Therefore, the main functionality of the library concerns the modeling step of data science pipeline, through the implementation of an UQModel wrapper that can combine several implementations of forecasting models, UQestimators, and post-processor to form a mathematical processing chain. Then UQModel library provide :
- Several UQEstimators able to estimate several nature of UQ-measure as statistical measure of ML-uncertainty and optionally make prediction.
- UQ-Processing functions that process UQ-measure and/or prediction and/or observation to produce some usefull UQ-KPI.
- Anom-processing functions that process UQ-measures and predictions and observations into a contextual deviation score based on residual (difference between observation and model) prediction normalised by UQmeasure thank to UQ-processing function.
- UQKPI-Processor that wrap UQ-Processing function in a scikit learn Transformer format (fit/tranform procedure) to provide both UQ or Anom KPI
- UQModel wrapper that combine a whole pipeline as a (fit/predict/score) object that produce both prediction and specified UQ-KPI

Such UQ-Model object can be used for example :
 - In the context of monitoring key variables, to characterize the state of a dynamic system
 - For time series monitoring, by providing forecast augmented by uncertainty KPIs (as error margin or model unreliability score) using predict method or anomaly score using score method.

In addition to  modeling-post-processing pipeline, the library also implements minor functionality (in the form of wrappers) designed to facilitate the formalization of the pre-processing and evaluation step. The library aim is illustrated in the following figure.

<figure style="text-align:center">
<img src="_static/UQModels_for_TS_schema.png" width=4000px/>
</figure>


`UQMODELS` is an initiative under [Confiance.ai](https://www.confiance.ai/), an organization dedicated to fostering transparency, fairness, and trust in the field of artificial intelligence.


<div id='quickstart'/>

## 🚀 Quick Start
To install and use the uqmodels library, it is recommended to create a Python virtual environment. You can do that with virtualenv, as follows:

### Setting environement
```bash
pip install virtualenv
virtualenv -p <path/to/python3.9> myenv
source myenv/bin/activate
```
### Installation
Once your virtual environment is activated, you can install the uqmodels library directly from Pypi by typing :

```bash
pip install uqmodels
```

This command will install the uqmodels package and all required dependencies.

<div id='usage'/>

## Tutorial

Tutorials notebooks are provided here : [examples](https://github.com/IRT-SystemX/uqmodels/tree/main/examples)

<div id='description_main'/>

## 🔀 Description of main object.

UQModel define and manipulate several object in a format close of Scikit learn library.

- UQEstimator are (fit/predict) procedure that follow scikitlearn standards.
  - fit(X,y)
  - predict(X)->pred,UQ
- UQModel are (fit/predict&score) procedure that follow scikitlearn standards.
  - fit(X,y)
  - predict(X) -> pred,list_KPI
  - score(X,y) -> list_KPI

- Processor are (fit/transform) procedure inspired from scikitlearn Transformer but without input/ouput constraint of form.
  - Preprocessor transform data (any form) into processed data (any form)
    - a specific kind of preprocessor that take (any form) and return (X,y) can be set-up witinh and UQModels.
  - Postprocessor transform UQ-modeling output (UQ,type_UQ,pred,y) into KPI (any form)

The chain of a UQModels then breaks down into
 - 1: (optional) A preprocessor that compute features from raw data
 - 2: (optional) A predictor that compute a prediction form features.
 - 3: A UQEstimator, that may compute both prediction, and have to compute UQmeasure.
 - 4: Predict-Post-Processor (optional) that exploit prediction and UQmeasure to compute UQ-KPI
 - 5: Score-Post-Processor (optional) that exploit prediction, UQmeasure and target to compute score-KPI

## 🎮 Usage

The uqmodels library provides several UQEstimators that aim to make UQ-regression, and residu-based anomaly detection. Theses estimators can be used as sklearn or tensorflow model, this mean you need to first import the necessary modules and then initialize and fit the predictor with your own data.

4 notebooks of example are provide on a synthetic mutlivariate time series dataset.

- tutorial_UQModels.ipynb : Shows the basic use of RF-UQ as an UQEstimators to predict target and estimate UQmeasure, then combine it with an UQ-KPIprocessor to estimate confidence interval from the UQmeasre, and then directly encapsulates UQestimator and KPI-Processors for confidence interval, epistemic score and Anomaly-detection into an UQModels object.

- example_UQestimators.ipynb : Shows an loop that execute serverals UQ-Estimators that produce differents kinds of UQmeasure (Variance,Quantile,...) in order to illustrate that UQ-KPI processors can handle several type of UQmeasure.

- example_complex_UQModels.ipynb : Shows that it's possible to run and encapsulate in a UQModel more complex models with 3 probabilistic neural network MLP (mean & variance regression) under 3 UQ-overlay paradigm (MC-Dropout,Deep-ensemble,EDL), and 2 Deep neural newtork (LSTM-ED & Transform-ED) performing mutli-horizon mean & variance regression prediction with MC-Dropout based UQ-overlay.

- illustration_Anom_with_UQ_and_ML_risk_aversion.ipynb : Illustrate on a toy example, how uncertainty decomposition can be used to build an residu based anomaly score considering distinctly uncertainty and ML-risk.


```python

import uqmodels.modelization.ML_estimator.random_forest_UQ as RF_UQ
from uqmodels.UQModel import UQModel
import uqmodels.postprocessing.UQKPI_Processor as UQProc

# UQEstimator instanciation
UQEstimator_initializer = RF_UQ.RF_UQEstimator
UQEstimator_parameters = RF_UQ.get_params_dict()

# We can use UQEstimator as standards estimator :
UQEstimator = UQEstimator_initializer(**UQEstimator_parameters)
# Fit UQEstimator that produce regression and UQmeasure estimation.
UQEstimator.fit(X[train],y[train])
pred,UQ = UQEstimator.predict(X)

# Example of UQMOdels wrapping
import uqmodels.postprocessing.UQKPI_Processor as UQProc

#Specify Predict KPI-processors
list_predict_KPI_processors = [UQProc.UQKPI_Processor(),
                               UQProc.NormalPIs_processor(),
                               UQProc.Epistemicscorelvl_processor()]

#Specify Score KPI-processors
list_score_KPI_processors = [UQProc.Anomscore_processor()]

#Specify UQModels pipeline
RF_UQModel = UQModel(UQEstimator_initializer=UQEstimator_initializer,
                     UQEstimator_parameters=UQEstimator_parameters,
                     preprocessor = None, predictor=None,
                     list_predict_KPI_processors=list_predict_KPI_processors,
                     list_score_KPI_processors=list_score_KPI_processors,
                     random_state=0,name="RF_UQ",)

# Fit both UQEstimator and KPI-Processors
RF_UQModel.fit(X[train],y[train])

# Save fitted UQEstimator and KPI-Processors the RF_UQ at path "./model"
RF_UQModel.save('./model')

# New empty UQMOdel 
RF_UQModel = UQModel()

# Load fitted UQEstimator and fitted KPI-Processors
RF_UQModel.load(path='model/'+name)

# Perform prediction and UQ-KPI processing
pred,(UQ,PIs,Elvl) = RF_UQModel.predict(X)

# Perform score processing from features and targets.
KPI_ANOM = RF_UQModel.score(X,y)

# KPI_ANOM is here an multivariate residu based anomaly score considering UQ 
```

For more practical examples and detailed usage scenarios, please refer to the Examples directory in our [Github repository](https://github.com/IRT-SystemX/uqmodels/tree/main/examples). These examples help you to understand how to use uqmodels in your own projects.

## Structure

UQModel is currently made up of 6 parts:

<div id='structure_root'/>

### Root :  in charge of base object, pipeline and UQModel definition
Root folder of UQModel library contains 2 python sources:

- UQModel: define UQModel class (ML-dataset -> pred & UQ-KPI)
UQmodel is a pipeline of modeling & postprocessing aim to provide prediction and complementary UQKPI. It is composed of a (optional) preprocessor, (optinals) predictor, a UQEstimator (see Modeling), and a list of UQKPI_Processors (see Post-processing).
- Custom_UQModel : Implementation of Custom UQModel from the UQModel class

- processing : define utility object for processing task (data -> data processed) & (Load & Save functionality for data and Processor/estimators).
   - Processor : Class that aim to process in a standards (Fit/Transform procedure) data with agnostic form.
   - Cache_manger : Class that define load/save/chech_if_cache functionality within a Processor (or an Estimator).
   - Data_loader : Agnostic class that take a load_API function to create a data_loader that take query and return data.
   - Pipeline : Class that instanciate a pipeline composed of Data_loader and list of Processor.

<div id='structure_modeling'/>

### Modeling folder : in charge of uncertainty quantification (ML-dataset -> pred & UQmeasure)
- UQEstimator : Define the UQEstimator class aimed to handle in an agnostic way estimator that perform uncertainty quantification, to force mutualization of the valorization of the UQmeasures produced.
  - ML_estimator : folder that contains functionality link to uncertainty quantification with/on machine learning model.
    - basline.py : common implementation of UQ-estimators from scikit estimator.
    - random_forest_UQ.py : modification of sklearn-RF to perform UQ-regression.
  - DL_estimator : folder that contains functionality link to uncertainty quantification with/on deep learning model.
    - Neural_network_UQ : Instanciate common procedure (fit,transform) for differents Neural network UQ-paradigm, and allow to instanciate MLP_MC-Dropout,MLP_Deep-Ensemble MLP-EDL that perform Gaussian probabilist regression (mu,sigma) and respective meta-modeling procedure (MC-Dropout, Deep_Ensemble or EDL)  
    - loss, data_embedding,metalayers,utils : sources of loss, layers, and tools.
    - Lstm_ed,Transformer_ed : Naïve implementation of Encoder-Decoder models for multihorizon forecasting handling MC-Dropout.

<div id='structure_postprocessing'/>

### Post-processing folder : in charge of postprocessing that include UQ & Anomaly residual-based
- UQ_processing.py : Functionality link to UQ processing to intermediate form or to UQ-KPI form
- anomaly_processing.py : Functionality link to Anomaly processing (with UQ) to produce anomaly score.
- UQKPI_Processor : Class that define UQKPI_Processor in charge to tranform modeling output into specifics UQKPI as (Predictive intervals, Local model unreliability score, Anomaly score with UQ).
- Custom_UQKPI_Processor : Custom UQKPI_Processor 

<div id='structure_preprocessing'/>

### Preprocessing folder : in charge of preprocessing (data row -> structured data -> ML-dataset):
- Preprocessing.py : Functionality link to preprocessing or feature engenering
- Preprocessor : Class that define Preprocessor in charge to transform data into preprocessed data.
- structure.py : Class that aim to encapsulate metadata link to data structure.

<div id='structure_evaluation'/>

### Evaluation folder : (in progress) in charge of evaluation step
(Work in progress) : aim to provide metrics and functionality to faciliate evaluation step and to harmonise evaluation step and metrics definition to Abench library (agnostic benchmark purpose)

<div id='structure_visualization'/>

### Visualization folder : (in progress) in charge of visualization
(Work in progress) : Refactoring needed. Contains various visualization tools

<div id='structure_data_generation'/>

### Data_generation folder : (in progress) in charge of provide synthetic dataset for example
(Work in progress) : Refactoring needed. Contains a data_generation process to generate gaussian time dependent time series.




