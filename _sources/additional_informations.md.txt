1. Context of use of the component

With respect to the [End-to-End Approach](https://catalog.confiance.ai/records/n6ag2-b8q77/files/End-to-end%20methodology%20for%20engineering%20trusted%20AI-based%20systems%20V4.1%201.pdf?download=1), this component can be used in Data Engineering (section C.7):
- As a component that can be integrated into an AI-based system, the value contribution of the component lies in both "model uncertainty quantification" and "model based anomaly detection" in batch or stream processing, for example in a supervision system.
- As an engineering tool, its contribution lies also in both "model based data uncertainty quantification" and "model based anomaly detection", to feed in a data-qualication procedure some statistical analysis of data by both UQ and Anomaly indicator. Such indicators can help to identify outliers, contextual data-representativeness issues, or local abnormal noised subsets, to assess the quality of existing datasets before training Machine Learning models.

2. Prerequisites for using the component

The component is a library designed to be used by a data scientist. It then requires knowing how to design a processing chain using standard ML-tools as scikit-learn library, It also requires expert knowledge to translate an industrial problem into a machine learning dataset. The data preprocessing step is not covered by the library, since it requires expert knowledge of data structure, data semantics, and relevant representation according to the operational issues to solve. Then, more specifically, the data scientist has to :

- Be able to preprocess row data collected into structured data that could be valorized through mathematical processing.

- Be able to provide a well-defined machine learning Regression/Forecast task and with useful explanatory features with :

     - Targets having interest to be monitored for system state characterization

     - Features useful for ML-modeling, which means that provides a rich data representation informing about the system state through contextual information, historical values, and near-observations values.

- Be able to interpret the relevance of results :

     - Through qualitative analysis of results

     - By being able to apply the evaluation guidelines provided to analyze model KPI relevance.

However, the library (Although the use of specialized preprocessing libraries is recommended) can provide some naive processing, and help to execute a predesigned processing chain (see demonstrator use cases).

3. Level of maturity of the component

If we identify three main functionalities (linked to the UQ-KPIs) provided:
- For forecasting with error margins, this is a functionally mature task that has been successfully applied to public and synthetics datasets. 
- For the construction of anomaly scores taking uncertainty into account, this is a functionally mature task that has been successfully applied to synthetic data and private datasets
- For the KPI of model reliability at inference-time based on epistemic uncertainty, this is an upstream exploratory task linked to OOD detection, which presents technical and methodological difficulties and therefore has low functional maturity. However, encouraging results have been achieved on some industrial datasets.





