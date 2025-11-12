from abench.metric.metric import ABMetricGeneric
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Exemple of Agnostic benchmark used for regression task 
##################################################################################################

# Use ABMetricGeneric Wrapper to build ABRegressionMetric

class MetricsMse(ABMetricGeneric):
    def __init__(self):
        super().__init__(metric=mean_squared_error, name="mse")

class MetricsMae(ABMetricGeneric):
    def __init__(self):
        super().__init__(metric=mean_absolute_error, name="mae")
        
##################################################################################################
