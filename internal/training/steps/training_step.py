import pandas as pd
from ..utils import MetricsLogger

class TrainTestEvaluatorStep:

    """
    TrainTestStep handles the evaluation and merging of model predictions with test data.

    This class provides methods to calculate evaluation metrics for model predictions
    and to merge predictions with the test set for further analysis.

    Parameters
    ----------
    model_name : str
        Name of the model, used for labeling prediction columns.

    ids_col : str
        Name of the column identifying unique time series.
    
    date_col : str
        Name of the column containing date or time information.
    
    label_col : str
        Name of the column containing the target variable.

    """

    def __init__(self, model_name:str, ids_col:str, date_col:str, label_col:str):

        self.model_name = model_name
        self.metrics_logger = MetricsLogger(model_name, ids_col, date_col, label_col)

        self.ids_col = ids_col
        self.date_col = date_col
        
        self.label_col = label_col
    
    def calculate_test_results(self, predictions:pd.DataFrame, df_test: pd.DataFrame):

        """ Calculates and returns aggregated evaluation metrics for the predictions."""

        fold_results = self.metrics_logger.calculate_fold_metrics(df_test=df_test, predictions=predictions)
        aggregated_results = self.metrics_logger.get_aggregated_results(fold_results)

        return aggregated_results