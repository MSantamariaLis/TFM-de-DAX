from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
import pandas as pd 
import numpy as np

class MetricsLogger:

    """
    A utility class for logging and aggregating evaluation metrics for time series forecasting models.

    This class provides methods to calculate, store, and aggregate metrics such as MAE, MAPE, and RMSE
    for model predictions across multiple folds and unique time series IDs. It is designed to be used
    in cross-validation or repeated evaluation settings, where metrics need to be tracked per fold and per series.

    Parameters
    ----------
    model_name : str
        The name of the model whose predictions are being evaluated.
    
    ids_col : str
        The column name identifying unique time series (e.g., 'unique_id').
    
    date_col : str
        The column name containing date or time information.
    
    label_col : str
        The column name containing the true target values.

    """

    def __init__(self, model_name: str, ids_col:str, date_col:str, label_col:str):  

        self.model_name = model_name
        self.ids_col = ids_col

        self.date_col = date_col
        self.label_col = label_col

        self.overall_fold_results = {}

    def __calculate_results(self, y_test:np.array, y_pred:np.array):
        
        """ Calculates metrics between two arrays: predictions and the test results """
        
        mae = float(round(mean_absolute_error(y_test, y_pred), 3))
        mape = float(round(mean_absolute_percentage_error(y_test, y_pred), 3) * 100)
        rmse = float(round(root_mean_squared_error(y_test, y_pred), 3))
        
        return mae, mape, rmse
    
    def __calculate_group_metrics(self, group:pd.DataFrame):
        
        """Calculate metrics for a given group """

        group = group.reset_index()

        y_pred = group[self.model_name].values
        y_test = group[self.label_col].values

        mae, mape, rmse = self.__calculate_results(y_test=y_test, y_pred = y_pred)
        return {'mae': mae, 'mape': mape, 'rmse': rmse}
    
    def calculate_fold_metrics(self, df_test:pd.DataFrame, predictions:pd.DataFrame):

        """
        Calculates evaluation metrics between the predictions and the test/validation set. 
                
        Returns:
        --------
        dict: A dictionary containing the evaluation metrics for each unique id

        fold_results = {
            'ID-1': {'mae': 43.28, 'mape': 38.5, 'rmse': 49.84 },
            'ID-2': {'mae': 46.97, 'mape': 6.6, 'rmse': 60.86
        }

        """

        df_merged = pd.merge(predictions, df_test, on = [self.ids_col, self.date_col], how='inner')
        fold_results = df_merged.groupby(self.ids_col, group_keys=False).apply(self.__calculate_group_metrics).to_dict()
        
        return fold_results
    
    def save_results(self, fold_results: dict):

        """
        Saves and updates the "overall_fold_results" dictionary with the metrics from the current fold. The method 
        takes the results of a single fold and appends the calculated metrics (MAE, MAPE, RMSE) to the corresponding 
        lists for each unique ID in the `overall_fold_results`. If an entry for a unique ID doesn't exist, 
        it initializes the lists for the metrics.
        
        Updates:
        --------
        self.overall_fold_results (dict): A dictionary where each unique ID has a list of metrics for 
        each fold. The format is:
            
        overall_fold_results = {
                
            'ID-1': {'mae': [43.28, 45.73], 'mape': [38.5, 17.2], 'rmse': [49.84, 54.11]},
            'ID-2': {'mae': [46.97, 114.76], 'mape': [6.6, 15.1], 'rmse': [60.86, 137.11]}
        }

        """

        for unique_id, metrics in fold_results.items():
            self.overall_fold_results.setdefault(unique_id, {'mae': [], 'mape': [], 'rmse': []})
            for metric, value in metrics.items():
                self.overall_fold_results[unique_id][metric].append(value)

    
    def __calculate_mean_results(self, overall_fold_results:dict):
        
        """
        Helper method that computes the mean value fo each metric (MAE, MAPE , RMSE) per unique ID
        from the overall fold results. 
        
        Returns:
        --------
        mean_fold_results (dict): A dictionary where keys are unique IDs and values are dictionaries containing 
        the mean value of each metric, rounded to three decimal places.

        mean_fold_results = {
                'ID-1': {'mae': 44.51, 'mape': 27.85, 'rmse': 51.98},
                'ID-2': {'mae': 80.87, 'mape': 10.85, 'rmse': 98.99}
        }

        """

        return {
            unique_id: {metric: round(np.mean(values), 3) for metric, values in metrics.items()}
            for unique_id, metrics in overall_fold_results.items()
        }
    
    def get_aggregated_results(self, overall_fold_results:dict):
        
        """ 
        Aggregates the metrics across all unique_ids into a new dict (global_results). 
            
        Returns:
        --------
        global_results (dict): A dictionary containing the aggregated metrics, where each metric maps to a list of values 
        across all unique IDs. The format is:
        
        global_results = {
                    'mae': [10, 20, 40, 60, 80, 20],
                    'rmse': [20, 15, 40, 30, 15],
                    'mape': [30, 12, 15, 20, 30]
        }
    
        """
        
        mean_fold_results = self.__calculate_mean_results(overall_fold_results)
      
        global_results = {}
        for id_metrics in mean_fold_results.values():
            for metric, value in id_metrics.items():
                global_results.setdefault(metric, []).append(float(value))

        return global_results
        