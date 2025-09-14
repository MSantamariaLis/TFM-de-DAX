import pandas as pd

class ChampionModelStep:

    """ 
    ChampionModelStep selects and optionally refits the best-performing model (the "champion") 
    based on provided test metrics.

    The champion model is determined by finding the model with the lowest value for a specified 
    metric index across all candidate models. Optionally, the champion model can be refitted 
    on the full dataset.

    Parameters
    ----------
    metric : int
        Index of the metric in the metrics dictionary to use for selecting the champion model.
        
    refit_model : bool, optional
        If True, the champion model will be refitted using the provided data. Default is True.

    """
    
    def __init__(self, metric:int, refit_model:bool = True):
        
        self.metric = metric
        self.refit_model = refit_model
                
    def calculate_champion_model(self, mean_test_metrics:dict):

        """ Selects the champion model based on the lowest value of the specified metric. """
        
        champion_model_name = min(mean_test_metrics, key=lambda x: mean_test_metrics[x][self.metric])    
        return champion_model_name
    
    def fit(self, model, df:pd.DataFrame, df_covariates:pd.DataFrame = None, horizon:int = None):

        """ Optionally refits the champion model on the provided data."""

        if self.refit_model: model.fit(df = df, df_covariates = df_covariates, horizon = horizon)
        return model