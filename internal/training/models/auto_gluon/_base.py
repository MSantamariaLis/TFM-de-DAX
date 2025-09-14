import pandas as pd
from autogluon.timeseries import TimeSeriesPredictor
from ..default_arguments import auto_gluon_arguments
import os
from abc import ABC, abstractmethod

class BaseAutoGluonModel(ABC):

    def __init__(self, 
                 model_name: str = None, 
                 custom_params:dict = None, 
                 ids_col: str = 'unique_id', 
                 date_col: str = 'ds', 
                 label_col = 'y'
    ):

        self.model_name = model_name
        self.custom_params = custom_params or {}

        self.ids_col = ids_col
        self.date_col = date_col
        
        self.label_col = label_col
        self.model = None

        model_arguments = auto_gluon_arguments[self.model_name].copy()
        model_arguments.update(custom_params)

        self.model_arguments = model_arguments

    def __rename_columns(self, df:pd.DataFrame):
        """Helper method that renames a df to the required format. """
        return df.rename(columns = {self.ids_col:'item_id', self.date_col:'timestamp', self.label_col: 'target'})
    
    def __revert_column_names(self, df:pd.DataFrame):
        """Helper method that renames a df back to the original format"""
        return df.rename(columns= {'item_id': self.ids_col, 'timestamp': self.date_col, 'mean': self.model_name})

    def update_params(self, **new_params:dict):
        self.model_arguments.update(new_params)
    
    @abstractmethod
    def optuna_params(self, **params):
        """This method is implemented in the subclasses"""
        pass

    def fit(self, df:pd.DataFrame, df_covariates:pd.DataFrame = None, horizon = None):

        df_merged = pd.merge(df, df_covariates, on = [self.ids_col, self.date_col]) if df_covariates is not None else df
        self.df_merged = self.__rename_columns(df_merged)
      
        covariate_cols = [col for col in df_covariates.columns if col not in [self.ids_col, self.date_col]] if df_covariates is not None else []
        save_path = os.path.join(os.path.dirname(__file__), "saved_auto_gluon_models")

        self.model = TimeSeriesPredictor(
                    path=save_path, # folder to save the models
                    prediction_length=horizon,
                    freq = pd.infer_freq(self.df_merged['timestamp'].sort_values().unique()),
                    log_to_file=False,
                    verbosity=0,
                    known_covariates_names = covariate_cols)
        
        self.model.fit(train_data= self.df_merged, presets = self.model_arguments.get('presets'))
        
    def predict(self, df_future_covariates:pd.DataFrame = None, horizon = None):

        df_future_covariates = self.__rename_columns(df_future_covariates) if df_future_covariates is not None else None
        predictions = self.model.predict(data = self.df_merged,
                                         known_covariates= df_future_covariates,
                                         ).reset_index()[['item_id', 'timestamp', 'mean']]
        
        return self.__revert_column_names(predictions)

    def forward(self, df_predict:pd.DataFrame = None, horizon: pd.DataFrame = None, df_past_covariates:pd.DataFrame = None, df_future_covariates:pd.DataFrame = None):

        df_predict_merged = pd.merge(df_predict, df_past_covariates, on = [self.ids_col, self.date_col]) if df_past_covariates is not None else df_predict
        df_predict_merged = self.__rename_columns(df_predict_merged)
       
        df_future_covariates = self.__rename_columns(df_future_covariates)
        predictions = self.model.predict(data = df_predict_merged,
                                         known_covariates= df_future_covariates).reset_index()[['item_id', 'timestamp', 'mean']]
        
        return self.__revert_column_names(predictions)