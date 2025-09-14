from mlforecast import MLForecast
from ..default_arguments import ml_arguments
import pandas as pd
from mlforecast.target_transforms import Differences
from abc import ABC, abstractmethod

class BaseMLModel(ABC):

    """ Base class for ML models """

    def __init__(self, 
                 model_name:str = None,
                 custom_params:dict = None,
                 ids_col: str  = 'unique_id',
                 date_col: str  = 'ds',
                 label_col:str = 'y'           
    ):
        
        self.model_name = model_name
        self.custom_params = custom_params or {}

        self.ids_col = ids_col
        self.date_col = date_col
        
        self.label_col = label_col
        self.model = None

        self.fitted_models = {}
        model_arguments = ml_arguments[self.model_name].copy()
        
        model_arguments.update(custom_params)
        self.model_arguments = self.__process_model_arguments(model_arguments)

        self.covariate_cols = []
        self.static_features = []

        self.df_merged = pd.DataFrame()

    def __process_model_arguments(self, model_arguments:dict):   

        """ 
        Process model arguments, set attributes for multivariate, differencing, and window sizes.
        Remove timeseries keys not needed by the ML model.
        """

        self.is_multivariate = model_arguments.get('multivariate', True)
        self.differencing = 1 if model_arguments.get('differencing', False) else 0

        self.direct_strategy = model_arguments.get('direct_strategy', False)

        window_size = model_arguments.get("window_size")
        window_sizes = window_size if isinstance(window_size, list) else [window_size]

        self.min_window_size = min(window_sizes)
        self.max_window_size = max(window_sizes)
        
        self.window_size = max(window_sizes)
        keys_to_remove = {"window_size", "min_window_size", "max_window_size", "differencing", "multivariate", "direct_strategy"}
        
        model_args = {key: value for key, value in model_arguments.items() if key not in keys_to_remove}
        return model_args

    def __get_covariates_for_id(self, unique_id:str, df_covariates:pd.DataFrame):    

        """ Helper method that generates covariates for a specific unique_id. """

        if df_covariates is None: return None

        cols_to_include = [self.date_col, self.ids_col] + self.covariate_cols
        filtered_df = df_covariates[df_covariates[self.ids_col] == unique_id] 

        future_covariates = filtered_df[cols_to_include]
        return future_covariates
    
    def __calculate_static_features(self, df_covariates:pd.DataFrame):

        """ Helper method that identifies features that are static for all `unique_id` groups. """

        cols_to_check = [col for col in df_covariates.columns if col not in [self.date_col,self.ids_col,self.label_col]]
        static_features = []
        
        for col in cols_to_check:
            is_static_per_group = df_covariates.groupby(self.ids_col)[col].nunique() == 1
            if is_static_per_group.all():
                static_features.append(col)
        
        return static_features
    
    def __process_static_features(self, df_future_covariates:pd.DataFrame):

        """ 
        Helper method that processes static features in the future covariates dataframe.
        They are removed from the dataframe since they are dealt by the MLForecast class. 
        """

        if self.static_features: 

            df_future_covariates = df_future_covariates.drop(columns=self.static_features, errors='ignore')
            if sorted(df_future_covariates.columns.tolist()) == sorted([self.ids_col, self.date_col]):
                return None

        return df_future_covariates
    
    def update_params(self, **new_params:dict):

        """Helper method that updates params for a given model, mainly used for hyperparameter optimization. """

        self.model_arguments.update(new_params)
        self.window_size = self.model_arguments.pop('window_size')
        
        differencing = self.model_arguments.pop('differencing', None)
        self.differencing = 1 if differencing else 0

        self.is_multivariate = self.model_arguments.pop('multivariate', True)

    @abstractmethod
    def optuna_params(self, **params):
        """This method is implemented in the subclasses"""
        pass

    def fit_univariate(self, 
                       df: pd.DataFrame, 
                       df_covariates: pd.DataFrame = None, 
                       horizon:int = None
    ):

        """ Fits a separate MLForecast model for each unique_id in the data and stores them in self.fitted_models."""

        df_merged = pd.merge(df, df_covariates, on = [self.ids_col, self.date_col]) if df_covariates is not None else df
        self.static_features = self.__calculate_static_features(df_merged)
        self.covariate_cols = [col for col in df_merged.columns if col not in [self.date_col, self.ids_col, self.label_col]]

        for unique_id, group in df_merged.groupby(self.ids_col):

            model = MLForecast(
                models = [self.model(**self.model_arguments)], 
                freq = pd.infer_freq(group[self.date_col].sort_values().unique()), 
                lags = [num for num in range(1, self.window_size + 1)],
                target_transforms = [Differences([self.differencing])],
            )
        
            self.fitted_models[unique_id] = model.fit(
                group, 
                static_features = self.static_features, 
                id_col = self.ids_col, 
                time_col = self.date_col, 
                target_col = self.label_col,
                max_horizon= horizon if self.direct_strategy else None
            )


    def fit_multivariate(self, 
                            df: pd.DataFrame,
                        df_covariates: pd.DataFrame = None,
                        horizon:int = None
    ):
    
        """ Fits the ML model and stores it in the mlf instance. """
    
        self.df_merged = pd.merge(df, df_covariates, on = [self.ids_col, self.date_col]) if df_covariates is not None else df
        self.static_features = self.__calculate_static_features(self.df_merged) 
        
        self.mlf = MLForecast(
            models = [self.model(**self.model_arguments)], 
            freq = pd.infer_freq(self.df_merged[self.date_col].sort_values().unique()), 
            lags = [num for num in range(1, self.window_size + 1)],
            target_transforms = [Differences([self.differencing])]
        )

        self.mlf.fit(self.df_merged, 
                     static_features = self.static_features, 
                     id_col = self.ids_col, 
                     time_col=self.date_col,
                     target_col=self.label_col,
                     max_horizon= horizon if self.direct_strategy else None
        )

    def predict_univariate(self, 
                           horizon:int,
                           df_future_covariates:pd.DataFrame = None
    ):

        """ Predicts with each of the individual models stored in fitted_models. """

        global_predictions = pd.DataFrame()
        df_future_covariates = self.__process_static_features(df_future_covariates)

        for unique_id, model in self.fitted_models.items():
            df_individual_future_covariates = self.__get_covariates_for_id(unique_id, df_future_covariates)
            predictions = model.predict(horizon, X_df = df_individual_future_covariates)
            global_predictions = pd.concat([global_predictions, predictions], axis=0)

        return global_predictions
    
    def predict_multivariate(self, 
                             horizon:int, 
                             df_future_covariates:pd.DataFrame = None
    ):

        """ Predicts with the model stored in the self.mlf instance """

        processed_covariates = self.__process_static_features(df_future_covariates)
        return self.mlf.predict(horizon, X_df=processed_covariates)

    
    def forward_univariate(self, 
                           df_predict:pd.DataFrame, 
                           horizon: pd.DataFrame, 
                           df_past_covariates:pd.DataFrame = None, 
                           df_future_covariates:pd.DataFrame = None
    ):

        """ Generates inference predictions based on the models stored in the fitted_models dict """

        global_predictions = pd.DataFrame()

        for unique_id, model in self.fitted_models.items():
            past_covariates = self.__get_covariates_for_id(unique_id, df_past_covariates)
            future_covariates = self.__get_covariates_for_id(unique_id, df_future_covariates)

            df_predict_reduced = self.__get_covariates_for_id(unique_id, df_predict)
            df_predict_merged =  pd.merge(df_predict_reduced, past_covariates, on = [self.ids_col, self.date_col]) if df_past_covariates is not None else df_predict
           
            predictions = model.predict(h = horizon, new_df=df_predict_merged, X_df=future_covariates)
            global_predictions = pd.concat([global_predictions, predictions], axis = 0)

        return global_predictions
    
    def forward_multivariate(self, 
                             df_predict:pd.DataFrame, 
                             horizon: pd.DataFrame, 
                             df_past_covariates:pd.DataFrame = None, 
                             df_future_covariates:pd.DataFrame = None
    ):

        """ Generates inference predictions based on the model stored in the self.mlf instance. """

        predict_merged = (
            pd.merge(df_predict, df_past_covariates, on=[self.ids_col, self.date_col])
            if df_past_covariates is not None else df_predict
        )
    
        return self.mlf.predict(h=horizon, new_df=predict_merged, X_df=df_future_covariates)
    
    def fit(self, df: pd.DataFrame, df_covariates: pd.DataFrame = None, horizon: int = None):
        """ This method delegates fitting to either a multivariate or univariate approach based on the `is_multivariate` attribute"""
        self.fit_multivariate(df, df_covariates, horizon) if self.is_multivariate else self.fit_univariate(df, df_covariates, horizon)

    def predict(self, horizon: int, df_future_covariates: pd.DataFrame = None):
        """ The method delegates prediction based on the model type:"""

        predictions = (
            self.predict_multivariate(horizon, df_future_covariates) 
            if self.is_multivariate 
            else self.predict_univariate(horizon, df_future_covariates)
        )

        predictions = predictions.rename(columns = {self.model.__name__: self.model_name})
        return predictions

    def forward(self, df_predict:pd.DataFrame, horizon: pd.DataFrame, df_past_covariates:pd.DataFrame, df_future_covariates:pd.DataFrame):
        """The method delegates prediction based on the model type"""

        predictions = (
            self.forward_multivariate(df_predict, horizon, df_past_covariates, df_future_covariates) 
            if self.is_multivariate 
            else self.forward_univariate(df_predict, horizon, df_past_covariates, df_future_covariates)
        )

        predictions = predictions.rename(columns = {self.model.__name__: self.model_name})
        return predictions