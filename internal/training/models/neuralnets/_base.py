from neuralforecast import NeuralForecast
from ..default_arguments import neural_arguments
import pandas as pd


class BaseNeuralModel:

    def __init__(self, 
                 model_name: str = None, 
                 custom_params: dict = None, 
                 ids_col: str = 'unique_id', 
                 date_col: str = 'ds', 
                 label_col:str = 'y'
    ):

        self.model_name = model_name
        self.custom_params = custom_params

        self.ids_col = ids_col
        self.date_col = date_col
        
        self.label_col = label_col
        self.model = None  

        self.fitted_models = {}
        model_arguments = neural_arguments[self.model_name].copy()

        model_arguments.update(custom_params)
        self.model_arguments = self.__process_model_arguments(model_arguments)

    def __process_model_arguments(self, model_arguments:dict):

        window_size = model_arguments.get("window_size")

        window_sizes = window_size if isinstance(window_size, list) else [window_size]
        self.min_window_size = min(window_sizes)

        self.max_window_size = max(window_sizes)
        self.window_size = max(window_sizes)

        model_arguments.update({'window_size': self.window_size})
        model_arguments.setdefault('input_size', model_arguments.pop('window_size', None))
        
        return model_arguments   
    
    def __calculate_static_features(self, df_covariates:pd.DataFrame):

        """Helper method that identifies features that are static for all `unique_id` groups. """

        cols_to_check = [col for col in df_covariates.columns if col not in [self.date_col,self.ids_col,self.label_col]]
        static_features = []
        
        for col in cols_to_check:
            is_static_per_group = df_covariates.groupby(self.ids_col)[col].nunique() == 1
            if is_static_per_group.all():
                static_features.append(col)
        
        return static_features
    
    def __process_static_features(self, df_merged:pd.DataFrame):
         
        if self.static_features:

            df_static_features = df_merged[self.static_features].copy()
            df_merged = df_merged.drop(columns = self.static_features)

            return df_merged, df_static_features.drop(columns = [self.date_col])
         
        return df_merged, None

    def update_params(self, **new_params:dict):

        self.model_arguments.update(new_params)
        self.model_arguments.setdefault('input_size', self.model_arguments.pop('window_size', None))

    def fit(self, df: pd.DataFrame,df_covariates:pd.DataFrame ,horizon: int):

        self.model_arguments.setdefault('h', horizon)
        df_merged = pd.merge(df, df_covariates, on = [self.ids_col, self.date_col]) if df_covariates is not None else df
        
        self.static_features = self.__calculate_static_features(df_merged)
        df_merged, df_satic_features = self.__process_static_features(df_merged)

        self.nlf = NeuralForecast(
                    models = [self.model(**self.model_arguments)], 
                    freq = pd.infer_freq(df_merged[self.date_col].sort_values().unique())               
    )
        
        self.nlf.fit(df = df_merged,
                     static_df= df_satic_features,
                     id_col = self.ids_col, 
                     time_col = self.date_col, 
                     target_col = self.label_col
        )

    def predict(self, horizon:pd.DataFrame = None, df_future_covariates:int = None):

        df_future_covariates, df_static_features = self.__process_static_features(df_future_covariates)
        predictions = self.nlf.predict(futr_df = df_future_covariates, static_df=df_static_features)

        return predictions.reset_index()

    def forward(self, df_predict:pd.DataFrame, horizon: pd.DataFrame, df_past_covariates:pd.DataFrame, df_future_covariates:pd.DataFrame):

        df_merged = pd.merge(df_predict, df_past_covariates, on = [self.ids_col, self.date_col]) if df_past_covariates is not None else df_predict
        df_merged, df_static_features = self.__process_static_features(df_merged)

        predictions = self.nlf.predict(df = df_predict, static_df = df_static_features, futr_df = df_future_covariates)
        return predictions.reset_index()
