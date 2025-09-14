import pandas as pd

class NameChanger:

    def __init__(self, ids_col, date_col, label_col,  **kwargs): 
        
        self.ids_col = ids_col
        self.date_col = date_col
        self.label_col = label_col

        self.input_to_standard = {ids_col: 'unique_id', date_col: 'ds', label_col: 'y'}
        self.standard_to_original = {'unique_id': ids_col, 'ds': date_col, 'y': label_col}

        self.partial_input_to_standard = {ids_col: 'unique_id', date_col: 'ds'}
        self.partial_standard_to_original = {'unique_id': ids_col, 'ds': date_col}

    def fit_transform(self, df: pd.DataFrame):
        return df.rename(columns = self.input_to_standard) 
    
    def fit_transform_covariates(self, df_covariates:pd.DataFrame):
        return df_covariates.rename(columns = self.partial_input_to_standard)
    
    def transform(self, df_predict:pd.DataFrame):
        return df_predict.rename(columns = self.partial_input_to_standard)
    
    def transform_covariates(self, df_covariates:pd.DataFrame):
        return df_covariates.rename(columns = self.partial_input_to_standard)

    def inverse_transform(self, forecast: pd.DataFrame):
        return forecast.rename(columns = self.standard_to_original)
      