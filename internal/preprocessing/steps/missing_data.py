import pandas as pd
import numpy as np

class MissingData:

    def __init__(self, ids_col, date_col, label_col, **kwargs):

        self.ids_col = ids_col
        self.date_col = date_col

        self.label_col = label_col
        self.covariate_cols = []

        self.missing_data_method = kwargs.get('missing_data_method', 'backward')

    def __get_filling_method(self, df:pd.DataFrame, method_name, cols: list = []):

        """Helper method to get the filling method for missing data. """
                             
        imputation_methods = {
            'linear':  lambda d: d[cols].interpolate(method='linear').ffill().bfill(),
            'ffill':  lambda d: d[cols].ffill().bfill(),
            'bfill': lambda d: d[cols].bfill().ffill(),
            'zero': lambda d: d[cols].fillna(0)
        }

        if method_name in imputation_methods: 
            df[cols] = imputation_methods[method_name](df)
            return df
        
        else: raise ValueError(f"Method {method_name} not recognized. Use 'linear', 'ffill', 'bfill' or 'zero'.") 

    def group_transform(self, df_grouped:pd.DataFrame):

        df_grouped = self.__get_filling_method(df_grouped, self.missing_data_method, self.label_col)
        return df_grouped
    
    def group_transform_covariates(self, df_grouped_covariates:pd.DataFrame):

        df_grouped_covariates = self.__get_filling_method(df_grouped_covariates, self.missing_data_method, self.covariate_cols)
        return df_grouped_covariates

    def fit_transform(self, df:pd.DataFrame):

        df_clean = (
            df.groupby(self.ids_col, group_keys=False)
            .apply(self.group_transform)
        )

        return df_clean

    def transform(self, df_predict:pd.DataFrame):

        df_predict_clean = (
            df_predict.groupby(self.ids_col, group_keys=False)
            .apply(self.group_transform)
        )

        return df_predict_clean

    def fit_transform_covariates(self, df_covariates:pd.DataFrame):

        """Here is not required to add self.static_features as in the other preprocessing classes. """

        self.covariate_cols = [
            col for col in df_covariates.columns if col not in [self.date_col, self.ids_col]
        ]

        df_covariates_clean = (
            df_covariates.groupby(self.ids_col, group_keys=False)
            .apply(self.group_transform_covariates)
        )

        return df_covariates_clean
    
    def transform_covariates(self, df_covariates_future:pd.DataFrame):
          
        df_covariates_future_clean  = (
            df_covariates_future.groupby(self.ids_col, group_keys=False)
            .apply(self.group_transform_covariates)
        )

        return df_covariates_future_clean   