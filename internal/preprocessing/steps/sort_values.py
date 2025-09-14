import pandas as pd

class SortValues:

    def __init__(self, ids_col, date_col, label_col, **kwargs):

        self.ids_col = ids_col
        self.date_col = date_col

        self.label_col = label_col

    def __sort_values(self, df:pd.DataFrame):
        """Helper method that sorts values from dataset. """
        return df.sort_values(by=[self.ids_col, self.date_col])
    
    def fit_transform(self, df):
        return self.__sort_values(df)

    def transform(self, df_predict:pd.DataFrame):
        return self.__sort_values(df_predict)

    def fit_transform_covariates(self, df_covariates:pd.DataFrame):
        return self.__sort_values(df_covariates)
        
    def transform_covariates(self, df_covariates_future: pd.DataFrame):
        return self.__sort_values(df_covariates_future)
