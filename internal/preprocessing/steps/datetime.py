import pandas as pd

class DateTimeFormatter:

    def __init__(self, ids_col, date_col, label_col, **kwargs):

        self.ids_col = ids_col
        self.date_col = date_col
        self.label_col = label_col

    def __date_time(self, df):
        """Helper method that converts date col to datetime format"""
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        return df

    def fit_transform(self, df):
        return self.__date_time(df)

    def transform(self, df_predict:pd.DataFrame):
        return self.__date_time(df_predict)

    def fit_transform_covariates(self, df_covariates:pd.DataFrame):
        return self.__date_time(df_covariates)
        
    def transform_covariates(self, df_covariates_future: pd.DataFrame):
        return self.__date_time(df_covariates_future)