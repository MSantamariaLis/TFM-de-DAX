import pandas as pd

class DropDuplicates:

    def __init__(self, ids_col, date_col, label_col, **kwargs):

        self.ids_col = ids_col
        self.date_col = date_col

        self.label_col = label_col

    def __drop_duplicates(self, df:pd.DataFrame):
        """Helper method that drop duplicates from dataset. """
        return df.drop_duplicates(subset = [self.ids_col, self.date_col])
    
    def fit_transform(self, df):
        return self.__drop_duplicates(df)

    def transform(self, df_predict:pd.DataFrame):
        return self.__drop_duplicates(df_predict)

    def fit_transform_covariates(self, df_covariates:pd.DataFrame):
        return self.__drop_duplicates(df_covariates)
        
    def transform_covariates(self, df_covariates_future: pd.DataFrame):
        return self.__drop_duplicates(df_covariates_future)

