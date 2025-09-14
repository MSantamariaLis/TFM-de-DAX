import pandas as pd
from scipy.signal import butter, sosfilt

class ButterworthFilter:

    """
    A class for applying a Butterworth low-pass filter to timeseries data. The filter is applied
    group wise based on the unique identifier column. It can process both target labels and covariates. 


    Attributes: 
    -----------

        ids_col: str 
            Column name for unique identifiers of groups

        date_col: str 
            Column name for the date information

        label_col: str
            Column name for the target label to filter

    Methods:
    --------

        group_fit_transform(df_grouped, col):
            Applies the Butterworth filter to a grouped DataFrame.

        fit_transform(df_train):
            Applies the Butterworth filter to the target column across all groups. 

        fit_transform_covariates(df_train_covariates):
            Applies the Butterworth filter to covariate across all groups.

    """
    
    def __init__(self, ids_col, date_col, label_col, **kwargs): 
        
        self.ids_col = ids_col
        self.date_col = date_col

        self.label_col = label_col
        self.static_features = kwargs.get('static_features', [])

        self.boolean_features = kwargs.get('boolean_features', [])

    def group_butterworth_filter(self, df_grouped:pd.DataFrame, col: list):

        """
        Applies the Butterworth filter to a grouped dataframe.

        """

        sos = butter(N = 5 , Wn=0.33, btype = 'low', output = 'sos')
        df_grouped[col] = sosfilt(sos, df_grouped[col]) * 100

        return df_grouped

    def fit_transform(self, df: pd.DataFrame):

        """
        Applies the Butterworth filter to the target label across all groups.

        """

        df_cleaned = (
            df.groupby(self.ids_col, group_keys=False)
            .apply(self.group_butterworth_filter, col = [self.label_col])
        )

        return df_cleaned
    
    def fit_transform_covariates(self, df_covariates: pd.DataFrame):

        """
        It calculates the covariate columns and applies the Butterworth filter 
        to the covariated across all groups.

        """

        covariate_cols = [
            col for col in df_covariates.columns 
            if col not in [self.ids_col, self.date_col] 
            + self.static_features + self.boolean_features
            ]
        
        df_cleaned = (
            df_covariates.groupby(self.ids_col, group_keys=False)
            .apply(self.group_butterworth_filter, col = covariate_cols)
        )

        return df_cleaned
