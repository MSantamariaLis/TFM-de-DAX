import pandas as pd
from scipy.signal import medfilt


class ClearOutliers:

    """
    A class for removing outliers from timeseries data using a median filter. 
    The filter is applied group-wise based on a unique identifier column and it
    processes both target labels and covariate columns in the dataset. 

    Attributes: 
    -----------
        ids_col: str 
            Column name for unique identifiers of groups

        date_col: str 
            Column name for the date information

        label_col: str 
            Column name for the target label to clean

    Methods:
    --------
        group_fit_transform(df_grouped, cols_to_clean):
            Applies the median filter to a grouped DataFrame for cleaning outliers.

        fit_transform(df_train):
            Applies the median filter to the target label column across all groups.

        fit_transform_covariates(df_train_covariates):
            Applies the median filter to covariate columns across all groups.

    """
    
    def __init__(self, ids_col, date_col, label_col, **kwargs): 

        self.ids_col = ids_col
        self.date_col = date_col

        self.label_col = label_col
        self.static_features = kwargs.get('static_features', [])

        self.boolean_features = kwargs.get('boolean_features', [])

    def group_fit_transform(self, df_grouped: pd.DataFrame, cols_to_clean: list):

        """
        Applies the median filter to a grouped DataFrame.

        """

        if len(df_grouped) < 3:
            return df_grouped  

        df_grouped[cols_to_clean] = df_grouped[cols_to_clean].apply(
            lambda col: medfilt(col, kernel_size=3) 
        )

        return df_grouped

    def fit_transform(self, df: pd.DataFrame):

        """
        Applies the median filter to the target label across all groups.

        """

        df_cleaned = (
            df.groupby(self.ids_col, group_keys=False)
            .apply(self.group_fit_transform, cols_to_clean=[self.label_col])
        )

        return df_cleaned

    def fit_transform_covariates(self, df_covariates: pd.DataFrame):

        """
        It calculates the covariate columns and applies the median filter to the 
        covariates across all groups. 
        
        """
        
        covariate_cols = [
            col for col in df_covariates.columns 
            if col not in [self.date_col, self.ids_col] 
            + self.static_features + self.boolean_features
        ]

        df_covariates_clean = (
            df_covariates.groupby(self.ids_col, group_keys=False)
            .apply(self.group_fit_transform, cols_to_clean=covariate_cols)
        )

        return df_covariates_clean
           
     
        

