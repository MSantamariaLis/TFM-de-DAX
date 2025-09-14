import pandas as pd

class Difference:

    """
    A class for applying differentiation and inverse differentiation (integration) to
    timeseries data. 

    The class calculates the difference between consecutive data points (first order difference).
    It can be applied to both target labels and covariate columns. The inverse transform (integration)
    will be perfomed to reconstruct the forecast by using the last available value, stored in the last_values_dict. 

    Attributes:
    -----------
        ids_col: str 
            Column name for unique identifiers of groups.

        date_col: str 
            Column name for the date information.

        label_col: str 
            Column name for the target label to apply the difference transformation.

        last_values_dict: dict
            Dictionary to store the last value of the target labels for each group (used for inverse transform).
    
    Methods:
    --------

        group_differencing(df_grouped, col, store_values=None):
            Applies differencing to a grouped DataFrame and optionally stores the last value for inverse transformation.

        group_integration(df_grouped, col):
            Applies inverse differencing (integration) to a grouped DataFrame to reconstruct original values.

        fit_transform(df_train):
            Applies differencing to the target label column across all groups in training data and stores the last values.

        fit_transform_covariates(df_train_covariates):
            Applies differencing to covariate columns across all groups in training data.

        transform(df_test):
            Applies differencing to the target label column across all groups in test data.

        transform_covariates(df_test_covariates):
            Applies differencing to covariate columns across all groups in test data.

        inverse_transform(forecast):
            Applies inverse differencing to reconstruct original values for forecast data.

    """

    def __init__(self, ids_col, date_col, label_col, **kwargs):
      
        self.ids_col = ids_col
        self.date_col = date_col

        self.label_col = label_col
        self.last_values_dict = {}

        self.static_features = kwargs.get('static_features', [])
        self.boolean_features = kwargs.get('boolean_features', [])

        self.predictions_col = 'predictions'

    def group_differencing(self, df_grouped: pd.DataFrame, col: str = 'y'):

        """
        Applies a differencing transformation to a grouped DataFrame of the target data. 
        The last values for each grouped dataset will be saved to perform integration on
        the forecast. 
        """

        unique_id = df_grouped[self.ids_col].iloc[0] 
        self.last_values_dict[unique_id] = df_grouped[col].iloc[-1]  

        df_grouped[col] = df_grouped[col].diff() 
        return df_grouped.dropna() 
    
    def group_differencing_covariates(self, df_grouped: pd.DataFrame, covariate_cols: list):

        """
        Applies a differencing transformation to a grouped DataFrame.

        I think storing the last value at this step is unnecessary. 

        """

        df_grouped[covariate_cols] = df_grouped[covariate_cols].diff() 
        return df_grouped.dropna() 


    def group_integration(self, df_grouped: pd.DataFrame, col: str = 'y'):

        """
        Applies the inverse difference transformation (Integration) to the grouped forecast.
       
        """
        unique_id = df_grouped[self.ids_col].iloc[0]
        last_value = self.last_values_dict[unique_id]  

        df_grouped[col] = last_value.values + df_grouped[col].cumsum()  
        return df_grouped

    def fit_transform(self, df: pd.DataFrame):

        """
        Applies the difference transformation to the target label across all groups and stores the last values
        in the last_values_dict object.

        """

        df_derived = (
            df.groupby(self.ids_col, group_keys=False)
            .apply(self.group_differencing, col=[self.label_col])
        )

        return df_derived

    def fit_transform_covariates(self, df_covariates: pd.DataFrame):

        """
        Applies the difference transformation to the covariate columns across all groups. 

        """

        self.covariate_cols = [
            col for col in df_covariates.columns if col not in [self.ids_col, self.date_col] 
            + self.static_features + self.boolean_features
        ]

        df_covariates_derived = (
            df_covariates.groupby(self.ids_col, group_keys=False)
            .apply(self.group_differencing_covariates, covariate_cols=self.covariate_cols)
        )

        return df_covariates_derived

    def transform(self, df_predict: pd.DataFrame):

        """
        Applies the difference transform to the target label across all groups on the test data. 
    
        """

        df_predict_derived = (
            df_predict.groupby(self.ids_col, group_keys=False)
            .apply(self.group_differencing, col=[self.label_col])
        )

        return df_predict_derived

    def transform_covariates(self, df_covariates_future: pd.DataFrame):

        """
        Applies the difference transformation to the covariate columns across all groups on the test data.

        """
        df_covariates_derived = (
            df_covariates_future.groupby(self.ids_col, group_keys=False)
            .apply(self.group_differencing_covariates, covariate_cols=self.covariate_cols)
        )

        return df_covariates_derived

    def inverse_transform(self, forecast: pd.DataFrame):

        """
        Applies the inverse difference transformation (Integration) across all groups on the forecast.

        """
        df_integrated = (
            forecast.groupby(self.ids_col, group_keys=False)
            .apply(self.group_integration, col= self.predictions_col)
        )

        return df_integrated




