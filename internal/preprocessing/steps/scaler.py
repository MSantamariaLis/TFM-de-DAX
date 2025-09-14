import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, FunctionTransformer


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, FunctionTransformer


class Scaler:

    """
    A class for applying scaling transformations and inverse transformartions (re-storing) to timeseries data.

    This class supports scaling and inverse scaling of target labels and covariate columns at the group level
    using various scalers such as MinMaxScaler, StandardScaler, and MaxAbsScaler. It maintains per-group scalers
    for consistent transformations across training and test datasets.

    Attributes:
    -----------
    ids_col : str
        Column name for unique identifiers of groups.

    date_col : str
        Column name for date information.

    label_col : str
        Column name for the target label to be scaled.

    scaler_name : str, default='minmax'
        Name of the scaler to be used. Supported options: 'minmax', 'standard', 'maxabs'.


    Methods:
    --------
    get_scaler_instance(scaler_name):
        Returns a scaler instance based on the provided scaler name.

    group_fit_transform(df_grouped, cols_to_scale, store_scaler):
        Applies scaling transformation to a grouped DataFrame and stores the scaler instance.

    group_transform(df_grouped, cols_to_scale, store_scaler):
        Applies scaling transformation to a grouped DataFrame using previously fitted scalers.

    group_inverse_transform(df_grouped, cols_to_scale, store_scaler):
        Applies inverse scaling transformation to a grouped DataFrame using previously fitted scalers.

    fit_transform(df_train):
        Applies scaling transformation to the target label across all groups in training data and stores scalers.

    transform(df_test):
        Applies scaling transformation to the target label across all groups in test data.

    fit_transform_covariates(df_train_covariates):
        Applies scaling transformation to covariate columns across all groups in training data and stores scalers.

    transform_covariates(df_test_covariates):
        Applies scaling transformation to covariate columns across all groups in test data.

    inverse_transform(forecast):
        Applies inverse scaling transformation to forecast data for the target label.
    """

    def __init__(self, ids_col, date_col, label_col, **kwargs):

        self.ids_col = ids_col
        self.date_col = date_col

        self.label_col = label_col
        self.scaler = kwargs.get('scaler')

        self.static_features = kwargs.get('static_features', [])
        self.boolean_features = kwargs.get('boolean_features', [])

        self.scaler_dict = {}
        self.scaler_covariates_dict = {}

        self.predictions_col = 'predictions'
        self.covariate_cols = []

    def get_scaler_instance(self, scaler_name):

        """
        Returns an instance of the scaler. An error is if the name is not supported.
        """

        scaler_dict = {
            'minmax': MinMaxScaler,
            'standard': StandardScaler,
            'maxabs': MaxAbsScaler,
            'log': FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=True, feature_names_out='one-to-one')
        }

        scaler = scaler_dict.get(scaler_name, None)

        if scaler: return scaler() if callable(scaler) else scaler
        else: raise ValueError(f"The method {scaler_name} is not supported. Check for syntax.")

    def group_fit_transform(self, df_grouped: pd.DataFrame, cols_to_scale: list, store_scaler: dict):

        """
        Applies scaling transformation to a grouped DataFrame and stores fitted scaler. -> only to the main data (y)??

        Parameters:
        ------------
        df_grouped: grouped dataframe to scale

        cols_to_scale: either the target variable or covariate columns.

        store_scaler: where the fitted scaler will be saved.  self.scaler_dict,  self.scaler_covariates_dict

        """

        unique_id = df_grouped[self.ids_col].iloc[0]
        scaler = self.get_scaler_instance(self.scaler)

        df_grouped[cols_to_scale] = scaler.fit_transform(df_grouped[cols_to_scale])
        store_scaler[unique_id] = scaler

        return df_grouped

    def group_transform(self, df_grouped: pd.DataFrame, cols_to_scale: list, store_scaler: dict):

        """
        Applies scaling transformation to a grouped dataframe using previously fitted scalers.
        This method can be used to both target labels and covariate data.

        """
        unique_id = df_grouped[self.ids_col].iloc[0]
        scaler = store_scaler[unique_id]

        df_grouped[cols_to_scale] = scaler.transform(df_grouped[cols_to_scale])
        return df_grouped

    def group_inverse_transform(self, df_grouped: pd.DataFrame, predictions_col = None):

        """
        Applies inverse scaling transformation to a grouped forecast.

        """
        unique_id = df_grouped[self.ids_col].iloc[0]
        scaler = self.scaler_dict[unique_id]

        predictions_col = predictions_col or self.predictions_col
        df_grouped[predictions_col] = scaler.inverse_transform(df_grouped[predictions_col].to_frame()).flatten()

        return df_grouped

    def group_inverse_transform_covariates(self, df_grouped: pd.DataFrame, covariate_cols=None):
        """
        Applies inverse scaling transformation to covariate columns for a grouped DataFrame.

        """
        unique_id = df_grouped[self.ids_col].iloc[0]
        scaler = self.scaler_covariates_dict[unique_id]

        cols = covariate_cols or self.covariate_cols
        df_grouped[cols] = scaler.inverse_transform(df_grouped[cols])

        return df_grouped

    def fit_transform(self, df: pd.DataFrame):

        """
        Applies scaling transformation to the target label across all groups in training data
        and stores the fitted scalers.

        """
        df_scaled = (df.groupby(self.ids_col, group_keys=False).apply(
            self.group_fit_transform, cols_to_scale=[self.label_col], store_scaler=self.scaler_dict))

        return df_scaled

    def transform(self, df_predict: pd.DataFrame):

        """
        Applies scaling transformation to the target label across all groups in test data.

        """
        df_predict_scaled  = (df_predict.groupby(self.ids_col, group_keys=False).apply(
            self.group_transform, cols_to_scale=[self.label_col], store_scaler=self.scaler_dict))

        return df_predict_scaled

    def fit_transform_covariates(self, df_covariates: pd.DataFrame):

        """
        Applies scaling transformation to covariate columns across all groups in training data and stores scalers.

        """
        self.covariate_cols = [
            col for col in df_covariates.columns if col not in [self.date_col, self.ids_col]
            + self.static_features + self.boolean_features
        ]

        if self.covariate_cols:

            df_covariates = (df_covariates.groupby(self.ids_col, group_keys=False).apply(
                self.group_fit_transform, cols_to_scale=self.covariate_cols, store_scaler=self.scaler_covariates_dict
            ))

        return df_covariates


    def transform_covariates(self, df_covariates_future: pd.DataFrame, predictions_col: str = None):
        """
        Applies scaling transformation to covariate columns across all groups in test data.

        """
        if self.covariate_cols:

            df_covariates_future = (df_covariates_future.groupby(self.ids_col, group_keys=False).apply(
                self.group_transform, cols_to_scale=self.covariate_cols, store_scaler=self.scaler_covariates_dict
            ))

        return df_covariates_future

    def inverse_transform(self, forecast: pd.DataFrame, predictions_col = None):

        """
        Applies inverse scaling transformation to forecast data.

        """
        forecast = (forecast.groupby(self.ids_col, group_keys=False).apply(
            self.group_inverse_transform, predictions_col=predictions_col))

        return forecast

