import pandas as pd

class FilterShortTimeSeries:

    """
    A class for filtering out short time series from a dataset.

    This class identifies and removes time series that do not meet a minimum length criterion 
    across various datasets (e.g., training data, covariates, future covariates). The filtering 
    is based on a unique identifier for each time series.

    Attributes:
    -----------
        ids_col: str
            Column name for unique identifiers of time series.

        date_col: str
            Column name for the date information.

        label_col: str
            Column name for the target label to analyze.

        minimum_length: int, optional
            Minimum length of a time series to be considered valid. 

        short_unique_ids: list
            List of unique IDs that represent short time series.
    """

    def __init__(self, ids_col, date_col, label_col, **kwargs):
        
        self.ids_col = ids_col
        self.date_col = date_col

        self.label_col = label_col

        self.minimum_length = kwargs.get('minimum_length')
        self.short_unique_ids = []

    def calculate_short_timeseries(self, df:pd.DataFrame):

        """Helper methods that returns a list of the unique_id ids that do not meet the criteria.  """
        if self.ids_col in df.index.names:
            df = df.reset_index()

        result = df.groupby(self.ids_col).filter(lambda x: len(x) < self.minimum_length)
        short_unique_ids = list(result[self.ids_col].unique())

        return short_unique_ids
    
    def fit_transform(self, df:pd.DataFrame):

        """Calculate the short ids and removes them from the dataset. """

        self.short_unique_ids = self.calculate_short_timeseries(df)
        df = df[~df[self.ids_col].isin(self.short_unique_ids)]
        
        return df
    
    def transform(self, df_predict:pd.DataFrame):
        
        """Removes the ids from the dataset. """
        return df_predict[~df_predict[self.ids_col].isin(self.short_unique_ids)]
    
    def fit_transform_covariates(self, df_covariates: pd.DataFrame):
        
        """Removes the ids from the dataset. """
        return df_covariates[~df_covariates[self.ids_col].isin(self.short_unique_ids)]
        
    def transform_future_covariates(self, df_future_covariates: pd.DataFrame):
        
        """Removes the ids from the dataset. """
        return df_future_covariates[~df_future_covariates[self.ids_col].isin(self.short_unique_ids)]
    
    def transform_past_covariates(self, df_past_covariates: pd.DataFrame):
        
        """Removes the ids from the dataset. """
        return df_past_covariates[~df_past_covariates[self.ids_col].isin(self.short_unique_ids)]
