import pandas as pd

class TrainTestSplit:
    
    def __init__(self, horizon, ids_col):

        self.horizon = horizon
        self.ids_col = ids_col

    def train_test(self, group):

        train = group[:-self.horizon]
        test = group[-self.horizon:]

        return train, test

    def train_test_split(self, df:pd.DataFrame):
        
        split_data = df.groupby(self.ids_col).apply(self.train_test)
        
        train_series = pd.concat([split[0] for split in split_data])
        test_series = pd.concat([split[1] for split in split_data])
        
        return train_series, test_series
    
def temporal_split(df:pd.DataFrame, horizon:int, ids_col:str):

    """     
    Splits a time series DataFrame into train and test sets for each unique time series.

    For each group identified by `ids_col`, the function splits the data such that the last `horizon`
    rows are used as the test set and the remaining rows as the training set. This is useful for
    time series forecasting, ensuring that the test set always contains the most recent observations.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing time series data.

    horizon : int
        The number of most recent observations per group to use as the test set.
    
    ids_col : str
        The column name identifying unique time series (e.g., 'unique_id').

    """
    
    if df is None: return None, None
    
    splitter = TrainTestSplit(horizon, ids_col)    
    df_train, df_test = splitter.train_test_split(df)

    return df_train, df_test