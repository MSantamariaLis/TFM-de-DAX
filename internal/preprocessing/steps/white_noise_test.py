import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox


class WhiteNoiseTest:

    """
    A class to perform the Ljung-Box test for white noise detection in grouped timeseries data.
    Identifies and removes groups that are classified as white noise.

    
    Attributes:
    -----------
    ids_col : str
        Column name for unique identifiers of groups.

    date_col : str
        Column name for date information.

    label_col : str
        Column name for the target label to be tested for white noise.

    Methods:
    -------

    ljungbox_test(df_grouped):
        Performs the white noise test for a given group of training data
       
    fit_transform(df_train):
        Calculates the noisy unique ids and removes them from the training data. 

    transform(df_test):
        Removes rows corresponding to noisy groups from the test dataset.

    fit_transform_covariates(df_train_covariates):
        Removes rows corresponding to noisy groups from the training covariates dataset.

    transform_covariates(df_test_covariates):
        Removes rows corresponding to noisy groups from the test covariates

    """

    def __init__(self, ids_col, date_col, label_col, **kwargs):
      
        self.ids_col = ids_col
        self.date_col = date_col

        self.label_col = label_col
        self.alpha = 0.05  
        self.noisy_ids = set() 

    def ljungbox_test(self, df_grouped: pd.DataFrame):

        """
        Performs the Ljung-Box test on a grouped dataframe. Returns True if the timeseries
        is classifided as white noise, False otherwise.

        """
        X = df_grouped[self.label_col].values

        try:
           
            p_value = acorr_ljungbox(X, lags=[min(10, len(X) // 5)])['lb_pvalue'].values[0]
            return p_value > self.alpha  
        except:
           
            return False

    def fit_transform(self, df: pd.DataFrame):

        """
        Applies the test on all groups across the training data. Stores the noisy ids in 
        set and removes those ids from the training data. 

        """
        results = (
            df.groupby(self.ids_col, group_keys=False)
            .apply(lambda df_grouped: (df_grouped[self.ids_col].iloc[0], self.ljungbox_test(df_grouped)))
        )
        
        self.noisy_ids = {unique_id for unique_id, is_noisy in results if is_noisy}
        return df[~df[self.ids_col].isin(self.noisy_ids)]

    def transform(self, df_predict: pd.DataFrame):
    
        """
        Removes noisy ids from the test data. 

        """
        return df_predict[~df_predict[self.ids_col].isin(self.noisy_ids)]

    def fit_transform_covariates(self, df_covariates: pd.DataFrame):

        """
        Removes noisy ids from the training covrariates dataset.
        """
        return df_covariates[~df_covariates[self.ids_col].isin(self.noisy_ids)]

    def transform_covariates(self, df_covariates_future: pd.DataFrame):

        """
        Removes noisy ids from the future covariates dataset. 

        """
        return df_covariates_future[~df_covariates_future[self.ids_col].isin(self.noisy_ids)]
 
    
if __name__ == '__main__':

    df_train = pd.read_parquet('examples\Kaggle\df_train.parquet')
    df_test = pd.read_parquet('examples\Kaggle\df_test.parquet')
    date_col, ids_col, label_col = 'date', 'store_item', 'sales'

    test = WhiteNoiseTest(label_col = label_col,date_col = date_col, ids_col = ids_col)

    df_train_clean = test.fit_transform(df_train)
    print(df_train_clean)

           
     




