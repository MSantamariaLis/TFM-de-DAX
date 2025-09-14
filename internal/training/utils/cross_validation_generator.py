import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from .train_test_splitter import temporal_split

class TimeCrossValidation:
    
    def __init__(self, n_splits, horizon, ids_col, date_col):
        
        self.n_splits = n_splits
        self.horizon = horizon

        self.ids_col = ids_col
        self.date_col = date_col
        
    def groupwise_time_series_split(self, group:pd.DataFrame):
        
        """
        Perform time series cross-validation splits for a single group.

        This method uses `TimeSeriesSplit` from scikit-learn to generate train-test splits 
        for time series data, ensuring that the temporal order is preserved for each `unique_id` group.

        Parameters
        ----------
        group : pd.DataFrame
            A DataFrame containing time series data for a single `unique_id`. The data must have a 
            time-based index (e.g., a `ds` column or datetime index) and be sorted chronologically.
            
            Example structure:
            ```
            group = pd.DataFrame({
                'unique_id': ['A', 'A', 'A', 'A'],
                'ds': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
                'y': [100, 110, 120, 130]
            })
            ```

        Returns
        -------
        list of tuples
            A list of train-test splits, where each split is represented as a tuple `(train, test)`:
            - `train`: A DataFrame containing the training data for the fold.
            - `test`: A DataFrame containing the testing data for the fold.

        Each fold contains data for the entire `unique_id` group and respects the time series structure, meaning 
        the training set only contains past data, and the test set contains future data.
        
        """
        tscv = TimeSeriesSplit(n_splits= self.n_splits, test_size=self.horizon)
        train_test_indices = [(group.index[train_idx], group.index[test_idx]) for train_idx, test_idx in tscv.split(group)]
        
        splits = [(group.loc[train_idx], group.loc[test_idx]) for train_idx, test_idx in train_test_indices]
        return splits
    
    def train_test_split(self, df_merged:pd.DataFrame):

        df_train_combined, df_test_combined = temporal_split(df=df_merged, horizon=self.horizon, ids_col=self.ids_col)
        return df_train_combined, df_test_combined
    
    def cross_validation_split(self, df_merged: pd.DataFrame):
        
        train_test_splits = df_merged.groupby(self.ids_col).apply(lambda group: self.groupwise_time_series_split(group)).to_dict()
        
        for fold in range(self.n_splits):
            
            df_train_combined = pd.concat([splits[fold][0] for splits in train_test_splits.values()], ignore_index=True)
            df_test_combined = pd.concat([splits[fold][1] for splits in train_test_splits.values()], ignore_index=True)
            
            yield df_train_combined, df_test_combined
    
    def cross_val_generator(self, df:pd.DataFrame, df_covariates: pd.DataFrame = None):
    
        """
        Generate cross-validation train-test splits for a time series dataset.

        Parameters
        ----------
        df : pd.DataFram

            df = pd.DataFrame({
                'unique_id': ['A', 'A', 'A', 'B', 'B', 'B'],
                'ds': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-01', '2024-01-02', '2024-01-03'],
                'y': [100, 110, 120, 200, 210, 220]
            })

        df_covariates : pd.DataFrame, optional
            Additional covariate data with the same structure as `df`, but containing covariate columns instead of the target variable `y`.

            df_covariates = pd.DataFrame({
                'unique_id': ['A', 'A', 'A', 'B', 'B', 'B'],
                'ds': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-01', '2024-01-02', '2024-01-03'],
                'age': [25, 25, 25, 30, 30, 30],
                'salary': [50000, 50000, 50000, 60000, 60000, 60000]
            })

        Returns
        -------
        generator
            A generator yielding train-test splits for each fold. Each split contains:
            - `combined_train`: Training dataset for the current fold.
            - `combined_test`: Testing dataset for the current fold.

        Description
        -----------
        - Merges the main dataset (`df`) with the covariate dataset (`df_covariates`) on `unique_id` and `ds` columns, if provided.
        - Performs time series cross-validation on each `unique_id` group using `self.groupwise_time_series_split`.
        - Aggregates train and test splits across all `unique_id` groups for each fold, producing combined datasets.

        Example
        -------
        After merging `df` and `df_covariates`, the resulting dataset (`df_merged`) will have the following structure:
        ```
        df_merged = pd.DataFrame({
            'unique_id': ['A', 'A', 'A', 'B', 'B', 'B'],
            'ds': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-01', '2024-01-02', '2024-01-03'],
            'y': [100, 110, 120, 200, 210, 220],
            'age': [25, 25, 25, 30, 30, 30],
            'salary': [50000, 50000, 50000, 60000, 60000, 60000]
        })
        ```

        The generator yields train-test splits for `n_splits` folds, where each fold aggregates data from all `unique_id` groups.

        # Combined Train for Fold 1

        df_train = pd.DataFrame({
            'unique_id': ['A', 'A', 'B', 'B'],
            'ds': ['2024-01-01', '2024-01-02', '2024-01-01', '2024-01-02'],
            'y': [100, 110, 200, 210],
            'age': [25, 25, 30, 30],
            'salary': [50000, 50000, 60000, 60000]
        })

        # Combined Test for Fold 1
        
        df_test = pd.DataFrame({
            'unique_id': ['A', 'B'],
            'ds': ['2024-01-03', '2024-01-03'],
            'y': [120, 220],
            'age': [25, 30],
            'salary': [50000, 60000]
        })

    """

        df_merged = pd.merge(df, df_covariates, on=[self.ids_col, self.date_col]) if df_covariates is not None else df

        if self.n_splits == 1:

            df_train_combined, df_test_combined = self.train_test_split(df_merged)
            yield df_train_combined, df_test_combined

        else:

            train_test_splits = self.cross_validation_split(df_merged)
            for df_train_combined, df_test_combined in train_test_splits:
                yield df_train_combined, df_test_combined

def cross_val_generator(df:pd.DataFrame, df_covariates:pd.DataFrame = None,n_splits:int = 1, horizon:int = 1, ids_col = None, date_col = None):

    time_cv = TimeCrossValidation(n_splits= n_splits, horizon=horizon, ids_col=ids_col, date_col=date_col)

    for df_train_merged, df_test_merged in time_cv.cross_val_generator(df, df_covariates):
        yield df_train_merged, df_test_merged           