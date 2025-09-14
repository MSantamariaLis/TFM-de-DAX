import numpy as np
import pandas as pd

from ..utils import cross_val_generator, MetricsLogger

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

class OptimizerStep:
      
    """
    OptimizerStep finds the optimal combination of hyperparameters for a given model using Optuna.

    This class automates hyperparameter optimization by running multiple trials with Optuna, 
    evaluating each set of parameters using cross-validation. It supports custom metrics, 
    handles covariates, and processes model-specific parameter formats.

    Parameters
    ----------
    model : object
        The model instance to optimize. Must implement `fit`, `predict`, and `optuna_params` methods.
    
    horizon : int
        Forecast horizon for model training and evaluation.
    
    ids_col : str
        Name of the column identifying unique time series.
    
    date_col : str
        Name of the column containing date or time information.
    
    label_col : str
        Name of the column containing the target variable.
    
    n_trials : int, optional
        Number of Optuna trials to run. Default is 60.
    
    n_splits : int, optional
        Number of cross-validation splits. Default is 2.
    
    metric : str, optional
        Metric name to optimize (e.g., 'mae'). Default is 'mae'.

    """

    def __init__(self, model:object, horizon:int, ids_col:str, date_col:str, label_col:str, n_trials:int = 60, n_splits = 2, metric = 'mae'):

        self.n_splits = n_splits
        self.n_trials = n_trials

        self.metric = metric
        self.model = model
        
        self.horizon = horizon
        self.best_params = None

        self.ids_col = ids_col
        self.date_col = date_col

        self.label_col = label_col

    def __process_fold(self, df_merged:pd.DataFrame, covariate_cols: list = []):

        """ Splits a merged DataFrame into the main data and covariates DataFrame."""

        df = df_merged[[self.ids_col, self.date_col, self.label_col]] 
        df_covariates = df_merged[covariate_cols + [self.ids_col, self.date_col]] if covariate_cols else None

        return df, df_covariates
        
    def cross_val_trainer(self, model:object, df:pd.DataFrame, n_splits:int, horizon:int, df_covariates:pd.DataFrame):

        """ Runs cross-validation for the given model and data, logging metrics for each fold. """

        metrics_logger = MetricsLogger(model.model_name, self.ids_col, self.date_col, self.label_col)  
        covariate_cols = [col for col in df_covariates.columns if col not in [self.ids_col, self.date_col, self.label_col]] if df_covariates is not None else []

        for df_train_merged, df_test_merged in cross_val_generator(df, df_covariates, n_splits, horizon, ids_col=self.ids_col, date_col=self.date_col):

            df_train, df_train_covariates = self.__process_fold(df_train_merged, covariate_cols)
            df_test, df_test_covariates = self.__process_fold(df_test_merged, covariate_cols)

            model.fit(df = df_train,df_covariates = df_train_covariates, horizon = horizon)  
            predictions = model.predict(df_future_covariates = df_test_covariates, horizon = horizon)  
        
            fold_results = metrics_logger.calculate_fold_metrics(df_test=df_test, predictions=predictions)
            metrics_logger.save_results(fold_results)
        
        overall_results = metrics_logger.overall_fold_results
        training_results = metrics_logger.get_aggregated_results(overall_results)
        
        return training_results
        
    def evaluate_params(self):

        """ Evaluates the current model parameters using cross-validation."""

        training_results = self.cross_val_trainer(model = self.model, 
                                                df = self.df, 
                                                df_covariates=self.df_covariates,
                                                n_splits=self.n_splits, 
                                                horizon = self.horizon
        )
        
        mean_results = {metric: float(round(np.mean(values), 3)) for metric, values in training_results.items()}
        return mean_results.get(self.metric)

    def calculate_best_params(self, trial: optuna.Trial):

        """ Objective function for Optuna optimization. Updates model parameters and evaluates them."""
        
        optuna_params = self.model.optuna_params(trial)
        self.model.update_params(**optuna_params)

        mean_score = self.evaluate_params()
        return mean_score
    
    def __process_params(self, best_params):

        """ Helper method is used to process some models params. """

        model_param_mapping = {
            'ARIMA': lambda params: {'order': (params.pop('p'), params.pop('d'), params.pop('q')), 'seasonal_order': (params.pop('P'), params.pop('D'), params.pop('Q'))},
            'ETS': lambda params: {'model': (params.pop('error'), params.pop('trend'), params.pop('seasonal'))}
        }

        if self.model.model_name in model_param_mapping: return model_param_mapping[self.model.model_name](best_params)
        else: return best_params
        
    def fit(self, df:pd.DataFrame, df_covariates: pd.DataFrame=None):

        """  Runs the Optuna optimization process on the provided data. """
        
        self.df = df
        self.df_covariates = df_covariates

        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5, n_min_trials = 10)
        study = optuna.create_study(direction="minimize", pruner=pruner)

        study.optimize(self.calculate_best_params, n_trials = self.n_trials, show_progress_bar=True)
        self.best_params = self.__process_params(study.best_params)

    def get_optimized_params(self):

        """ Returns the best parameters found during optimization. """
        
        return self.best_params 