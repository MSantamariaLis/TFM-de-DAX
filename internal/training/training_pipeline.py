import pandas as pd
import numpy as np

from .steps import TrainTestEvaluatorStep, OptimizerStep, ChampionModelStep
from .utils import temporal_split

from .models.stats import ARIMAModel, ETSModel, THETAModel
from .models.neuralnets import NBEATSModel, LSTMModel, NHITSModel, RNNModel, KANModel, TFTModel
from .models.ml import LightGBMModel, DecisionTreesModel, RandomForestModel, ExtraTreesModel, LinearRegressionModel, XGBoostModel
from .models.auto_gluon import TomoeModel, ChronosModel


class TrainingPipeline:

    """     
    Manages the end-to-end training process for time series models, including data splitting,
    hyperparameter optimization, model training, evaluation, champion model selection, and error analysis.

    Attributes
    ----------
        df : pd.DataFrame
            The main dataset containing the time series data.

        df_covariates : pd.DataFrame
            Dataset containing covariates (additional features).
        
        label_col : str
            Name of the column containing the target variable.

        date_col : str
            Name of the column containing date or time information.

        ids_col : str
            Name of the column identifying unique time series.

        models : dict
            Dictionary mapping model names to their corresponding model parameters.

        n_splits : int
            Number of splits for cross-validation.

        horizon : int
            Forecasting horizon (number of periods to forecast ahead).

        metric : str
            Evaluation metric used for model comparison (e.g., 'mae', 'rmse').

        use_best_params : bool
            Whether to optimize the hyperparameters of each model.


        refit_model : bool
            Whether to refit the champion model on the full dataset.
    """

    def __init__(self, 
                 df:pd.DataFrame,
                 label_col:str,
                 date_col:str,
                 ids_col:str, 
                 horizon:int,
                 n_splits:int,
                 models:dict,
                 df_covariates:pd.DataFrame = None,
                 metric:str = 'mae',
                 use_best_params:bool = False,
                 refit_model:bool = False,
                 ):
        
        self.df = df
        self.df_covariates = df_covariates  

        self.label_col = label_col
        self.date_col = date_col

        self.ids_col = ids_col
        self.horizon = horizon

        self.n_splits = n_splits
        self.models = models if isinstance(models, dict) else {model: {} for model in models}

        self.metric = metric
        self.use_best_params = use_best_params

        self.refit_model = refit_model

    def __get_model_instance(self, model_name:str, custom_params:dict):

        """ Helper method that r eturns an instance of the specified model with the given parameters. """

        model_dict = {
            'ARIMA': ARIMAModel,
            'ETS': ETSModel,
            'THETA': THETAModel,
            'LightGBM': LightGBMModel,
            'ExtraTrees': ExtraTreesModel,
            'DecisionTrees': DecisionTreesModel,
            'RandomForest': RandomForestModel,
            'LinearRegression': LinearRegressionModel,
            'XGBoost': XGBoostModel,
            'NBEATS': NBEATSModel,
            'LSTM': LSTMModel,
            'NHITS': NHITSModel,
            'RNN': RNNModel,
            'KAN': KANModel,
            'TFT': TFTModel,
            'Tomoe':TomoeModel,
            'Chronos': ChronosModel,
        }

        if model_name in model_dict: return model_dict[model_name](ids_col = self.ids_col, label_col = self.label_col, date_col = self.date_col, custom_params = custom_params)
        else: raise ValueError(f"Model {model_name} is not supported. Supported models are: {list(model_dict.keys())}. ")

    def __process_metrics(self, metrics:dict):

        """ Helper method that returns the mean metrics for each model. """

        mean_metrics = {
                model: {metric: float(round(np.mean(values), 3)) 
                for metric, values in metrics.items()} for model, metrics in metrics.items()
        }
         
        return mean_metrics
    
    def __optimize_single_model(self, model_name:str, custom_params:dict):

        """ 
        Runs hyperparameter optimization for a single model and returns its best parameters.

        This method creates an instance of the specified model, performs hyperparameter tuning
        using the OptimizerStep, and returns the optimal parameters found.

        """

        model = self.__get_model_instance(model_name, custom_params)
        tuner = OptimizerStep(model, 
                              date_col = self.date_col, 
                              label_col = self.label_col, 
                              ids_col = self.ids_col, 
                              horizon = self.horizon, 
                              n_splits = self.n_splits, 
                              metric = self.metric
        )
        
        tuner.fit(df = self.df_train,
                  df_covariates= self.df_train_covariates
        )
        
        best_model_params = tuner.get_optimized_params()
        return best_model_params
    
    def hyperparameter_optimization(self, use_best_params:bool = False):

        """
        Performs hyperparameter optimization for all models in the pipeline using temporal cross-validation.

        If use_best_params is True, this method tunes each model's hyperparameters using temporal cross-validation
        (i.e., time-based splits that respect the order of observations) and returns the best found parameters.
        If False, it returns the default parameters for each model as provided.

        Returns
        -------
        dict
            Dictionary where each key is a model name and each value is a dictionary of parameters,
            which may include optimized values if hyperparameter tuning was performed.

        Example
        -------
        model_params = {
            'RandomForest': {'window_size': 12, 'n_estimators': 100},
            'ARIMA': {'order': (3, 0, 1)}
        }

        """

        optimized_model_params = {
            model_name: self.__optimize_single_model(model_name, custom_params) if use_best_params else custom_params
            for model_name, custom_params in self.models.items()
        }

        return optimized_model_params
    
    def __train_test_single_model(self,model_name:str, model_params:dict):

        """ 
        Helper method that computes the metrics for the test set based on the predictions made 
        by the specified model, using either user-defined or optimized hyperparameters.

        """

        model = self.__get_model_instance(model_name, model_params)
        model.fit(df = self.df_train, df_covariates = self.df_train_covariates, horizon = self.horizon)

        predictions = model.predict(df_future_covariates = self.df_test_covariates, horizon = self.horizon)
        evaluator = TrainTestEvaluatorStep(model_name, self.ids_col, self.date_col, self.label_col) 

        test_metrics = evaluator.calculate_test_results(predictions, self.df_test)
        return predictions, test_metrics

    def train_test_models(self, model_params:dict):

        """
        Trains and evaluates all models in the pipeline using the provided parameters.

        This method iterates through each model, trains it on the training set, and evaluates its performance
        on the test set. It returns a dictionary of evaluation metrics for each model.

        """

        global_test_metrics = {} 
        global_test_predictions = self.df_test.copy()

        for model_name, model_params in model_params.items():

            predictions, test_metrics = self.__train_test_single_model(model_name, model_params)
            global_test_metrics[model_name] = test_metrics

            global_test_predictions = global_test_predictions.merge(predictions, 
            on = [self.ids_col, self.date_col], how="outer")
            
        return global_test_metrics, global_test_predictions
    
    def fit_champion_model(self, model_params:dict, mean_test_metrics:dict, refit_model:bool = False):

        """
        Calculates the champion model based on the mean test metrics and fits it to the entire dataset.
        If re_fit_model is True, the champion model is retrained on the entire dataset.
        """

        trainer = ChampionModelStep(self.metric, refit_model=refit_model)
        model_name = trainer.calculate_champion_model(mean_test_metrics)
        
        model = self.__get_model_instance(model_name, custom_params= model_params[model_name])
        champion_model = trainer.fit(
                                    model = model,
                                    df = self.df,
                                    df_covariates = self.df_covariates,
                                    horizon = self.horizon,                                                     
        )
        
        return champion_model if self.refit_model else 'No champion model, set refit to True'
    
    def get_training_results(self):

        """
        Executes the training pipeline and returns the results.

        Steps:
        ------
        1. Splits the data into training and testing sets (both `df` and `df_covariates`).
        2. Calculates model parameters, which may include user-selected optimized parameters.
        3. Computes the metrics for the test set.
        4. Identifies the champion model and re-trains it using the full dataset and covariates.

        """

        self.df_train, self.df_test = temporal_split(self.df, horizon= self.horizon, ids_col=self.ids_col)
        self.df_train_covariates, self.df_test_covariates = temporal_split(self.df_covariates, horizon=self.horizon, ids_col=self.ids_col)
        
        model_params = self.hyperparameter_optimization(self.use_best_params)
        test_metrics, global_test_predictions = self.train_test_models(model_params)

        mean_test_metrics = self.__process_metrics(test_metrics)
        champion_model = self.fit_champion_model(model_params, mean_test_metrics, refit_model = self.refit_model)

        training_metrics = {'mean_test_metrics': mean_test_metrics, 'model_params': model_params, 'test_metrics': test_metrics}
        return champion_model, training_metrics,  global_test_predictions

def run_training_pipeline(df:pd.DataFrame,
                          label_col:str,
                          date_col:str,
                          ids_col:str,
                          models:dict,
                          horizon:int,
                          n_splits:int = 1,
                          metric:str = 'mae',
                          df_covariates:pd.DataFrame = None,
                          use_best_params:bool = False, 
                          refit_model: bool = True
):
    
    """
    Runs the complete training pipeline for time series forecasting models.

    This function orchestrates the end-to-end process of model training, hyperparameter optimization,
    evaluation, and champion model selection for time series forecasting tasks. It supports multiple
    models, cross-validation, and optional covariates.

    Parameters
    ----------
    df : pd.DataFrame
        Main dataset containing the time series data.

    label_col : str
        Name of the column containing the target variable.
    
    date_col : str
        Name of the column containing date or time information.
    
    ids_col : str
        Name of the column identifying unique time series.
    
    models : dict
        Dictionary mapping model names to their corresponding hyperparameters.
    
    horizon : int
        Forecasting horizon (number of periods to forecast ahead).
    
    n_splits : int, optional
        Number of splits for cross-validation (default is 1).
    
    metric : str, optional
        Evaluation metric used for model comparison (default is 'mae').
    
    df_covariates : pd.DataFrame, optional
        Dataset containing additional covariates/features (default is None).
    
    use_best_params : bool, optional
        Whether to perform hyperparameter optimization for each model (default is False).
    
    refit_model : bool, optional
        Whether to refit the champion model on the full dataset (default is True).

    Returns
    -------
    champion_model : object
        The champion model instance, which is the best-performing model based on the evaluation metrics.
    
    training_results : dict
        Dictionary containing mean test metrics, model parameters, and test metrics.
    
    test_predictions : pd.DataFrame
        DataFrame with test set predictions from all evaluated models.
    
    """
    
    training_pipeline = TrainingPipeline(df, 
                                        df_covariates = df_covariates,
                                        label_col = label_col, 
                                        date_col = date_col, 
                                        ids_col = ids_col, 
                                        models = models, 
                                        n_splits = n_splits, 
                                        horizon = horizon, 
                                        metric = metric, 
                                        use_best_params = use_best_params, 
                                        refit_model = refit_model)

    champion_model, training_results, test_predictions = training_pipeline.get_training_results()
    return champion_model, training_results, test_predictions


