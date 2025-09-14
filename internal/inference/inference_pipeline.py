import pandas as pd
from .steps import InferencePreprocessing, InferencePredictions, InferencePostprocessing

class InferencePipeline:

    """
    Orchestrates the inference process for time series or tabular prediction pipelines.

    This class manages the three main steps of inference:
    1. Preprocessing input data using provided preprocessing objects.
    2. Generating predictions using trained models.
    3. Postprocessing the predictions for final output.

    Args:
        preprocessing_objects (dict): Dictionary containing preprocessing artifacts (e.g., scalers, encoders).
        training_objects (dict): Dictionary containing trained model objects and related artifacts.
        horizon (int): Prediction horizon (number of steps to forecast).

    Attributes:
        preprocessing_objects (dict): Stores preprocessing artifacts.
        training_objects (dict): Stores trained model artifacts.
        horizon (int): Number of steps to predict.
        best_model: Reference to the champion model from training_objects.
    
    """

    def __init__(self, 
                 preprocessing_objects:dict, 
                 champion_model:dict, 
                 horizon:int
    ):

        self.preprocessing_objects = preprocessing_objects
        self.horizon = horizon

        self.champion_model = champion_model

    def inference_preprocessing(self, df_predict:pd.DataFrame = None, df_past_covariates:pd.DataFrame = None, df_future_covariates:pd.DataFrame = None):

        """ Preprocesses the input data for inference using the provided preprocessing objects. """

        preprocessing = InferencePreprocessing(self.preprocessing_objects)
        df_predict, df_past_covariates, df_future_covariates = preprocessing.inference_preprocessing(df_predict, df_past_covariates, df_future_covariates)

        return df_predict, df_past_covariates, df_future_covariates
    
    def inference_predictions(self, df_predict:pd.DataFrame = None, df_past_covariates:pd.DataFrame = None, df_future_covariates:pd.DataFrame = None ):

        """ Generates predictions using the best trained model and the provided (preprocessed) data. """

        predict = InferencePredictions(self.champion_model)
        predictions = predict.inference_predictions(self.horizon, df_predict, df_past_covariates, df_future_covariates)

        return predictions
    
    def inference_postpreprocessing(self, predictions: pd.DataFrame):

        """ Postprocesses the predictions to return them to their original scale and format. """

        postprocessing = InferencePostprocessing(self.preprocessing_objects)
        predictions = postprocessing.inference_postprocessing(predictions)

        return predictions
    
def run_inference_pipeline(preprocessing_objects:dict, 
                           champion_model:dict, 
                           horizon:int, 
                           df_predict:pd.DataFrame = None, 
                           df_past_covariates:pd.DataFrame = None,
                           df_future_covariates:pd.DataFrame = None
):
    """
    Executes the complete inference pipeline for time series or tabular prediction tasks.

    This function coordinates the following steps:
        1. Preprocesses input data using provided preprocessing objects.
        2. Generates predictions using trained model artifacts.
        3. Postprocesses predictions to return them to their original scale and format.

    Args:
        preprocessing_objects (dict): Dictionary containing preprocessing artifacts (e.g., scalers, encoders).
        training_objects (dict): Dictionary containing trained model objects and related artifacts.
        horizon (int): Number of steps to forecast (prediction horizon).
        df_predict (pd.DataFrame, optional): DataFrame with data to predict on. Defaults to None.
        df_past_covariates (pd.DataFrame, optional): DataFrame with past covariate features. Defaults to None.
        df_future_covariates (pd.DataFrame, optional): DataFrame with future covariate features. Defaults to None.

    Returns:
        pd.DataFrame: Postprocessed predictions in their original scale and format.
    """

    inference = InferencePipeline(preprocessing_objects, champion_model, horizon)
    df_predict, df_past_covariates, df_future_covariates = inference.inference_preprocessing(df_predict, df_past_covariates, df_future_covariates)

    predictions = inference.inference_predictions(df_predict, df_past_covariates, df_future_covariates)
    predictions = inference.inference_postpreprocessing(predictions)

    return predictions