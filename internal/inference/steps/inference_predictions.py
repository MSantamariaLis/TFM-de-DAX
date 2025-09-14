import pandas as pd

class InferencePredictions:

    """
    Handles the prediction step during inference using the provided champion model.

    This class abstracts the logic for generating predictions, supporting both direct prediction
    and forward prediction modes depending on the input data.

    Args:
        champion_model (object): The trained model object used for generating predictions.
    
    """

    def __init__(self, champion_model:object):

        self.champion_model = champion_model
        if not hasattr(self.champion_model, 'predict'): raise ValueError('The chamion model must have a predict method. Set the refit_model parameter to True in the training pipeline to ensure that the model is fitted and has a predict method.')

    def inference_predictions(self, horizon:int, df_predict:pd.DataFrame = None, df_past_covariates:pd.DataFrame = None, df_future_covariates:pd.DataFrame = None):

        """ Generates predictions using the champion model. Uses 'forward' if df_predict is provided, otherwise uses 'predict'. """

        if df_predict is not None:

            predictions = self.champion_model.forward(horizon = horizon,
                                                    df_predict = df_predict,
                                                    df_past_covariates = df_past_covariates,
                                                    df_future_covariates = df_future_covariates)
        else: 
            predictions = self.champion_model.predict(
                                                horizon = horizon,
                                                df_future_covariates = df_future_covariates,
            )
        
        return predictions.rename(columns = {self.champion_model.model_name: 'predictions'})

