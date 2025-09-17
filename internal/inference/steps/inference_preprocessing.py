import pandas as pd

class InferencePreprocessing:

    """
    Handles the preprocessing of the data for inference, including target, past covariates, and future covariates.

    Attributes:
        preprocessing_objects (dict): Dictionary mapping step names to preprocessor objects.

    """

    def __init__(self, preprocessing_objects:dict):

        self.preprocessing_objects = preprocessing_objects

    def __apply_step(self, df:pd.DataFrame, step_name: str, is_covariate: bool = None):

        """
        Helper method that applies a single preprocessing step to the DataFrame.

        Checks whether the preprocessing object for the given step exists (i.e., was selected during preprocessing).
        If it exists, applies the corresponding transformation; otherwise, returns the DataFrame unchanged.
        
        """

        preprocessor = self.preprocessing_objects.get(step_name)
        if preprocessor: 
        
            df = (
                preprocessor.transform_covariates(df)
                if is_covariate
                else preprocessor.transform(df)
        )    
        return df
    
    def __preprocess(self, df:pd.DataFrame, preprocessing_steps: str, is_covariate: bool = None):

        """
        Helper method that applies a sequence of preprocessing steps to the DataFrame.
        
        For each step name in the preprocessing_steps list, this method calls "__apply_step" to apply the corresponding transformation.
        If the input DataFrame is None, the method returns None immediately.
        The is_covariate flag determines whether to use covariate-specific transformations for each step.
        
        """

        if df is None: return None

        for step in preprocessing_steps:
            df = self.__apply_step(df, step, is_covariate)

        return df
    
    def preprocessing_target(self, df_predict:pd.DataFrame):

        """ Applies a predefined sequence of preprocessing steps to the target DataFrame for inference. """
        
        preprocessing_steps = ['sort_values', 'drop_duplicates', 'datetime', 'white_noise_filter', 'missing_dates', 'missing_data','minimum_length','scaler', 'difference']
        return self.__preprocess(df_predict, preprocessing_steps)
    
    def preprocessing_past_covariates(self, df_past_covariates:pd.DataFrame):

        """ Applies a predefined sequence of preprocessing steps to the past covariates DataFrame for inference. """

        preprocessing_steps = ['sort_values', 'drop_duplicates', 'datetime', 'white_noise_filter','missing_dates', 'missing_data','minimum_length','scaler', 'difference', 'static_features']
        return self.__preprocess(df_past_covariates, preprocessing_steps, is_covariate=True)
    
    def inference_preprocessing_future_covariates(self, df_future_covariates:pd.DataFrame):

        """ Applies a predefined sequence of preprocessing steps to the future covariates DataFrame for inference. """
        
        preprocessing_steps = ['sort_values', 'drop_duplicates', 'datetime', 'white_noise_filter', 'missing_dates','missing_data','minimum_length','scaler', 'difference', 'static_features']
        return self.__preprocess(df_future_covariates, preprocessing_steps, is_covariate=True)
    
    def inference_preprocessing(self, df_predict:pd.DataFrame = None, df_past_covariates:pd.DataFrame = None, df_future_covariates:pd.DataFrame = None):

        """ Applies the preprocessing steps to the target, past covariates, and future covariates DataFrames for inference. """
        
        df_predict = self.preprocessing_target(df_predict)
        df_past_covariates = self.preprocessing_past_covariates(df_past_covariates)
        
        df_future_covariates = self.inference_preprocessing_future_covariates(df_future_covariates)
        return df_predict, df_past_covariates, df_future_covariates