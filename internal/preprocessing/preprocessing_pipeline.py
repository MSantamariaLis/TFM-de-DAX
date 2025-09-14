import pandas as pd
from .steps import (
                Scaler,
                Difference,
                ClearOutliers,
                ButterworthFilter,
                WhiteNoiseTest,
                MissingDates,
                NameChanger,
                MissingData,
                FilterShortTimeSeries, 
                StaticFeaturesEncoder,
                DropDuplicates,
                DateTimeFormatter,
                SortValues
)

from ..utils import temporal_split, temporal_combine
from .utils import (
                temporal_split, 
                temporal_combine, 
                calculate_frequency, 
                calculate_static_features,
                calculate_boolean_features
)   

class PreprocessingPipeline:

    def __init__(self, 
                 label_col:str, 
                 date_col:str, 
                 ids_col:str, 
                 horizon: int,
                 **kwargs):

        self.ids_col = ids_col
        self.date_col = date_col

        self.label_col = label_col
        self.horizon = horizon

        self.kwargs = kwargs
        self.missing_data_method = kwargs.get('missing_data_method', False)
        
        self.frequency = kwargs.get('frequency', None)
        self.scaler = kwargs.get('scaler', None)

        self.difference = kwargs.get('difference', False)
        
        self.minimum_length = kwargs.get('minimum_length', 0)
        self.clear_outliers = kwargs.get('clear_outliers', False)
        
        self.butterworth_filter = kwargs.get('butterworth_filter', False)
        self.white_noise_filter = kwargs.get('white_noise_filter', False)

        self.boolean_features = kwargs.get('boolean_features', [])
        self.preprocessing_objects = {}

    
    def get_preprocessor_instance(self, preprocessor_name:str):

        """ Returns an instance of the specified preprocessor class. """

        preprocessor_dict = {

            'sort_values': SortValues,
            'drop_duplicates':DropDuplicates,
            'datetime':DateTimeFormatter,
            'difference': Difference,
            'scaler': Scaler,
            'clear_outliers': ClearOutliers,
            'butterworth_filter': ButterworthFilter,
            'white_noise_filter': WhiteNoiseTest,
            "missing_dates": MissingDates,
            "missing_data": MissingData,
            "name_changer": NameChanger,
            "length_filter" : FilterShortTimeSeries,
            "static_features": StaticFeaturesEncoder
        }

        if preprocessor_name in preprocessor_dict: return preprocessor_dict[preprocessor_name](ids_col = self.ids_col, 
                                                                                               date_col = self.date_col, 
                                                                                               label_col = self.label_col, 
                                                                                               ** self.kwargs
        )
        
        else: raise NameError(f'The method: {preprocessor_name} is not supported. Check for syntax')

    def apply_fit_step(self, df:pd.DataFrame, df_covariates:pd.DataFrame, preprocessor_name:str, condition:bool = False):
         
        """

        If the condition is met, applies the specified preprocessing step to the provided data and covariates dataset. 
        The preprocessor object is saved for use during inference. The function returns the preprocessed data and covariates.
        
        """

        if condition:

            preprocessor = self.get_preprocessor_instance(preprocessor_name)
            df = preprocessor.fit_transform(df)

            df_covariates = preprocessor.fit_transform_covariates(df_covariates) if df_covariates is not None else None
            self.preprocessing_objects[preprocessor_name] = preprocessor

        return df, df_covariates
    

    def apply_transform_step(self, df:pd.DataFrame, df_covariates:pd.DataFrame, preprocessor_name:str, condition:bool = False):
         
        """

        If the condition is met, applies the specified preprocessing step to the provided data and covariates dataset. 
        The preprocessor object is saved for use during inference. The function returns the preprocessed data and covariates.
        
        """

        if condition:

            preprocessor = self.preprocessing_objects.get(preprocessor_name)
            df = preprocessor.transform(df)
            df_covariates = preprocessor.transform_covariates(df_covariates) if df_covariates is not None else None

        return df, df_covariates
    
    def mandatory_steps(self, df:pd.DataFrame, df_covariates:pd.DataFrame):

        """ Executes mandatory preprocessing steps on the DataFrame and covariates. """

        self.kwargs['frequency'] = calculate_frequency(df, frequency=self.frequency, date_col=self.date_col)
        self.static_features = calculate_static_features(df_covariates, ids_col=self.ids_col, date_col=self.date_col, label_col=self.label_col)
        
        self.kwargs['boolean_features'] = calculate_boolean_features(df_covariates, ids_col=self.ids_col, date_col=self.date_col, label_col=self.label_col)
        df, df_covariates = self.apply_fit_step(df, df_covariates, 'sort_values', condition = True)

        df, df_covariates = self.apply_fit_step(df, df_covariates, 'drop_duplicates', condition = True)
        df, df_covariates = self.apply_fit_step(df, df_covariates, 'datetime', condition = True)

        df, df_covariates = self.apply_fit_step(df, df_covariates, 'missing_dates', condition = True)
        df, df_covariates = self.apply_fit_step(df, df_covariates, 'missing_data', condition = self.missing_data_method)
        df, df_covariates = self.apply_fit_step(df, df_covariates, 'length_filter', condition = self.minimum_length)

        return df, df_covariates
    
    def fit_transform(self, df:pd.DataFrame, df_covariates:pd.DataFrame = None):

        """ Executes the fit_transform steps on the DataFrame and covariates. """
       
        df, df_covariates = self.apply_fit_step(df, df_covariates, 'white_noise_filter', condition = self.white_noise_filter)     
        df, df_covariates = self.apply_fit_step(df, df_covariates, 'difference', condition = self.difference)

        df, df_covariates = self.apply_fit_step(df, df_covariates, 'clear_outliers', condition = self.clear_outliers)
        df, df_covariates = self.apply_fit_step(df, df_covariates, 'butterworth_filter', condition = self.butterworth_filter)

        df, df_covariates = self.apply_fit_step(df, df_covariates, 'static_features', condition = self.static_features)
        df, df_covariates = self.apply_fit_step(df, df_covariates, 'scaler', condition = self.scaler)

        return df, df_covariates, self.preprocessing_objects
    
    def transform(self, df:pd.DataFrame, df_covariates:pd.DataFrame = None):

        """ Executes the transform steps on the DataFrame and covariates. """
         
        df, df_covariates = self.apply_transform_step(df, df_covariates, 'white_noise_filter', condition = self.white_noise_filter)     
        df, df_covariates = self.apply_transform_step(df, df_covariates, 'difference', condition = self.difference)

        df, df_covariates = self.apply_transform_step(df, df_covariates, 'clear_outliers', condition = self.clear_outliers)
        df, df_covariates = self.apply_transform_step(df, df_covariates, 'butterworth_filter', condition = self.butterworth_filter)

        df, df_covariates = self.apply_transform_step(df, df_covariates, 'static_features', condition = self.static_features)
        df, df_covariates = self.apply_transform_step(df, df_covariates, 'scaler', condition = self.scaler)

        return df, df_covariates
    

def run_preprocessing_pipeline(
        df: pd.DataFrame,
        label_col: str,
        date_col: str,
        ids_col: str,
        horizon: int,
        df_covariates: pd.DataFrame = None,
        frequency:str = None,
        scaler: str = None,
        difference: bool = False,
        minimum_length: int = None,
        clear_outliers: bool = False,
        butterworth_filter: bool = False,
        white_noise_filter: bool = False,
        missing_data_method: str = None
):
    """

    Executes the preprocessing pipeline on the input data.

    Steps that applies:

        mandatory_steps: 
        - Sorts the DataFrame by date and ids.
        - Removes duplicate entries
        - Formats the date column to datetime.
        - Handles missing dates
        - Applies missing data handling if specified.

        splits the data into training and testing sets based on the specified horizon.

        steps with fit_transform on train data and train covariates:
        - Applies static feature encoding if static_features are detected.
        - Applies scaling if specified.
        - Applies differencing if specified.
        - Applies outlier removal if specified.
        - Applies Butterworth filter if specified.
        - Applies white noise filtering if specified.

        repeats the steps with transform on test data and test covariates:

    Executes the preprocessing pipeline on the input data.

    Applies a sequence of preprocessing steps to the input DataFrame and optional covariates.
    Steps include sorting, duplicate removal, date formatting, scaling, filtering, differencing,
    outlier removal, missing data handling, and static feature encoding.

    Args:
        df (pd.DataFrame): Main input DataFrame containing the time series data.
        label_col (str): Name of the column containing the target variable.
        date_col (str): Name of the column containing date information.
        ids_col (str): Name of the column identifying unique time series.
        df_covariates (pd.DataFrame, optional): DataFrame containing covariate features. Defaults to None.
        scaler (str, optional): Scaler to use for normalization (e.g., 'standard', 'minmax'). Defaults to None.
        difference (bool, optional): Whether to difference the data for stationarity. Defaults to False.
        minimum_length (int, optional): Minimum length required for each time series. Defaults to None.
        clear_outliers (bool, optional): Whether to remove outliers. Defaults to False.
        butterworth_filter (bool, optional): Whether to apply a Butterworth filter. Defaults to False.
        white_noise_filter (bool, optional): Whether to filter out white noise series. Defaults to False.
        missing_data_method (str, optional): Method to fill missing data (e.g., 'bfill', 'ffill'). Defaults to None.

    Returns:

        - Preprocessed main DataFrame,
        - Preprocessed covariates DataFrame (if provided),
        - Dictionary of fitted preprocessing objects for inference.

    """

    preproc = PreprocessingPipeline(label_col = label_col, 
                                    date_col = date_col, 
                                    ids_col = ids_col, 
                                    horizon=horizon,
                                    scaler = scaler, 
                                    difference = difference, 
                                    clear_outliers = clear_outliers, 
                                    butterworth_filter = butterworth_filter, 
                                    white_noise_filter = white_noise_filter,
                                    minimum_length = minimum_length,
                                    missing_data_method = missing_data_method,
                                    frequency = frequency, 

    )

    df, df_covariates = preproc.mandatory_steps(df, df_covariates)

    df_train, df_test = temporal_split(df, horizon = horizon, ids_col = ids_col)
    df_train_covariates, df_test_covariates = temporal_split(df_covariates, horizon = horizon, ids_col = ids_col) 

    df_train, df_train_covariates, preprocessing_objects = preproc.fit_transform(df_train, df_train_covariates)
    df_test, df_test_covariates = preproc.transform(df_test, df_test_covariates)

    df_processed = temporal_combine(df_train, df_test, ids_col, date_col)
    df_covariates_processed = temporal_combine(df_train_covariates, df_test_covariates, ids_col, date_col)
  
    return df_processed, df_covariates_processed, preprocessing_objects