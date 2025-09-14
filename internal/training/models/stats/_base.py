import pandas as pd
from ..default_arguments import stats_arguments
from abc import ABC, abstractmethod

class BaseStatsModel(ABC):

    """Base class for statistical models"""
 
    def __init__(self,
                model_name: str = None, 
                custom_params:dict = None, 
                ids_col: str = 'unique_id', 
                date_col: str = 'ds', 
                label_col = 'y'
    ):
        
        self.model_name = model_name
        self.custom_params = custom_params or {}

        self.ids_col = ids_col
        self.date_col = date_col
        
        self.label_col = label_col
        self.model = None

        self.fitted_models = {}
        model_arguments = stats_arguments[self.model_name].copy()

        model_arguments.update(custom_params)
        self.model_arguments = model_arguments

        self.start_date = None
        self.freq = None


    @abstractmethod
    def optuna_params(self, **params):
        """This method is implemented in the subclasses"""
        pass

    def update_params(self, **new_params:dict):
        self.model_arguments.update(new_params)

    def __generate_forecast_range(self, start_date, freq, horizon):

        """
        This helper method calculates a forecast date range based on the given
        starting date, frequency, and horizon (number of periods). The range
        begins one frequency interval after the specified `start_date`.

        Parameters:
        -----------
        start_date : str or datetime-like
            The starting date for the forecast range. This can be a string
            in a recognized date format or a datetime-like object.
        
        freq : str
            The frequency string, e.g., 'D' for daily, 'W' for weekly, 'M' for monthly,
            which defines the time intervals of the forecast.
        
        horizon : int
            The number of periods to include in the forecast range.

        Returns:
        --------
        pandas.DatetimeIndex
            A sequence of dates representing the forecast range, starting
            from one frequency interval after `start_date` and spanning `horizon` periods.
        
        Examples:
        ---------
        >>> self.__generate_forecast_range("2024-12-01", "D", 5)
        DatetimeIndex(['2024-12-02', '2024-12-03', '2024-12-04', '2024-12-05', '2024-12-06'], dtype='datetime64[ns]', freq='D')

        >>> self.__generate_forecast_range("2024-01-01", "M", 3)
        DatetimeIndex(['2024-02-01', '2024-03-01', '2024-04-01'], dtype='datetime64[ns]', freq='M')
        """

        return pd.date_range(
            start=pd.Timestamp(start_date) + pd.tseries.frequencies.to_offset(freq),
            periods=horizon, 
            freq = freq)
    
    def __generate_covariates(self, unique_id: str, df_covariates: pd.DataFrame = None):

        """
        This helper method filters the input DataFrame to isolate covariate values
        associated with a specific `unique_id`. If no DataFrame is provided or no
        matching rows are found, it returns `None`.

        Parameters:
        -----------
        unique_id : str
            The unique identifier for the time series to filter covariates.

        df_covariates : pd.DataFrame, optional
            A DataFrame containing covariates for multiple time series. Must include
            a `unique_id` column and columns specified in `self.covariate_cols`.

        Returns:
        --------
        numpy.ndarray or None
            An array of covariate values for the specified `unique_id`. If the
            input DataFrame is `None` or no matching rows are found, returns `None`.

        Examples:
        ---------
        >>> df_covariates = pd.DataFrame({
        ...     "unique_id": ["id1", "id2", "id1"],
        ...     "cov1": [0.1, 0.2, 0.3],
        ...     "cov2": [1, 2, 3]
        ... })
        >>> self.covariate_cols = ["cov1", "cov2"]
        >>> self.__generate_covariates("id1", df_covariates)
        array([[0.1, 1. ],
            [0.3, 3. ]])

        >>> self.__generate_covariates("id3", df_covariates)

        None
        """

        if df_covariates is None: return None

        filtered_df = df_covariates[df_covariates[self.ids_col] == unique_id]
        future_covariates = filtered_df[self.covariate_cols].values 

        return future_covariates

    
    def fit(self, df:pd.DataFrame, df_covariates:pd.DataFrame = None, horizon= None):
        
        """
        Fits all the models and stores them in a dictionary (`self.fitted_models`) with the 
        unique identifier (`unique_id`) as the key and the corresponding model as the value.

        Parameters:
        -----------

        df : pd.DataFrame
            DataFrame containing the target variable (`y`) with a unique identifier (`unique_id`) 
            and timestamps (`ds`).

        df_covariates : pd.DataFrame, optional
            DataFrame containing covariates with matching `unique_id` and `ds`.

        horizon : int, optional
            Unused in this implementation, just for syntax purposes.

        Calculates:
        -----------
        - **`self.start_date`**: Extracted as the minimum value from the `ds` column in the input 
          `df` DataFrame. This represents the earliest timestamp in the dataset and will be used 
          as a reference for generating predictions.
        - **`self.freq`**: Determined by inferring the frequency of the `ds` column using 
          pandas' `pd.infer_freq(df['ds'])` function. This identifies the regular interval 
          (e.g., daily, monthly) in the dataset and ensures that predictions align with the 
          temporal structure of the data.

        Examples:
        ---------

        The models are stored as follows:

            fitted_models = {
                'ID-1.1': ARIMA(),
                'ID-1.2': ARIMA(),
                ...,
                'ID-N.N': ARIMA()}

        Sample Input:
        -------------
        The `df` DataFrame for the target variable:

            df = pd.DataFrame({
                'unique_id': ['A', 'A', 'A', 'B', 'B', 'B'],
                'ds': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-01', '2024-01-02', '2024-01-03'],
                'y': [100, 110, 120, 200, 210, 220]
            })

        The `df_covariates` DataFrame for additional covariate data:

            df_covariates = pd.DataFrame({
                'unique_id': ['A', 'A', 'A', 'B', 'B', 'B'],
                'ds': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-01', '2024-01-02', '2024-01-03'],
                'age': [25, 25, 25, 30, 30, 30],
                'salary': [50000, 50000, 50000, 60000, 60000, 60000]
            })

        When fitting, the target variable (`y`) is extracted from `group['y']`, and the covariates (`X`) 
        are derived from columns excluding `['unique_id', 'ds', 'y']`. In this example, the covariates are 
        `['age', 'salary']`.

        These extracted values (`self.start_date` and `self.freq`) ensure consistency when generating 
        predictions by aligning the prediction timeline with the original data structure.
        """

        self.start_date = df[self.date_col].max()
        self.freq = pd.infer_freq(df.sort_values(by = [self.date_col, self.ids_col])[self.date_col].unique())

        df_merged = df if df_covariates is None else pd.merge(df, df_covariates, on=[self.date_col, self.ids_col])
        self.covariate_cols = [col for col in df_merged.columns if col not in [self.date_col, self.ids_col, self.label_col]]

        for unique_id, group in df_merged.groupby(self.ids_col):

            covariates = group[self.covariate_cols].values if self.covariate_cols else None
            self.fitted_models[unique_id] = self.model(**self.model_arguments).fit(y = group[self.label_col].values, X = covariates)


    def predict(self, horizon: int, df_future_covariates: pd.DataFrame = None):

        """
        Generates predictions for all fitted models and returns a DataFrame with the results.

        Args:
            horizon: Number of time steps to predict into the future.
            df_future_covariates: DataFrame containing future covariates, with columns `unique_id` and the matching covariates.

        Returns:

            DataFrame containing the predictions for all `unique_id` values with the following columns:
                - `unique_id`: Identifier for each time series.
                - `ds`: Future timestamps for predictions.
                - `predictions`: Predicted values for each timestamp.
        """

        forecast_dates = self.__generate_forecast_range(self.start_date, self.freq, horizon)
        predictions_overall = []
        
        for unique_id, model in self.fitted_models.items():

            future_covariates = self.__generate_covariates(unique_id, df_future_covariates)
            predictions = model.predict(h = horizon, X = future_covariates)['mean']

            predictions_overall.extend([{self.ids_col: unique_id, self.date_col: date, self.model_name: value} for date, value in zip(forecast_dates, predictions)])

        return  pd.DataFrame(predictions_overall).sort_values(by=[self.ids_col, self.date_col])

    def fit_predict(self, df:pd.DataFrame, horizon:int,df_covariates:pd.DataFrame = None,  df_future_covariates:pd.DataFrame = None):
        
        """
        Calls fit() and predict()
        
        """
    
        self.fit(df = df, df_covariates=df_covariates, horizon=horizon)
        return self.predict(horizon = horizon, df_future_covariates=df_future_covariates)
            
    def forward(self, df_predict:pd.DataFrame, horizon: pd.DataFrame, df_past_covariates:pd.DataFrame, df_future_covariates:pd.DataFrame):

        """
        Generates forward predictions based on the provided horizon and input data.

        Args:
            df_predict: DataFrame containing historical data with columns `unique_id` and `y`.
            horizon: Number of time steps to predict into the future.

        Returns:
            DataFrame containing the predictions for all `unique_id` values with the following columns:
                - `unique_id`: Identifier for each time series.
                - `ds`: Future timestamps for predictions.
                - `predictions`: Predicted values for each timestamp.

        """

        start_date = df_predict[self.date_col].max()
        forecast_dates = self.__generate_forecast_range(start_date, self.freq, horizon)

        predictions = []
        for unique_id, model in self.fitted_models.items():
            
            past_covariates = self.__generate_covariates(unique_id, df_past_covariates)
            future_covariates = self.__generate_covariates(unique_id, df_future_covariates)

            historical_data = df_predict[df_predict[self.ids_col] == unique_id][self.label_col].values
            future_predictions = model.forward(h=horizon, y= historical_data, X = past_covariates, X_future = future_covariates)['mean']

            predictions.extend([{self.ids_col: unique_id, self.date_col: date, self.model_name: value} for date, value in zip(forecast_dates, future_predictions)])

        return  pd.DataFrame(predictions).sort_values(by=[self.ids_col, self.date_col])
