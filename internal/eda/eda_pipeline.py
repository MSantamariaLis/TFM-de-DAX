import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from .steps import StationarityTest

class ExploratoryDataAnalysis:

    """ Class for performing exploratory data analysis (EDA) on time series data."""

    def __init__(self, 
                df:pd.DataFrame,
                label_col:str,
                date_col:str,
                ids_col:str,
                frequency:str = None,
                df_covariates:pd.DataFrame = None,
                ):
        
        self.df = df
        self.df_covariates = df_covariates

        self.label_col = label_col
        self.date_col = date_col

        self.ids_col = ids_col
        self.frequency = frequency

    def __calculate_frequency(self, df:pd.DataFrame):

        """ Calculate the frequency of the time series data."""

        df[self.date_col] = pd.to_datetime(df[self.date_col], errors = 'coerce')
        sorted_dates = df[self.date_col].sort_values().drop_duplicates()
        
        inferred_frequency = pd.infer_freq(sorted_dates)
        frequency = self.frequency or inferred_frequency

        if frequency is None:
            raise ValueError("Unable to infer frequency. Please provide a valid frequency string.")
        
        return frequency
    
    def __calculate_shape(self, df: pd.DataFrame):

        """Calculate the shape of the DataFrame and other relevant statistics."""

        self.frequency = self.__calculate_frequency(self.df)
        shape = df.shape

        num_timeseries = len(df[self.ids_col].unique()) 
        timeseries_lengths = df.groupby(self.ids_col).size()

        return {
            'frequency': self.frequency,
            'data_shape': shape,
            'number_timeseries': num_timeseries,    
            'min_length': timeseries_lengths.min(),
            'max_length': timeseries_lengths.max(),
            'median_length': timeseries_lengths.median(),
        }

    def __calculate_missing_data(self, df: pd.DataFrame) -> dict:
        
        """Calculate the percentage of missing data for each group in the DataFrame."""

        def missing_data_per_group(group: pd.DataFrame, end_date: pd.Timestamp) -> pd.DataFrame:
            
            """Fill missing dates for an individual group and calculate missing data."""
            
            start_date = group[self.date_col].min()
            full_range = pd.date_range(start=start_date, end=end_date, freq=self.frequency)
            
            group = (
                group.sort_values(self.date_col)
                .drop_duplicates(subset=[self.date_col])
                .set_index(self.date_col)
                .reindex(full_range)
                .reset_index()
                .rename(columns={'index': self.date_col}))
            
            group[self.ids_col] = group[self.ids_col].ffill().bfill()
            return group.infer_objects()

        end_date = df[self.date_col].max()
        df_clean = df.groupby(self.ids_col, group_keys=False).apply(missing_data_per_group, end_date=end_date)
        
        missing_data_dict = {
            group_id: (group.isna().sum().sum() / group.size) * 100
            for group_id, group in df_clean.groupby(self.ids_col)
            if (group.isna().sum().sum() / group.size) * 100 > 0 }

        return missing_data_dict
                    
    def __calculate_missing_dates(self, df: pd.DataFrame) -> dict:
        """Calculate missing dates for each group in the DataFrame."""
        
        def missing_dates_per_group(group: pd.DataFrame, end_date: pd.Timestamp) -> list:

            """Calculate missing dates for an individual group."""
            
            start_date = group[self.date_col].min()
            full_range = pd.date_range(start=start_date, end=end_date, freq=self.frequency)
            
            return pd.Index(full_range).difference(group[self.date_col]).tolist()
        
        end_date = df[self.date_col].max()
        missing_dates_dict = (
            df.groupby(self.ids_col)
            .apply(missing_dates_per_group, end_date=end_date)
            .loc[lambda x: x.str.len() > 0]  # Filter out IDs with no missing dates
            .to_dict()
        )
        
        return missing_dates_dict
    
    def __calculate_stationarity(self, df: pd.DataFrame) -> dict:

        """Calculate stationarity for each timeseries in the DataFrame."""

        stationarity_test = StationarityTest(
                            ids_col = self.ids_col,
                            date_col = self.date_col,
                            label_col = self.label_col,
                            length_filter = 12
        )

        stationarity_results = stationarity_test.get_results(df)
        return stationarity_results
        
    def get_eda_results(self):

        """ Perform EDA and return the results."""

        eda_results = {
            'basic_info': self.__calculate_shape(self.df),
            'number_of_observations': dict(self.df.groupby(self.ids_col).size()),
            'missing_dates': self.__calculate_missing_dates(self.df),
            'missing_data': self.__calculate_missing_data(self.df),
            'stationarity_test' : self.__calculate_stationarity(self.df)
        }

        return eda_results
    
def run_eda_pipeline(
    df: pd.DataFrame,
    label_col: str,
    date_col: str,
    ids_col: str,
    frequency: str = None,
    df_covariates: pd.DataFrame = None
):
    """
    Runs the Exploratory Data Analysis (EDA) pipeline on the provided time series data.

    This function initializes the EDA process using the main DataFrame and optional covariates.
    It computes basic statistics, missing data and dates, and stationarity tests for each time series.

    Args:
        df (pd.DataFrame): Main input DataFrame containing the time series data.
        label_col (str): Name of the column containing the target variable.
        date_col (str): Name of the column containing date information.
        ids_col (str): Name of the column identifying unique time series.
        frequency (str, optional): Frequency string for the time series (e.g., 'D', 'MS'). If None, it will be inferred.
        df_covariates (pd.DataFrame, optional): DataFrame containing covariate features. Defaults to None.

    Returns:
        dict: Dictionary containing EDA results, including basic info, number of observations per series,
              missing dates, missing data percentages, and stationarity test results.

    """
    eda = ExploratoryDataAnalysis(df = df, 
                                  df_covariates = df_covariates,
                                  label_col = label_col, 
                                  date_col = date_col, 
                                  ids_col = ids_col,
                                  frequency = frequency)
    
    eda_results = eda.get_eda_results()
    return eda_results