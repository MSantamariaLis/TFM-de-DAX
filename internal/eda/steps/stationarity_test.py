# Test de estacionariedad para series temporales: Augmented Dickey-Fuller y
# Kwiatkowski-Phillips-Schmidt-Shin. Lleva a cabo dos contrastes de hipótesis para cada una de las
# series especificadas: (ADF) en el que la hipótesis nula es que la serie NO es estacionaria (KPS)
# en el que la hipótesis nula es que la series SÍ es estacionaria. Devuelve una tabla con los
# resultados de los tests y una lista con las series que son no-estacionarias para al menos uno de
# los tests.

from statsmodels.tsa.stattools import adfuller, kpss
import pandas as pd

class StationarityTest:

    def __init__(self, ids_col, date_col, label_col, **kwargs):

        self.ids_col = ids_col
        self.date_col = date_col

        self.label_col = label_col
        self.stationarity_results = {}

        self.constant_ids_ts = []
        self.short_ids_ts = []

        self.kwargs = kwargs

    def drop_constant_timeseries(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Drops constant time series from the dataset.
        """
        is_constant = df.groupby(self.ids_col)[self.label_col].nunique() == 1
        self.constant_ids_ts = is_constant[is_constant].index.tolist()

        df_filtered = df[~df[self.ids_col].isin(self.constant_ids_ts)].reset_index(drop=True)
        return df_filtered
    
    def drop_short_timeseries(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Drops time series that have fewer than X time stamps, as required by the AD Fuller test.
        """
        total_length_filter = self.kwargs.get('length_filter', 12)
        result = df.groupby(self.ids_col).filter(lambda x: len(x) < total_length_filter)

        self.short_ids_ts = list(result[self.ids_col].unique())
        df_reduced = df[~df[self.ids_col].isin(self.short_ids_ts)]

        return df_reduced
    
    def kpss_test(self, series: pd.Series):

        """
        Perform KPSS test on a series. Returns the p-value.
        """

        kpss_result = kpss(series, regression='c')
        return kpss_result[1]

    def adfuller_test(self, series: pd.Series):
        
        """
        Perform the ADFuller test on a series and return the p-value.
        """

        kpss_result = adfuller(series, regression='c')
        return kpss_result[1]  # Return the p-value
  
    def get_results(self, df:pd.DataFrame):

        stationary = []
        non_stationary = []

        df = self.drop_short_timeseries(df)
        df = self.drop_constant_timeseries(df)

        for unique_id, df_grouped in df.groupby(self.ids_col):
            df_individual = df_grouped[self.label_col]

            kpss_pvalue = self.kpss_test(df_individual)
            adfuller_pvalue = self.adfuller_test(df_individual)

            if kpss_pvalue > 0.05 and adfuller_pvalue < 0.05:
                stationary.append(unique_id)
                
            else:
                non_stationary.append(unique_id)

        self.stationarity_results["Stationary"] = stationary
        self.stationarity_results["Non-Stationary"] = non_stationary

        self.stationarity_results['NA'] = {
            'constant_ids': self.constant_ids_ts,
            'short_unique_ids': self.short_ids_ts
        }

        return self.stationarity_results