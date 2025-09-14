import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# Funciones de autocorrelación y autocorrelación parcial, con plots e intervalos de confianza
# (para ambas funciones).
class AutoCorrelation:

    DEFAULTS = {'target_column': 'y', 'alpha': 0.05, 'partial': False, 'season_length': 1}

    def __init__(self, **kwargs): 
        
        self.__dict__.update(self.DEFAULTS)
        self.__dict__.update(kwargs)
    
    def autocorrelation(self, df):

        X = df[self.target_column].to_numpy()

        acf_function = acf if not self.partial else pacf

        X_corr, X_conf = acf_function(X, nlags = min(self.season_length, X.shape[0] // 2 - 1), alpha = self.alpha)

        lags = np.arange(X_corr.shape[0])
        X_corr = np.c_[lags, X_corr]

        df_corr = pd.DataFrame(X_corr, columns = ['lag', self.target_column + '_corr'])
        df_corr['confint_min'] = X_conf[:, 0]
        df_corr['confint_max'] = X_conf[:, 1]

        return df_corr
        
    def autocorrelation_plot(self, df):

        plot_acf_function = plot_acf if not self.partial else plot_pacf

        X = df[self.target_column].to_numpy()
        plot_acf_function(X, lags = min(self.season_length, X.shape[0] // 2 - 1))

        return


# Calcula la correlación entre una (única, aunque puede haber dos o más series) variable objetivo
# ``y`` y los regresores exógenos ``X``. Si hay dos o más series objetivo, las correlaciones se
# calculan como el valor mediano de todas ellas.
class Correlation:

    DEFAULTS = {'time_column_name': 'ds', 'ids_column_name': 'unique_id'}

    def __init__(self, **kwargs): 
        
        self.__dict__.update(self.DEFAULTS)
        self.__dict__.update(kwargs)

    def exogenous_regressors_correlation(self, Y_df, X_df):

        ids_column_name = self.ids_column_name or 'unique_id'

        Y_df = Y_df.set_index(self.time_column_name).unstack().reset_index(name = 'y')
        Y_df = Y_df.rename(columns = {'level_0': ids_column_name})

        Y_df = Y_df.merge(X_df, on = [self.time_column_name] + [ids_column_name] * (ids_column_name in X_df), how = 'inner')

        correlation_function = lambda x: x.corr(numeric_only = True)['y'].drop('y')

        correlation_matrix = Y_df.groupby(ids_column_name).apply(correlation_function)
        correlation_matrix = correlation_matrix.median()

        return correlation_matrix