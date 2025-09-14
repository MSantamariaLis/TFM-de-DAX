from statsforecast.models import ARIMA
from ._base import BaseStatsModel

class ARIMAModel(BaseStatsModel):
    
    def __init__(self, model_name = 'ARIMA', custom_params = None, ids_col: str = 'unique_id', date_col: str = 'ds', label_col = 'y' ):
        super().__init__(model_name, custom_params, ids_col, date_col, label_col)

        self.model = ARIMA # do not change this

    def optuna_params(self, trial):

        p = trial.suggest_int('p', 1, 10)
        d = trial.suggest_int('d', 0, 2)

        q = trial.suggest_int('q', 0, 3)
        P = trial.suggest_int('P', 0, 2)

        D = trial.suggest_int('D', 0, 2)  
        Q = trial.suggest_int('Q', 0, 2)  

        seasonal_length = trial.suggest_int('season_length', 3, 12)
        
        return {
            'order': (p, d, q),
            'season_length': seasonal_length,
            'seasonal_order': (P,D,Q),
            }