from statsforecast.models import ETS
from ._base import BaseStatsModel
 
 
class ETSModel(BaseStatsModel):
    
    def __init__(self, model_name = 'ETS', custom_params = None, ids_col: str = 'unique_id', date_col: str = 'ds', label_col = 'y'):
        super().__init__(model_name, custom_params, ids_col, date_col, label_col)

        self.model = ETS # do not change this

    def optuna_params(self, trial):

        error = trial.suggest_categorical('error', ['A', 'M'])
        trend = trial.suggest_categorical('trend', ['N', 'A', 'Ad', 'M', 'Md'])

        seasonal = trial.suggest_categorical('seasonal', ['N', 'A', 'M'])
        season_length = trial.suggest_int('season_length', 3, 12)

        return  {
            'model': (error, trend, seasonal),
            'season_length': season_length
            }
       
    
            

            
            
            
            
