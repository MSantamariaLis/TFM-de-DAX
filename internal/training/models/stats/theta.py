from statsforecast.models import Theta
from ._base import BaseStatsModel
 
class THETAModel(BaseStatsModel):
    
    def __init__(self, model_name = 'THETA', custom_params = None,ids_col: str = 'unique_id', date_col: str = 'ds', label_col = 'y'):
        super().__init__(model_name, custom_params, ids_col, date_col, label_col)

        self.model = Theta # do not change this

    def optuna_params(self, trial):

        return  {
            'season_length': trial.suggest_numerical('seanson_length', 3, 2),
            'decomposition_type': trial.suggest_categorical('decomposition_type', ['multiplicative', 'additive'])
            } 