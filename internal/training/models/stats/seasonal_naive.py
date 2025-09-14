from statsforecast.models import SeasonalNaive
from ._base import BaseStatsModel

class SeasonalNaiveModel(BaseStatsModel):

    def __init__(self, model_name = 'seasonal_naive', custom_params = None, ids_col: str = 'unique_id', date_col: str = 'ds', label_col = 'y'):
        super().__init__(model_name, custom_params, ids_col, date_col, label_col)

        self.model = SeasonalNaive # do not change

    def optuna_params(self, trial):

        return  {
            'season_length': trial.suggest_int('season_length', 3, 12)
            }
       
    def forward(self):
        raise ValueError(f'The model {self.model} does not support inference predictions')
