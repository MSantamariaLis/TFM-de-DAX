from neuralforecast.models import KAN
from ._base import BaseNeuralModel

class KANModel(BaseNeuralModel):

    def __init__(self, model_name = 'KAN', custom_params = None,  ids_col: str = 'unique_id', date_col: str = 'ds', label_col:str = 'y' ):
        super().__init__(model_name, custom_params, ids_col, date_col, label_col)

        self.model = KAN # do not change this

    def optuna_params(self, trial):

        params = {
            "window_size": trial.suggest_int("window_size",  self.min_window_size, self.max_window_size),
        }

        return params
    
    