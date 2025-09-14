from neuralforecast.models import TFT
from ._base import BaseNeuralModel

class TFTModel(BaseNeuralModel):

    def __init__(self, model_name = 'TFT', custom_params = None, ids_col: str = 'unique_id', date_col: str = 'ds', label_col:str = 'y' ):
        super().__init__(model_name, custom_params, ids_col, date_col, label_col)

        self.model = TFT # do not change this

    def optuna_params(self, trial):

        params = {
            "window_size": trial.suggest_int("window_size",  self.min_window_size, self.max_window_size),
        }

        return params