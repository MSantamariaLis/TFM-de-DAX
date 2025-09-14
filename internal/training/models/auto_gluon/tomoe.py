from ._base import BaseAutoGluonModel

class TomoeModel(BaseAutoGluonModel):
    
    def __init__(self, model_name: str = 'Tomoe', custom_params: dict = None, ids_col: str = 'unique_id', date_col: str = 'ds', label_col: str = 'y'):
        super().__init__(model_name, custom_params, ids_col, date_col, label_col)

    def optuna_params(self, trial):

        params = {
            "presets": trial.suggest_categorical("presets", ["medium_quality", "fast_training", "high_quality", "best_quality"]),
        }

        return params
