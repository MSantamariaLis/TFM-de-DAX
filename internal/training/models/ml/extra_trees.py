from ._base import BaseMLModel
from sklearn.ensemble import ExtraTreesRegressor

class ExtraTreesModel(BaseMLModel):
    def __init__(self, model_name='ExtraTrees', custom_params = None, ids_col: str = 'unique_id', date_col: str = 'ds', label_col = 'y' ):
        super().__init__(model_name, custom_params, ids_col, date_col, label_col)

        self.model = ExtraTreesRegressor # do not change this

    def optuna_params(self, trial):

        params = {
            "differencing" : trial.suggest_categorical("differencing", [True, False]),
            "window_size": trial.suggest_int("window_size",  self.min_window_size, self.max_window_size),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 50),
        }

        return params
