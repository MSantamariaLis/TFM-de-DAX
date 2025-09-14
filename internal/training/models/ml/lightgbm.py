from ._base import BaseMLModel
from lightgbm import LGBMRegressor 

class LightGBMModel(BaseMLModel):

    def __init__(self, model_name = 'LightGBM', custom_params = None, ids_col: str = 'unique_id', date_col: str = 'ds', label_col = 'y' ):
        super().__init__(model_name, custom_params, ids_col, date_col, label_col)

        self.model = LGBMRegressor # do not change this 

    def optuna_params(self, trial):

        params = {
            "differencing" : trial.suggest_categorical("differencing", [True, False]),
            "window_size": trial.suggest_int("window_size",  self.min_window_size, self.max_window_size),         
            "num_leaves": trial.suggest_int("num_leaves", 10, 250),
            # "max_depth": trial.suggest_int("max_depth", -1, 50),
            "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log = True),
        }

        return params
    
    def feature_importance(self):

        feature_importance = self.mlf.models_['LGBMRegressor'].feature_importances_
        total_columns = [col for col in 
                         self.mlf.preprocess(self.df_merged, id_col = self.ids_col, time_col = self.date_col, target_col=self.label_col, static_features=self.static_features).columns 
                         if col not in ['y', 'ds', 'unique_id']]

        feature_importance = feature_importance / feature_importance.sum()
        return total_columns, feature_importance
