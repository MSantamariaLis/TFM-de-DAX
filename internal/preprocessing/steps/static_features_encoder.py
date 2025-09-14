import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class StaticFeaturesEncoder:

    def __init__(self, ids_col, date_col, label_col, **kwargs): 
    
        self.ids_col = ids_col
        self.date_col = date_col

        self.label_col = label_col
        self.static_features = kwargs.get('static_features', [])

        self.scaler = OneHotEncoder(sparse_output=False)

    def fit_transform(self, df:pd.DataFrame):
        """This class only applies to the static covariates"""
        return df
    
    def transform(self, df_predict:pd.DataFrame):
        """This class only applies to the static covariates"""
        return df_predict
    
    def _encode_features(self, df, features):
        """Helper method to encode static features."""
        encoded_features = self.scaler.transform(df[features])
        encoded_df = pd.DataFrame(
            encoded_features,
            columns=self.scaler.get_feature_names_out(features),
            index=df.index 
        )
        return encoded_df

    def fit_transform_covariates(self, df_covariates: pd.DataFrame):

        self.scaler.fit(df_covariates[self.static_features])
        encoded_df = self._encode_features(df_covariates, self.static_features)

        return pd.concat(
            [df_covariates.drop(columns=self.static_features), encoded_df],
            axis=1
        )

    def transform_covariates(self, df_future_covariates: pd.DataFrame):
       
        encoded_df = self._encode_features(df_future_covariates, self.static_features)
        return pd.concat(
            [df_future_covariates.drop(columns=self.static_features), encoded_df],
            axis=1
        )