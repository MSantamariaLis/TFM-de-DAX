import pandas as pd

def calculate_static_features(df_covariates:pd.DataFrame, ids_col:str, date_col:str, label_col:str):

    """  Identifies static features in the covariates DataFrame. Thosea features that remain constant over time, such as product category, store location, etc. """
    
    if df_covariates is None: return []

    cols_to_check = [col for col in df_covariates.columns if col not in [ids_col, date_col, label_col]]
    static_features = []
    
    for col in cols_to_check:
        is_static_per_group = df_covariates.groupby(ids_col)[col].nunique() == 1
        if is_static_per_group.all():
            static_features.append(col)
    
    return static_features