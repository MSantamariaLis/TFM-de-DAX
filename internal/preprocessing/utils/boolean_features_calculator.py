import pandas as pd

def calculate_boolean_features(df_covariates:pd.DataFrame, ids_col:str, date_col:str, label_col:str):

    """
    Helper method to identify boolean features in the DataFrame. Boolean features are features that can take only two values, 
    such as True/False or 0/1.
    """
    
    if df_covariates is None:
        return []

    cols_to_check = [col for col in df_covariates.columns if col not in [ids_col, date_col, label_col]]
    boolean_features = []
    
    for col in cols_to_check:
        unique_values = df_covariates[col].drop_duplicates().nunique()
        if unique_values == 2:
            boolean_features.append(col)
    
    return boolean_features