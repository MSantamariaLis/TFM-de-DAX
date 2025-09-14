import pandas as pd


def calculate_frequency(df:pd.DataFrame, frequency:str, date_col:str):

    """ 
    Determines the frequency of the time series data based on the date column. If the frequency cannot be determined, raises a ValueError.
    Updates self.kwargs['frequency'] with the determined frequency.

    Parameters:
    -----------

    df (pd.DataFrame): The DataFrame containing the time series data.
    frequency (str): The frequency of the time series data, if known.
    date_col (str): The name of the date column in the DataFrame.
    """
    
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    sorted_dates = df[date_col].sort_values().drop_duplicates()

    inferred_frequency = pd.infer_freq(sorted_dates)
    frequency = frequency or inferred_frequency

    if frequency is None: 
        raise ValueError('Frequency could not be determined from the dataset. Please specify it in the preprocessing arguments. ')

    return frequency