import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

import warnings
warnings.filterwarnings('ignore')

def generate_plots(forecast, df_test, unique_id, df_past, unique_id_col='unique_id', date_col='ds', value_col='y', show_metrics=False, predictions_col = 'predictions' ):
    
    # df_past_subset = df_past[df_past[unique_id_col] == unique_id]
    # forecast_subset = forecast[forecast[unique_id_col] == unique_id]
    # df_test_subset = df_test[df_test[unique_id_col] == unique_id]

    # df_past_subset[date_col] = pd.to_datetime(df_past_subset[date_col])
    # df_test_subset[date_col] = pd.to_datetime(df_test_subset[date_col])
    # forecast_subset[date_col] = pd.to_datetime(forecast_subset[date_col])

    # df_historical = pd.concat([df_past_subset, df_test_subset], ignore_index=True)
    
    # plt.plot(df_historical[date_col], df_historical[value_col], label = 'Historical', linewidth = 2)
    # plt.plot(forecast_subset[date_col], forecast_subset[predictions_col], label='Forecast', linewidth = 2)

    # split_date = df_past[date_col].max()
    # plt.axvline(split_date, color='black', linestyle='-', linewidth=2)

    # if show_metrics:

    #     mae_forecast = mean_absolute_error(y_true=df_test_subset[value_col].values, y_pred=forecast_subset[predictions_col].values) 
    #     plt.text(0.02, 0.90, f'Forecast MAE: {mae_forecast:.2f}', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=1))


    # plt.xlabel('Date')
    # plt.ylabel('Values')
    # plt.title(f'Forecast vs Actual for unique_id: {unique_id}')

    # plt.legend()
    # plt.grid(True)
    # plt.show()

    unique_ids = df_past[unique_id_col].unique()

    for unique_id in unique_ids:
        # Subset data for the current unique_id
        df_past_subset = df_past[df_past[unique_id_col] == unique_id]
        forecast_subset = forecast[forecast[unique_id_col] == unique_id]
        df_test_subset = df_test[df_test[unique_id_col] == unique_id]

        # Convert date columns to datetime
        df_past_subset[date_col] = pd.to_datetime(df_past_subset[date_col])
        df_test_subset[date_col] = pd.to_datetime(df_test_subset[date_col])
        forecast_subset[date_col] = pd.to_datetime(forecast_subset[date_col])

        # Combine past and test data
        df_historical = pd.concat([df_past_subset, df_test_subset], ignore_index=True)

        # Plot historical and forecast data
        plt.figure(figsize=(10, 5))
        plt.plot(df_historical[date_col], df_historical[value_col], label='Historical', linewidth=2)
        plt.plot(forecast_subset[date_col], forecast_subset[predictions_col], label='Forecast', linewidth=2)

        # Mark split date
        split_date = df_past[date_col].max()
        plt.axvline(split_date, color='black', linestyle='-', linewidth=2)

        # Show metrics if enabled
        if show_metrics:
            mae_forecast = mean_absolute_error(
                y_true=df_test_subset[value_col].values,
                y_pred=forecast_subset[predictions_col].values
            )
            plt.text(0.02, 0.90, f'Forecast MAE: {mae_forecast:.2f}', 
                    transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=1))

        # Labels and title
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.title(f'Forecast vs Actual for unique_id: {unique_id}')

        plt.legend()
        plt.grid(True)
        plt.show()

def generate_plots_overall(forecast, df_test, df_past, ids_column, date_col='ds', value_col='y', show_metrics=False):
    
    df_past[date_col] = pd.to_datetime(df_past[date_col])
    df_test[date_col] = pd.to_datetime(df_test[date_col])
    forecast[date_col] = pd.to_datetime(forecast[date_col])

   
    df_past_subset = df_past.groupby(date_col, as_index=False)[value_col].sum()
    df_test_subset = df_test.groupby(date_col, as_index=False)[value_col].sum()
    forecast_subset = forecast.groupby(date_col, as_index=False)[value_col].sum()
    
    df_historical = pd.concat([df_past_subset, df_test_subset], ignore_index=True)

    seasonal_period = len(df_test_subset)
    last_value = df_past_subset[value_col].iloc[-1]  
    naive_forecast = pd.DataFrame({date_col: df_test_subset[date_col], value_col: [last_value] * len(df_test_subset)})

    seasonal_values = df_past_subset[value_col].iloc[-seasonal_period:].values
    seasonal_naive_forecast = pd.DataFrame({date_col: df_test_subset[date_col], 
                                            value_col: list(seasonal_values) * (len(df_test_subset) // seasonal_period) + 
                                            list(seasonal_values[:len(df_test_subset) % seasonal_period])})

    plt.figure(figsize=(10, 6))
    plt.plot(df_historical[date_col], df_historical[value_col], label='Historical', linewidth=2)
    plt.plot(forecast_subset[date_col], forecast_subset[value_col], label='Mutua Forecast', linewidth=2)
    # plt.plot(naive_forecast[date_col], naive_forecast[value_col], label='Naive Forecast', linestyle='--', color='gray')
    plt.plot(seasonal_naive_forecast[date_col], seasonal_naive_forecast[value_col], label='Seasonal Naive Forecast', linewidth = 2, linestyle='--', color='forestgreen')
    
    split_date = df_past[date_col].max()
    plt.axvline(split_date, color='black', linestyle='-', linewidth=2)
    
   
    if show_metrics:
        mae_forecast = mean_absolute_error(df_test_subset[value_col].values, forecast_subset[value_col].values)
        mae_naive = mean_absolute_error(df_test_subset.tail(12)[value_col].values, naive_forecast[value_col].values)
        mae_snaive = mean_absolute_error(df_test_subset.tail(12)[value_col].values, seasonal_naive_forecast[value_col].values)

        plt.text(0.02, 0.90, f'Forecast MAE: {mae_forecast:.2f}', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=1))
        plt.text(0.02, 0.85, f'Naive MAE: {mae_naive:.2f}', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=1))
        plt.text(0.02, 0.8, f'S-Naive MAE: {mae_snaive:.2f}', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=1))
    
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title(f'Forecast vs Actual for unique_id ')
    plt.legend()
    plt.grid(True)
    plt.show()

   
    df_merged = pd.merge(df_test, forecast, on=[ids_column, date_col], suffixes=('_actual', '_forecast'))

    better_count = []

    for unique_id in df_merged[ids_column].unique():
        df_past_id = df_past[df_past[ids_column] == unique_id]
        df_test_id = df_merged[df_merged[ids_column] == unique_id]

        naive_forecast = []
        for date in df_test_id[date_col]:
            past_date = date - pd.DateOffset(years=1)
            if past_date in df_past_id[date_col].values:
                naive_value = df_past_id[df_past_id[date_col] == past_date][value_col].values[0]
            else:
                naive_value = np.nan
            naive_forecast.append(naive_value)

        df_test_id['naive_forecast'] = naive_forecast
        
        df_test_id['ae_model'] = np.abs(df_test_id[f'{value_col}_actual'] - df_test_id[f'{value_col}_forecast'])
        df_test_id['ae_naive'] = np.abs(df_test_id[f'{value_col}_actual'] - df_test_id['naive_forecast'])

        better = df_test_id['ae_model'] < df_test_id['ae_naive']
        better_count.append(better.mean())

    overall_percentage_better = np.mean(better_count) * 100
    print(f'Mutua Engine es mejor en  {round(overall_percentage_better, 2)}')



import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def evaluate_forecasts(df_past, df_future, forecast, ids_col, date_col, value_col):

    unique_ids_list  = forecast[ids_col].unique()
    results = {}

    for unique_id in unique_ids_list:
      
        df_past_subset = df_past[df_past[ids_col] == unique_id]
        forecast_subset = forecast[forecast[ids_col] == unique_id]
        df_test_subset = df_future[df_future[ids_col] == unique_id]

        df_past_subset[date_col] = pd.to_datetime(df_past_subset[date_col])
        df_test_subset[date_col] = pd.to_datetime(df_test_subset[date_col])
        forecast_subset[date_col] = pd.to_datetime(forecast_subset[date_col])

        last_value = df_past_subset[value_col].iloc[-1]
        naive_forecast = pd.DataFrame({
            date_col: df_test_subset[date_col],
            value_col: [last_value] * len(df_test_subset) })
       
        seasonal_values = df_past_subset[value_col].tail(12).values
        seasonal_naive_forecast = pd.DataFrame({
            date_col: df_test_subset[date_col],
            value_col: list(seasonal_values) * (len(df_test_subset) // 12) +
                       list(seasonal_values[:len(df_test_subset) % 12])})
        
        rmse_forecast = root_mean_squared_error(
            y_true=df_test_subset[value_col].values,
            y_pred=forecast_subset['predictions'].values)
        
        rmse_naive = root_mean_squared_error(
            y_true=df_test_subset[value_col].values,
            y_pred=naive_forecast[value_col].values)
        
        rmse_snaive = root_mean_squared_error(
            y_true=df_test_subset[value_col].values,
            y_pred=seasonal_naive_forecast[value_col].values)

        results[unique_id] = {
            'forecast': float(round(rmse_forecast, 2)),
            'naive': float(round(rmse_naive, 2)),
            'seasonal_naive': float(round(rmse_snaive, 2)),
            'error_relative_naive':  float(round(rmse_forecast/rmse_naive,2)),
            'error_relative_snaive': float(round(rmse_forecast/rmse_snaive, 2))}

    return results
