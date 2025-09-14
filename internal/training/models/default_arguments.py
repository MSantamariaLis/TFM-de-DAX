
stats_arguments = {

    'ARIMA': {

        'order': (3, 0, 0),
        'season_length': 12,
        'seasonal_order': (0, 0, 0),
        'include_mean': True,
        'include_drift': False,
        'include_constant': None,
    },

    'ETS': {
        'season_length': 12,
        'model': "ZZZ",
        'damped': None,
        'phi': None,
    },

    'THETA': {
        'seasonal_length': 12,
    }
    
}

ml_arguments = {

    'XGBoost': {
        'window_size': 12, 
        'differencing':False,
        'min_window_size': 1,
        'max_window_size':12, 
        'multivariate': True,
        'direct_strategy': False
        },

    'LightGBM':{
        'differencing':False,
        'window_size': 12,
        'verbosity': -1,
        'num_leaves': 128,
        'max_depth': -1,
        'n_estimators': 100,
        'min_window_size': 1,
        'max_window_size':12,
        'multivariate': True,
        'direct_strategy': False
    },

    'ExtraTrees': {
        'differencing':False,
        'window_size': 12,
        'n_estimators': 10,
        'max_depth': 10,
        'min_window_size': 1,
        'max_window_size':12,
        'multivariate': True,
        'direct_strategy': False
    },

    'DecisionTrees': {
        'differencing':False,
        'window_size': 12,
        'max_depth': 10,
        'min_window_size': 1,
        'max_window_size':12,
        'multivariate': True,
        'direct_strategy': False
    },

    'RandomForest': {
        'differencing':False,
        'window_size': 12,
        'n_estimators': 10,
        'max_depth': 10,
        'min_window_size': 1,
        'max_window_size':12,
        'multivariate': True,
        'direct_strategy': False
    },


    'LinearRegression': {
        'differencing':False,
        'window_size': 12,
        'min_window_size': 1,
        'max_window_size':12,
        'multivariate': True,
        'direct_strategy': False
    }
}

neural_arguments = {

    'NBEATS': {
        'window_size': 12,
        'logger': False,
        'enable_progress_bar':False,
        'enable_checkpointing':False,

    },
    'LSTM': {
        'window_size': 12,
        'logger': False,
        'enable_progress_bar':False,
          'enable_checkpointing':False

    },
    'NHITS': {
        'window_size': 12,
        'logger': False,
        'enable_progress_bar':False,
          'enable_checkpointing':False

    },
    'RNN': {
        'window_size': 12,
        'logger': False,
        'enable_progress_bar':False,
        'enable_checkpointing':False

    },
    'KAN': {
        'window_size': 12,
        'logger': False,
        'enable_progress_bar':False,
        'enable_checkpointing':False

    },
    'TFT': {
        'window_size': 12,
        'logger': False,
        'enable_progress_bar':False,
        'enable_checkpointing':False
    },
}

auto_gluon_arguments = {
    'Tomoe': {
        'presets': 'fast_training'
    },
    'Chronos': {
        'presets': 'bolt_small'
    }


}
