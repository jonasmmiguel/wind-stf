from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils.validation import check_consistent_length
import pandas as pd
import numpy as np


def _get_error_naive(y_true, df_infer_unscaled):
    y_pred = df_infer_unscaled.tail(1)
    y_pred = np.tile(y_pred, reps=[len(y_true), 1])
    return y_pred - y_true


def root_mean_squared_error(y_true, y_pred, multioutput='uniform_average'):
    return mean_squared_error(y_true, y_pred, multioutput=multioutput) ** 0.5


def mean_absolute_percentage_error(y_true, y_pred,
                                   sample_weight=None,
                                   multioutput='uniform_average'):
    check_consistent_length(y_true, y_pred, sample_weight)
    epsilon = np.finfo(np.float64).eps
    mape = 100 * np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    output_errors = np.average(mape, weights=sample_weight, axis=0)
    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            return output_errors
        elif multioutput == 'uniform_average':
            return np.average(output_errors, weights=None)


def median_relative_absolute_error(y_true,
                                   y_pred,
                                   df_infer_unscaled,
                                   multioutput='uniform_average',
                                   ):
    error = y_pred - y_true
    error_naive = _get_error_naive(y_true, df_infer_unscaled)

    mdrae = np.median(np.abs(error / error_naive))

    if multioutput == 'raw_values':
        return mdrae
    elif multioutput == 'uniform_average':
        return np.average(mdrae, weights=None)


metrics_registered_basic = {
    'MAE': mean_absolute_error,
    'RMSE': root_mean_squared_error,
    'MAPE': mean_absolute_percentage_error,
}

metrics_registered_naivebased = {
    'MdRAE': median_relative_absolute_error,
}


