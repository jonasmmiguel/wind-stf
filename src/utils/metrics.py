from sklearn.metrics import mean_squared_error, mean_absolute_error as mae
from sklearn.utils.validation import check_consistent_length
import pandas as pd
import numpy as np


def _get_error_naive(y_true, df_infer_unscaled):
    y_pred = df_infer_unscaled.tail(1)
    y_pred = np.tile(y_pred, reps=[len(y_true), 1])
    return y_pred - y_true


def root_mean_squared_error(y_true, y_pred, multioutput_avg):
    if multioutput_avg:
        return mean_squared_error(y_true, y_pred, multioutput='uniform_average') ** 0.5
    else:
        return mean_squared_error(y_true, y_pred, multioutput='raw_values') ** 0.5


def mean_absolute_error(y_true, y_pred, multioutput_avg):
    if multioutput_avg:
        return mae(y_true, y_pred, multioutput='uniform_average')
    else:
        return mae(y_true, y_pred, multioutput='raw_values')


def mean_absolute_percentage_error(y_true, y_pred, multioutput_avg, sample_weight=None):
    check_consistent_length(y_true, y_pred, sample_weight)
    epsilon = np.finfo(np.float64).eps
    mape = 100 * np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    output_errors = np.average(mape, weights=sample_weight, axis=0)
    if not multioutput_avg:
        return output_errors
    else:
        return np.average(output_errors, weights=sample_weight)


def median_relative_absolute_error(y_true,
                                   y_pred,
                                   df_infer_unscaled,
                                   multioutput_avg,
                                   ):
    error = y_pred - y_true
    error_naive = _get_error_naive(y_true, df_infer_unscaled)

    mdrae = np.median(np.abs(error / error_naive))

    if not multioutput_avg:
        return mdrae
    else:
        return np.average(mdrae, weights=None)


metrics_registered_basic = {
    'MAE': mean_absolute_error,
    'RMSE': root_mean_squared_error,
    'MAPE': mean_absolute_percentage_error,
}

metrics_registered_naivebased = {
    'MdRAE': median_relative_absolute_error,
}


