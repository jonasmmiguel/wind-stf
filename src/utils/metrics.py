from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils.validation import check_consistent_length
import numpy as np


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
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


metrics_registered = {
    'MAE': mean_absolute_error,
    'RMSE': root_mean_squared_error,
    'MAPE': mean_absolute_percentage_error
}

