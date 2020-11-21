from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from sklearn.linear_model import LinearRegression
from sklearn.base import MultiOutputMixin, RegressorMixin
import numpy as np
import pandas as pd
from typing import Dict, Any


class HistoricalMedian(MultiOutputMixin, RegressorMixin):
    def __init__(self):
        self.medians = None
        self.targets = None
        self.freq = 'D'

    def fit(self, y):
        self.medians = np.median(y)
        self.targets = y.columns
        self.freq = y.index.freq
        return self

    def predict(self, start, end):
        forecasts_timestamps = pd.date_range(start=start, end=end, freq=self.freq)
        forecasts = np.tile(self.medians,
                            reps=[len(forecasts_timestamps), 1])
        return pd.DataFrame(forecasts,
                            columns=self.targets,
                            index=forecasts_timestamps)


class NaiveForwardFill(MultiOutputMixin, RegressorMixin):
    def __init__(self):
        self.last_train_observation = None
        self.targets = None
        self.freq = 'D'

    def fit(self, y):
        self.last_train_observation = y.tail(1)
        self.targets = y.columns
        self.freq = y.index.freq
        return self

    def predict(self, start, end) -> pd.DataFrame:
        forecasts_timestamps = pd.date_range(start=start, end=end, freq=self.freq)
        forecasts = np.tile(self.last_train_observation,
                            reps=[len(forecasts_timestamps), 1])
        return pd.DataFrame(forecasts,
                            columns=self.targets,
                            index=forecasts_timestamps)




