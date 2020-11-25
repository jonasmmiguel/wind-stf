from nbeats_forecast import NBeats
import pandas as pd
from typing import Dict, Any


class MultiOutputNBEATS(object):
    def __init__(self, modeling: Dict[str, Any], y: pd.DataFrame):
        self.freq = modeling['temporal_resolution']
        self.hyperpars = modeling['hyperpars']
        self.targets = modeling['targets']
        self.forecast_horizon = modeling['forecast_horizon']
        self.model = {}
        self.y = y

    def fit(self):
        self.model = {col: NBeats(data=self.y[[col]].to_numpy().reshape((-1, 1)),
                                  period_to_forecast=self.forecast_horizon,
                                  **self.hyperpars).fit() for col in self.y}
        self.freq = self.y.index.freq
        return self

    def predict(self, start, end) -> pd.DataFrame:
        forecasts_timestamps = pd.date_range(start=start, end=end, freq=self.freq)
        forecasts = {col: self.model[col].predict() for col in self.targets}

        df_pred = pd.DataFrame(
            forecasts,
            index=forecasts_timestamps,
            )

        return df_pred

