from nbeats_forecast import NBeats
import pandas as pd
from typing import Dict, Any
import numpy as np

class MultiOutputNBEATS(object):
    def __init__(self, modeling: Dict[str, Any], y: pd.DataFrame):
        self.freq = modeling['temporal_resolution']
        self.hyperpars = modeling['hyperpars']
        self.targets = modeling['targets']
        self.forecast_horizon = modeling['forecast_horizon']
        self.model = {}
        self.y = y[self.targets]

        self.preds = {}

    def fit(self):
        self.freq = self.y.index.freq

        for col in self.y:
            model = NBeats(data=self.y[[col]].to_numpy().reshape((-1, 1)),
                                      period_to_forecast=self.forecast_horizon,
                                      **self.hyperpars,
                                      )
            model.fit(epoch=1)
            self.preds[col] = model.predict()
        return self

    def predict(self, start, end) -> pd.DataFrame:
        forecasts_timestamps = pd.date_range(start=start, end=end, freq=self.freq)

        if start > pd.Timestamp('2015-01-01'):
            forecasts = {col: self.preds[col].flatten() for col in self.targets}
        else:
            forecasts = {col: np.zeros((len(forecasts_timestamps))) for col in self.targets}

        df_pred = pd.DataFrame(
            forecasts,
            index=forecasts_timestamps,
            )


        return df_pred

