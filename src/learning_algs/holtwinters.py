from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
from typing import Dict, Any


class ExponentialSmoothingRNN(object):
    ...


class VectorExponentialSmoothing(ExponentialSmoothing):
    def __init__(self, modeling: Dict[str, Any]):
        self.targets = modeling['targets']
        self.hyperpars = modeling['hyperpars']
        self.model = {}
        self.freq = 'D'

    def fit(self, y):
        self.model = {col: super(y[col], **self.hyperpars).fit() for col in y}
        self.targets = y.columns
        self.freq = y.index.freq
        return self

    def predict(self, start, end, scaler=None):
        forecasts_timestamps = pd.date_range(start=start, end=end, freq=self.freq)
        forecasts = {col: self.model[col].predict(start, end) for col in self.targets}

        df_pred = pd.DataFrame(
            forecasts,
            index=forecasts_timestamps,
            )

        df_pred_unscaled = df_pred
        if scaler:
            df_pred_unscaled = scaler.inverse_transform(df_pred)

        return df_pred_unscaled




