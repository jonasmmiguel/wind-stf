from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
from typing import Dict, Any


class VectorARIMA(object):
    def __init__(self, modeling: Dict[str, Any]):
        self.freq = modeling['temporal_resolution']
        self.hyperpars = modeling['hyperpars']
        self.targets = modeling['targets']
        self.model = {}

    def fit(self, y):
        self.order = (self.hyperpars['p'], self.hyperpars['d'], self.hyperpars['q'])
        self.model = {col: ARIMA(y[col], order=self.order).fit() for col in y}
        self.freq = y.index.freq
        return self

    def predict(self, start, end) -> pd.DataFrame:
        forecasts_timestamps = pd.date_range(start=start, end=end, freq=self.freq)
        forecasts = {col: self.model[col].predict(start, end) for col in self.targets}

        df_pred = pd.DataFrame(
            forecasts,
            index=forecasts_timestamps,
            )

        return df_pred

