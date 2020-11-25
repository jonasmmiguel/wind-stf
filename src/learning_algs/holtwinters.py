import pandas as pd
from typing import Dict, Any
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
warnings.simplefilter('ignore', ConvergenceWarning)

class ExponentialSmoothingRNN(object):
    ...


class VectorExponentialSmoothing(object):
    def __init__(self, modeling: Dict[str, Any]):
        self.freq = modeling['temporal_resolution']
        self.hyperpars = modeling['hyperpars']
        self.targets = modeling['targets']
        self.model = {}

    def fit(self, y):
        self.model = {col: ExponentialSmoothing(y[col], **self.hyperpars).fit() for col in y}
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



