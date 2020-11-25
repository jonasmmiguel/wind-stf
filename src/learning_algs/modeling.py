from typing import Dict, Any
import pandas as pd
from src.learning_algs.holtwinters import ExponentialSmoothing, ExponentialSmoothingRNN
from src.learning_algs.holtwinters import VectorExponentialSmoothing as MultioutputExponentialSmoothing
from src.learning_algs.arima import VectorARIMA
from src.learning_algs.nbeats import MultiOutputNBEATS
from src.learning_algs.baselines import HistoricalMedian, NaiveForwardFill
from src.utils.preprocessing import Scaler


class ForecastingModel:
    def __init__(self, modeling: Dict[str, Any]):
        self.modeling = modeling
        self.approach = modeling['approach']
        self.model = None

        self.targets = modeling['targets']
        self.freq = modeling['temporal_resolution']

    def fit(self, df):
        df = df[self.targets]

        if self.approach == 'HW-ES':
            self.model = MultioutputExponentialSmoothing(self.modeling).fit(df)

        elif self.approach == 'NaiveForwardFill':
            self.model = NaiveForwardFill().fit(df)

        elif self.approach == 'ARIMA':
            self.model = VectorARIMA(self.modeling).fit(df)

        elif self.approach == 'NBEATS':
            self.model = MultiOutputNBEATS(modeling=self.modeling, y=df).fit()

        elif self.approach == 'RNN-ES':
            self.model = None

        elif self.approach == 'GWNet':
            self.model = None

        else:
            raise NotImplementedError('')

        return self

    def predict(self, start, end, scaler: Scaler = None):
        df_pred = self.model.predict(start, end)

        if scaler:
            return scaler.inverse_transform(df_pred)[self.targets]
        else:
            return df_pred
