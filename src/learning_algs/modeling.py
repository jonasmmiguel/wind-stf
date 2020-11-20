from typing import Dict, Any
import pandas as pd
from src.learning_algs.holtwinters import ExponentialSmoothing, ExponentialSmoothingRNN
from src.learning_algs.holtwinters import VectorExponentialSmoothing as MultioutputExponentialSmoothing
from src.learning_algs.baselines import HistoricalMedian


class ForecastingModel:
    def __init__(self, modeling: Dict[str, Any]):
        self.modeling = modeling
        self.approach = modeling['approach']
        self.model = None

        self.targets = modeling['targets']

    def fit(self, df):
        if self.approach == 'HW-ES':
            self.model = MultioutputExponentialSmoothing(self.modeling).fit(df)

        elif self.approach == 'RNN-ES':
            self.model = None

        elif self.approach == 'GWNet':
            self.model = None

        else:
            raise NotImplementedError('')

        return self

    def predict(self, start, end, scaler):
        df_pred_unscaled = self.model.predict(start, end, scaler)
        # TODO: transform model outputs (CF) into outputs of interest (kW)
        return df_pred_unscaled[ self.targets ]
