from nbeats_forecast import NBeats
import pandas as pd
from typing import Dict, Any
import numpy as np
import os
import glob


# noinspection PyPackageRequirements
# TODO: use model.save(), model.load(), model.predict() for making predictions
class MultiOutputNBEATS(object):
    def __init__(self, modeling: Dict[str, Any], y: pd.DataFrame):
        self.freq = modeling['temporal_resolution']
        self.hyperpars_trainer = {k: v for k, v in modeling['hyperpars'].items() if k in ['epoch']}
        self.hyperpars_model = {k: v for k, v in modeling['hyperpars'].items() if k not in ['epoch']}
        self.targets = modeling['targets']
        self.forecast_horizon = modeling['forecast_horizon']
        self.model = {}
        self.y = y[self.targets]
        self.earliest_train_end = modeling['model_inference_horizon']['end']['earliest_allowed']
        self.n_splits = modeling['n_splits']

        self.preds = {}

    def fit(self):
        self.freq = self.y.index.freq

        for col in self.y:
            model = NBeats(data=self.y[[col]].to_numpy().reshape((-1, 1)),
                                      period_to_forecast=self.forecast_horizon,
                                      **self.hyperpars_model,
                                      )
            model.fit(**self.hyperpars_trainer)

            # we save the model for the last week (last split), so that we can use them to plot preds vs gtruths
            split = len(glob.glob(f'./data/06_models/nbeats/{col}_week*.th'))
            if split == self.n_splits -1:
                model.save(f'data/06_models/nbeats/{col}_split{split}.th')

            self.preds[col] = model.predict()
        return self

    def predict(self, start, end) -> pd.DataFrame:
        forecasts_timestamps = pd.date_range(start=start, end=end, freq=self.freq)

        if start > pd.Timestamp(self.earliest_train_end):
            forecasts = {col: self.preds[col].flatten() for col in self.targets}
        else:
            forecasts = {col: np.zeros((len(forecasts_timestamps))) for col in self.targets}

        df_pred = pd.DataFrame(
            forecasts,
            index=forecasts_timestamps,
            )


        return df_pred

