from typing import Dict
import pandas as pd
from src.wind_stf.pipelines.data_science.nodes import _get_predictions_e_gtruth
import pickle

def plot_gtruth_preds(gtruth: Dict[int, Dict[str, pd.Series]],
                      preds: Dict[int, Dict[str, pd.Series]]):
    return None


if __name__ == '__main__':
    with open(r'../data/06_models/model.pkl/2020-11-20T23.14.57.557Z/model.pkl',
              'rb') as f:
        model = pickle.load(f)

    capacity_factors_daily_2000to2015 = pd.read_hdf(
        '../data/04_feature/capfactors-daily-2000-2015.hdf',
        key='df'
    )

    with open(r'../data/05_model_input/inference_test_splits_positions.pkl/2020-11-20T23.10.52.665Z/inference_test_splits_positions.pkl',
              'rb') as f:
        inference_test_splits_positions = pickle.load(f)

    with open(r'../data/05_model_input/scaler.pkl/2020-11-20T23.14.49.566Z/scaler.pkl',
              'rb') as f:
        scaler = pickle.load(f)

    gtruth, preds = _get_predictions_e_gtruth(
        model=model,
        df_unscaled=capacity_factors_daily_2000to2015,
        inference_test_splits_positions=inference_test_splits_positions,
        scaler=scaler
    )


    print('done!')