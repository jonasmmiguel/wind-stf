from typing import Dict
import pandas as pd
import numpy as np
from src.wind_stf.pipelines.data_science.nodes import _get_predictions_e_gtruth
import pickle
import plotly.graph_objects as go
from matplotlib.colors import to_rgba


def set_rgba_color(css_code: str = 'MidnightBlue', alpha: float = 0.8) -> str:
    return 'rgba' + str(to_rgba(css_code, alpha=alpha))


def plot_gtruth_preds(gtruth: Dict[int, Dict[str, pd.Series]],
                      preds: Dict[int, Dict[str, pd.Series]],
                      node: str = 'DEF0C',
                      split: int = 0):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pd.concat((gtruth[split]['infer'][node], gtruth[split]['test'][node])).index,
                             y=pd.concat((gtruth[split]['infer'][node], gtruth[split]['test'][node])).values,
                             mode='lines',
                             name='gtruth',
                             line={'color': set_rgba_color('LightSlateGrey', 0.8)})
                  )

    fig.add_trace(go.Scatter(x=preds[split]['infer'][node].index,
                             y=preds[split]['infer'][node].values,
                             mode='lines',
                             name='preds train',
                             line={'color': set_rgba_color('Black', 0.8)}
                             )
                  )

    fig.add_trace(go.Scatter(x=preds[split]['test'][node].index,
                             y=preds[split]['test'][node].values,
                             mode='lines',
                             name='preds test',
                             line={'color': set_rgba_color('#ff6600', 0.8)}
                             )
                  )

    # Configure axes
    fig.update_xaxes(title_text=r'$date$')
    fig.update_yaxes(title_text=r'$CF$', range=[0.0, 1.0])

    # Configure layout
    fig.update_layout(
        legend={
            # 'xanchor': 'left',
            # 'x': 0.10,
            # 'yanchor': 'top',
            # 'y': 0.10
        },
        template='ggplot2',
    )

    fig.show()

    return


if __name__ == '__main__':
    with open(r'../data/06_models/model.pkl/2020-11-21T14.39.37.229Z/model.pkl',
              'rb') as f:
        model = pickle.load(f)

    capacity_factors_daily_2000to2015 = pd.read_hdf(
        '../data/04_feature/capfactors-daily-2000-2015.hdf',
        key='df'
    )

    with open(
            r'/home/jonasmmiguel/Documents/learning/poli/thesis/wind-stf/data/05_model_input/inference_test_splits_positions.pkl/2020-11-22T15.41.38.318Z/inference_test_splits_positions.pkl',
            'rb') as f:
        inference_test_splits_positions = pickle.load(f)

    with open(r'../data/05_model_input/scaler.pkl/2020-11-21T14.32.58.477Z/scaler.pkl',
              'rb') as f:
        scaler = pickle.load(f)

    gtruth, preds = _get_predictions_e_gtruth(
        model=model,
        df_unscaled=capacity_factors_daily_2000to2015,
        inference_test_splits_positions=inference_test_splits_positions,
        scaler=scaler
    )

    from src.learning_algs.baselines import NaiveForwardFill

    model = NaiveForwardFill().fit(gtruth[0]['infer'])

    forecasts = model.predict(
        start=gtruth[5]['test'].index[0],
        end=gtruth[5]['test'].index[-1],
    )

    print(gtruth[5]['infer'])
    print(forecasts)



    # print(inference_test_splits_positions[0]['infer'])
    # print(inference_test_splits_positions[0]['test'])
    print('done!')

