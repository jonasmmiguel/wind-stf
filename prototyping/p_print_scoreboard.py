from typing import Dict
import pandas as pd
import numpy as np
from src.wind_stf.pipelines.data_science.nodes import _get_predictions_e_gtruth
import pickle
import plotly.graph_objects as go
from matplotlib.colors import to_rgba

pd.set_option('display.max_columns', None)

if __name__ == '__main__':
    with open(r'../data/08_reporting/scores_averaged.pkl/2020-11-21T23.54.03.701Z/scores_averaged.pkl',
              'rb') as f:
        scores = pickle.load(f)

    print(scores)
    # fig = go.Figure(data=[go.Table(
    #     header=dict(values=list(scores.columns),
    #                 fill_color='paleturquoise',
    #                 align='left'),
    #     cells=dict(values=[scores[col] for col in scores],
    #                fill_color='lavender',
    #                align='left'))
    # ])
    #
    # fig.show()
    #
    # # Configure layout
    # fig.update_layout(
    #     template='ggplot2',
    # )
    # fig.show()