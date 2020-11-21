from typing import Dict
import pandas as pd
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
