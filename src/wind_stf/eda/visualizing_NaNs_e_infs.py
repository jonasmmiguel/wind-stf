import pandas as pd
import numpy as np
import pickle
import plotly
import plotly.graph_objects as go
from matplotlib.colors import to_rgba
import yaml


def validate_data(
        df_: pd.DataFrame,
        tolerance: dict ={
            'nan': False,
            'inf': False,
            'value_max': 2.0,
            'value_min': 0.0
        }):

    is_valid = False
    inconsistencies = {'NaNs': df_.isna().sum().sum(),
                       'infs count': np.isinf(df_).sum().sum(),
                       'value<min count': (df_ < tolerance['value_min']).sum().sum(),
                       'value>max count': (df_ > tolerance['value_max']).sum().sum()}

    inconsistencies_distribution = {'NaNs': df_.isna().sum(),
                'infs': np.isinf(df_).sum(),
                'value<min': (df_ < tolerance['value_min']).sum(),
                'value>max': (df_ > tolerance['value_max']).sum()}

    for culprit in inconsistencies_distribution['value>max'].index:
        print(df_[culprit].describe()[['50%', '75%', 'max']])

    return print(f'Data valid? {is_valid}; Length (years): {len(df_) / 365}; Inconsistency: {inconsistencies}; Col_counts: {inconsistencies_distribution}')


if __name__ == '__main__':
    # Load
    with open(r'../../../conf/base/parameters.yml') as file:
        pars = yaml.load(file, Loader=yaml.FullLoader)

    df = pd.read_hdf(
        '../../../data/04_feature/capfactors-daily-2000-2015.hdf',
        'df'
    )

    # Slice
    targets = pars['modeling']['targets']
    # df = df[targets]

    # Plot
    display = True
    if display:
        fig = go.Figure(go.Heatmap(
            z=df.values,
            x=df.columns,
            y=df.index,
            zmax=1.0,
            zmin=0.0,
            colorscale='Inferno',
            # showscale=False,
            )
        )

        # configure axes
        # fig.update_coloraxes(showscale=False)

        # Configure layout
        fig.update_layout(
            template='ggplot2',
            yaxis=dict(
                tickmode='array',
                tickvals=[pd.Timestamp(f'{year}-01-01') for year in range(2000, 2017) ],
                ticktext=[year for year in range(2000, 2017)],
            ),
            xaxis=dict(
                showticklabels=False,
                ticks='',
            ),
            coloraxis_colorbar=dict(
                # lenmode='pixels', len=700,
                # showscale=False,
            ),
            # height=600,
            width=800,
            margin=dict(l=0, r=0, t=0, b=0),

        )

        # fig.show()
        fig.write_image("/home/jonasmmiguel/Documents/learning/poli/thesis/wind-stf/data/08_reporting/data_availability.pdf", scale=3.0)
    # Validate
    # validate_data(df)




