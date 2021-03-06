import random
import pandas as pd
import numpy as np
import yaml

if __name__ == '__main__':
    # allowed_enddates = list( pd.date_range(start='2015-01-01', end='2015-12-31', freq='D') )
    # print(allowed_enddates)
    #
    # print(random.choice(allowed_enddates))

    with open(r'../conf/base/parameters.yml') as file:
        pars = yaml.load(file, Loader=yaml.FullLoader)

    freq = pars['modeling']['temporal_resolution']
    gap = pd.to_timedelta(pars['modeling']['gap'], freq)
    forecast_horizon = pd.to_timedelta(pars['modeling']['forecast_horizon'], freq)
    modinfer_window_start = pars['modeling']['model_inference_horizon']['start']
    modinfer_window_end_earliest = pars['modeling']['model_inference_horizon']['end']['earliest_allowed']
    modinfer_window_end_latest = pars['modeling']['model_inference_horizon']['end']['latest_allowed']
    n_splits = pars['modeling']['n_splits']

    available_window = pd.date_range(start=modinfer_window_end_earliest,
                                    end=modinfer_window_end_latest - gap - forecast_horizon,
                                    freq=freq)

    w = int( np.floor( len(available_window) / n_splits ) )  # subwindow width

    for k in range(n_splits):

        print(f'SPLIT {k}: start: {available_window[k * w]} --- end: {available_window[(k+1) * w]}')

        modinfer_window_end = random.choice(
            pd.date_range(start=available_window[k * w],
                          end=available_window[(k+1) * w],
                          freq=freq)
        )

        model_inference_window = slice(modinfer_window_start,
                                       modinfer_window_end)

        test_window = slice(modinfer_window_end + gap,
                            modinfer_window_end + gap + forecast_horizon)

        print(f'modinfer end: {modinfer_window_end}')


