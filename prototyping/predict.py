import pickle
import pandas as pd
from learning_algs.modeling import ForecastingModel

if __name__ == '__main__':
    df_infer = capacity_factors_daily_2000to2015 = pd.read_hdf(
        path_or_buf='../data/05_model_input/df_infer.hdf',
        key='df_infer'
    )

    df_infer_scaled = capacity_factors_daily_2000to2015 = pd.read_hdf(
        path_or_buf='../data/05_model_input/df_infer_scaled.hdf',
        key='df_infer_scaled'
    )

    splits_positions = pickle.load(
        open('../data/05_model_input/cv_splits_positions.pkl/2020-10-09T00.50.52.126Z/cv_splits_positions.pkl', 'rb'))

    scaler = pickle.load(open('../data/05_model_input/scaler.pkl/2020-10-09T15.53.34.176Z/scaler.pkl', 'rb'))

    modeling = {
        'approach': 'HW-ES',
        'mode': 'districtwise',
        'model_inference_window': {
            'start': '2013-01-01',
            'end': '2015-06-22',
        },
        'test_window': {
            'start': '2015-06-23',
            'end': '2015-06-29',
        },
        'preprocessing': ['get_quantile_equivalent_normal_dist', 'make_strictly_positive'],
        'hyperpars': {
            'trend': 'additive',
            'seasonal': 'multiplicative',
            'seasonal_periods': 7,
        },
        'targets': ['DEF0C', 'DE111'],
    }

    df = df_infer_scaled

    # CORE ––––––––––––––––––––––––––––––––––
    model = {}

    # ignore all vars we don't want to model
    targets = modeling['targets']
    df = df[targets]

    # train for every cv split
    for pass_id in splits_positions.keys():
        df_train = df[splits_positions[pass_id]['train']]
        model[pass_id] = ForecastingModel(modeling).fit(df_train)

    # train model on whole inference dataset
    model['full'] = ForecastingModel(modeling).fit(df)

    start = df.index[0]
    end = df.index[-1]

    # predict
    print( model['full'].predict(start, end, scaler) )

    print('done!')