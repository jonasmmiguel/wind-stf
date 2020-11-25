import pickle

import os

from nbeats_forecast import NBeats


if __name__ == '__main__':
    # print(os.getcwd())
    # with open('../data/05_model_input/df_infer_scaled.pkl', 'rb') as f:
    #     df = pickle.load(f)
    #
    # y = df[0]
    # data = y[['DE40D']].to_numpy().reshape((-1, 1))
    # model = NBeats(data=data, period_to_forecast=7, hidden_layer_units=8, backcast_length=3, train_percent=0.1, mode='cpu')
    # model.fit()
    # preds = model.predict()
    # print(preds)

    with open('../data/06_models/model.pkl', 'rb') as f:
        df = pickle.load(f)

    print('model')