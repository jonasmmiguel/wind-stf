# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

Delete this when you start working on your own Kedro project.
"""
# pylint: disable=invalid-name

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from src.utils.preprocessing import registered_transformers, make_pipeline
from learning_algs.modeling import ForecastingModel

from src.utils.metrics import metrics_registered


def _sort_col_level(df: pd.DataFrame, levelname:str ='nuts_id'):
    target_order = df.columns.sortlevel(level=levelname)[0]
    df = df[target_order]
    return df


def _get_districts(df: pd.DataFrame) -> set:
    return set(df.columns.get_level_values('nuts_id'))


def _split_train_eval(df: pd.DataFrame, train_window, eval_window):
    train = slice(
        train_window['start'],
        train_window['end']
    )

    eval = slice(
        eval_window['start'],
        eval_window['end']
    )
    return {
        'df_train': df[train],
        'df_eval': df[eval]
    }


def split_modinfer_test(df: pd.DataFrame, modeling: dict) -> Tuple[Any, Any]:
    infer_window = modeling['model_inference_window']
    test_window = modeling['test_window']

    infer = slice(
        infer_window['start'],
        infer_window['end']
    )

    test = slice(
        test_window['start'],
        test_window['end']
    )

    return df[infer], df[test]


def build_spatiotemporal_dataset(
        df_spatial: pd.DataFrame,
        df_temporal: pd.DataFrame,
) -> pd.DataFrame:

    # sort districts order in dataframes for easier dataframes tabular visualization.
    df_spatial = _sort_col_level(df_spatial, 'nuts_id')
    df_temporal = _sort_col_level(df_temporal, 'nuts_id')

    # build spatiotemporal dataframe via concatenatation
    df_spatiotemporal = pd.concat(
        {'spatial': df_spatial, 'temporal': df_temporal},
        axis=1,  # column-wise concatenation
        join='inner',  # only timestamps in both df's are included
    )
    df_spatiotemporal.columns.names = ['data_type', 'district', 'var']

    # include only districts present in both df_temporal and df_spatial
    districts_to_include = set(
        _get_districts(df_temporal).intersection(
            _get_districts(df_spatial))
    )

    df_spatiotemporal = df_spatiotemporal.loc[
        :,
        (slice(None), districts_to_include)
    ]

    return df_spatiotemporal


def _get_last_train_idx(tss: pd.DataFrame, cv_splits_dict: dict):
    longest_pass = list( cv_splits_dict.keys() )[-1]
    y = _split_train_test(tss, cv_splits_dict, pass_id=longest_pass)
    return y['train'].index[-1]


def _load_model():
    pass


def scale(df_infer: pd.DataFrame, modeling: List[str]) -> List[Any]:
    preprocessing = modeling['preprocessing']

    # instantiate pipeline with steps defined in preprocessing params
    scaler = make_pipeline(
        *[registered_transformers[step] for step in preprocessing]
    )

    scaler = scaler.fit(
        df_infer
    )

    # transformation output is a numpy array
    df_infer_scaled = pd.DataFrame(
        data=scaler.transform(df_infer),
        index=df_infer.index,
        columns=df_infer.columns,
    )

    return [df_infer_scaled, scaler]


def define_cvsplits(cv_params: Dict[str, Any], df_infer: pd.DataFrame) -> Dict[str, Any]:  # Dict[str, List[pd.date_range, List[str]]]:
    """
    :param df:
    :param window_size_first_pass:uz
    :param window_size_last_pass:
    :param n_passes:
    :param forecasting_window_size:
    :return: cv_splits_dict

    Example of Cross-Validation Splits Dictionary:

    cv_splits_dict = {
        'pass_1': {
            'train_idx': [0, 365],
            'test_idx': [365, 465],
        }
    }
    """
    cv_method = cv_params['method']
    n_passes = cv_params['n_passes']
    relsize_shortest_train_window = cv_params['relsize_shortest_train_window']
    size_forecasting_window = cv_params['size_forecast_window']
    steps_ahead = cv_params['steps_ahead']

    if cv_method == 'expanding window':
        cv_splits = {}

        # max train size so that train  + gap (steps_ahead) + val still fit in inference data
        relsize_longest_train_window = 1 - ((steps_ahead - 1) + size_forecasting_window) / len(df_infer)

        window_relsize = np.linspace(
            start=relsize_shortest_train_window,
            stop=relsize_longest_train_window,
            num=n_passes
        )

        for p in range(n_passes):
            pass_id = 'pass ' + str(p + 1)

            train_end_idx = round(window_relsize[p] * len(df_infer))

            cv_splits[pass_id] = {
                'train': slice(0, train_end_idx),
                'val': slice(train_end_idx, train_end_idx + size_forecasting_window)
            }
        return cv_splits

    else:
        raise NotImplementedError(f'CV method not recognized: {cv_method}')


def cv_train(df: pd.DataFrame,
             modeling: Dict[str, Any],
             splits_positions: Dict[str, Any]) -> Dict[str, Any]:
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

    return model


def _get_scores(gtruth: Dict[str, Any], preds: Dict[str, Any], avg=True):
    all_metrics = list( metrics_registered.keys() )
    all_passes = preds.keys()

    if avg:
        multioutput = 'uniform_average'
    else:
        multioutput = 'raw_values'

    scores = pd.DataFrame(
        data=None,
        index=pd.MultiIndex.from_product([all_metrics, ['train', 'val', 'test']]),
        columns=all_passes
    )
    all_passes = list(all_passes) + ['full']
    for pass_id in all_passes:
        for cat in ['train', 'val', 'test']:
            for metric in all_metrics:
                try:
                    scores.loc[(metric, cat), pass_id] = metrics_registered[metric](
                        gtruth[pass_id][cat],
                        preds[pass_id][cat],
                        multioutput=multioutput
                    )
                except:
                    pass
    return scores


def evaluate(model: Any, cv_splits_positions: Dict[str, Any], df_infer: pd.DataFrame, df_test: pd.DataFrame, scaler: Any) -> Any:
    gtruth, preds = _get_predictions_e_gtruth(model, cv_splits_positions, df_infer, df_test, scaler)
    scores_nodewise = _get_scores(gtruth, preds, avg=False)
    scores_averaged = _get_scores(gtruth, preds, avg=True)

    return scores_nodewise, scores_averaged


def _get_predictions_e_gtruth(model, cv_splits_positions, df_infer, df_test, scaler):
    gtruth = {}
    preds = {}
    targets = model['full'].modeling['targets']
    for pass_id in cv_splits_positions.keys():
        gtruth[pass_id] = {}
        preds[pass_id] = {}
        for cat in ['train', 'val']:

            window = df_infer[cv_splits_positions[pass_id][cat]].index
            start = window[0]
            end = window[-1]

            gtruth[pass_id][cat] = df_infer[slice(start, end)][targets]
            preds[pass_id][cat] = model[pass_id].predict(start, end, scaler)

    # predictions for model trained on entire inference dataset
    gtruth['full'] = {}
    preds['full'] = {}

    for cat in ['train', 'test']:

        if cat == 'train':
            df = df_infer
        else:
            df = df_test

        start = df.index[0]
        end = df.index[-1]

        gtruth['full'][cat] = df[slice(start, end)][targets]
        preds['full'][cat] = model['full'].predict(start, end, scaler)

    return gtruth, preds



def _convert_CFtokW():
    pass


def _evaluate_model(_model_metadata: Any, _cv_splits_dict: dict) -> dict:
    pass


def _update_scoreboard():
    pass


def report_scores(scoreboard: pd.DataFrame, model_metadata: Any, cv_splits_dict: dict):
    # TODO: use metrics from utils/metrics.py
    model_scores = _evaluate_model(model_metadata, cv_splits_dict)
    _update_scoreboard(model_scores)
