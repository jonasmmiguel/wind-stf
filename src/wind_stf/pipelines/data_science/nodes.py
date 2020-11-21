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
import logging

logging.basicConfig(filename='ds_nodes.log', level=logging.DEBUG)


import numpy as np
import pandas as pd
import random

from src.utils.preprocessing import registered_transformers, make_pipeline, Scaler
from src.utils.metrics import metrics_registered
from src.utils.visualization import _plot_gtruth_preds, _plot_error_boxplots
from learning_algs.modeling import ForecastingModel


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


def _load_model():
    pass


def scale(df: pd.DataFrame,
          modeling: Dict[str, Any],
          inference_test_splits_positions: Dict[int, Dict[str, Any]]
          ) -> List[Dict]:
    '''

    :param df:
    :param modeling:
    :param inference_test_splits_positions:
    :return:
        df_infer_scaled: e.g. {O: <pd.DataFrame>, 1: <pd.DataFrame>, ..., n_splits-1: <pd.DataFrame>}
        scaler:          e.g. {O: <Scaler>, 1: <Scaler>, ..., n_splits-1: <Scaler>}
    '''

    preprocessing = modeling['preprocessing']

    df_infer_scaled = {}
    scaler = {}
    for split in inference_test_splits_positions.keys():
        # TODO: split in a previous, separate pipeline node
        df_infer = df[ inference_test_splits_positions[split]['infer'] ]

        if preprocessing:
            # instantiate pipeline with steps defined in preprocessing params
            scaler[split] = make_pipeline(
                *[registered_transformers[step] for step in preprocessing]
            )

            scaler[split] = scaler[split].fit( df_infer )

            # transformation output is a numpy array
            df_infer_scaled[split] = pd.DataFrame(
                data=scaler[split].transform(df_infer),
                index=df_infer.index,
                columns=df_infer.columns,
            )
        else:
            scaler[split] = None
            df_infer_scaled[split] = df_infer

    return [df_infer_scaled, scaler]


def define_inference_test_splits(modeling: Dict[str, Any],
                                 stratify: bool = True
                                 ) -> Dict[int, Dict[str, Any]]:
    freq = modeling['temporal_resolution']
    gap = pd.to_timedelta(modeling['gap'], freq)  # +1 to ensure nonoverlapping infer-test slices
    unit_delta = pd.to_timedelta(1, freq)  # +1 to ensure nonoverlapping infer-test slices
    forecast_horizon = pd.to_timedelta(modeling['forecast_horizon'], freq)
    modinfer_window_start = modeling['model_inference_horizon']['start']
    modinfer_window_end_earliest = modeling['model_inference_horizon']['end']['earliest_allowed']
    modinfer_window_end_latest = modeling['model_inference_horizon']['end']['latest_allowed']
    n_splits = modeling['n_splits']
    random.seed(2020)

    available_window = pd.date_range(start=modinfer_window_end_earliest,
                                     end=modinfer_window_end_latest - gap - forecast_horizon,
                                     freq=freq)

    w = int(np.floor(len(available_window) / n_splits))  # strata width

    inference_test_splits_positions = {}
    for k in range(n_splits):
        if not stratify:
            modinfer_window_end = random.choice(
                pd.date_range(start=available_window[0],
                              end=available_window[-1],
                              freq=freq)
            )
        else:
            modinfer_window_end = random.choice(
                pd.date_range(start=available_window[k * w],
                              end=available_window[(k+1) * w],
                              freq=freq)
            )

        model_inference_window = slice(modinfer_window_start,
                                       modinfer_window_end)

        test_window = slice(modinfer_window_end + gap + unit_delta,
                            modinfer_window_end + gap + forecast_horizon)

        inference_test_splits_positions[k] = {
            'infer': model_inference_window,
            'test': test_window
        }
    return inference_test_splits_positions


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

        # max train size so that train  + gap (steps_ahead) + val still fit in inference window
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


def train(df_infer_scaled: pd.DataFrame,
          modeling: Dict[str, Any],
          # cv: Dict[str, Any]
          ) -> Dict[int, Dict[str, Any]]:
    # TODO: reduce memory usage. Probably too high.
    # TODO: enable training for every CV-split.
    model = {}

    for infer_test_split in range( modeling['n_splits'] ):
        df = df_infer_scaled[infer_test_split]

        # ignore all vars we don't want to model
        targets = modeling['targets']
        df = df[targets]

        # model[infer_test_split] = {}
        # if cv:
        #     # train for every cv (train-val) split
        #     for cv_split in train_val_splits_positions.keys():
        #         df_train = df[train_val_splits_positions[cv_split]['train']]
        #         model[infer_test_split][cv_split] = ForecastingModel(modeling).fit(df_train)

        # train model on whole inference dataset
        # model[infer_test_split]['df_infer_scaled'] = ForecastingModel(modeling).fit(df)
        model[infer_test_split] = ForecastingModel(modeling).fit(df)

    return model


def _get_scores_cv(gtruth: Dict[str, Any], preds: Dict[str, Any], avg=True):
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


def _get_predictions_e_gtruth_cv(model, cv_splits_positions, df_infer, df_test, scaler):
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


def evaluate_cv(
        model: Any,
        cv_splits_positions: Dict[str, Any],
        df_infer: pd.DataFrame,
        df_test: pd.DataFrame,
        scaler: Any) -> Any:

    gtruth, preds = _get_predictions_e_gtruth_cv(model, cv_splits_positions, df_infer, df_test, scaler)
    scores_nodewise = _get_scores_cv(gtruth, preds, avg=False)
    scores_averaged = _get_scores_cv(gtruth, preds, avg=True)

    return scores_nodewise, scores_averaged


def _get_predictions_e_gtruth(
        model: Dict[int, ForecastingModel],
        df_unscaled: pd.DataFrame,
        inference_test_splits_positions: Dict[int, Dict[str, Any]],
        scaler: Dict[int, Scaler]) -> Tuple[Dict[int, Dict[str, pd.Series]], ...]:

    gtruth = {}
    preds = {}

    targets = model[0].targets

    for split in inference_test_splits_positions.keys():
        gtruth[split] = {}
        preds[split] = {}
        split_slices = inference_test_splits_positions[split]

        for cat in ['infer', 'test']:
            gtruth[split][cat] = df_unscaled[split_slices[cat]][targets]
            preds[split][cat] = model[split].predict(start=gtruth[split][cat].index[0],
                                                     end=gtruth[split][cat].index[-1],
                                                     scaler=scaler[split])

    return gtruth, preds


def _get_scores(
        gtruth: Dict[int, Dict[str, pd.Series]],
        preds: Dict[int, Dict[str, pd.Series]],
        metrics: List[str],
        avg: bool = True) -> pd.DataFrame:

    all_metrics = [metrics_registered[m] for m in metrics]
    all_splits = list( preds.keys() )

    if avg:
        multioutput = 'uniform_average'
    else:
        multioutput = 'raw_values'

    scores = pd.DataFrame(
        data=None,
        index=pd.MultiIndex.from_product([all_metrics, ['infer', 'test']]),
        columns=all_splits
    )

    for split in all_splits:
        for cat in ['infer', 'test']:
            for metric in all_metrics:
                if metric not in ['MdRAE']:
                    try:
                        scores.loc[(metric, cat), split] = metrics_registered[metric](
                            gtruth[split][cat],
                            preds[split][cat],
                            multioutput=multioutput
                        )
                    except Exception as e:
                        logging.debug(f'Exception: {e}')
                else:
                    scores.loc[(metric, cat), split] = metrics_registered[metric](
                        gtruth[split][cat],
                        preds[split][cat],
                        multioutput=multioutput
                    )

    return scores


def evaluate(
        model: Any,
        df_unscaled: Dict[int, pd.DataFrame],
        inference_test_splits_positions: Dict[int, Dict[str, Any]],
        metrics: List[str],
        scaler: Any,
        display_gtruth_vs_pred: bool = True,
        display_error_boxplots: bool = True,
) -> Any:

    gtruth, preds = _get_predictions_e_gtruth(model, df_unscaled, inference_test_splits_positions, scaler)
    if display_gtruth_vs_pred:
        _plot_gtruth_preds(gtruth, preds, node='DEF0C', split=5)

    if display_error_boxplots:
        _plot_error_boxplots(gtruth,
                             preds,
                             nodes=['DEF0C', 'DEF07', 'DEF0B', 'DEF05', 'DEF0E'],
                             split=5)
    scores_nodewise = _get_scores(gtruth, preds, metrics, avg=False)
    scores_averaged = _get_scores(gtruth, preds, metrics, avg=True)

    return scores_nodewise, scores_averaged


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
