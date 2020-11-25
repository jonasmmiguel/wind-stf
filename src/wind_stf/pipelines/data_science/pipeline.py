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

from kedro.pipeline import Pipeline, node

from .nodes import (
    define_inference_test_splits,
    scale,
    scale_,
    train,
    evaluate,
    evaluate_in_kW,
    report_scores,
    test_nbeats
)


def create_pipeline(**kwargs):
    return Pipeline(
        {
            node(
                func=define_inference_test_splits,
                name=r'Define Inference-Test Splits',
                inputs=['params:modeling'],
                outputs='inference_test_splits_positions',
            ),
            node(
                func=scale,
                name=r'Scale',
                inputs=['capacity_factors_daily_noanomaly',
                        'params:modeling',
                        'inference_test_splits_positions'],
                outputs=['df_infer_scaled', 'scaler'],
            ),
            # node(
            #     func=scale_,
            #     name=r'Scale_',
            #     inputs=['capacity_factors_daily_2005to2015_noanomaly_targets',
            #             'params:modeling',
            #             ],
            #     outputs=['capacity_factors_daily_2005to2015_noanomaly_targets_scaled', 'scaler_'],
            # ),
            # node(
            #     func=define_cvsplits,
            #     name=r'Define CV Splits Positions',
            #     inputs=['params:cv', 'df_infer_scaled'],
            #     outputs='cv_splits_positions',
            # ),
            node(
                func=train,
                name=r'Train',
                inputs=['df_infer_scaled',
                        'params:modeling',
                        # 'params:cv',
                        ],
                outputs='model',
            ),
            # node(
            #     func=test_nbeats,
            #     name=r'TEST',
            #     inputs=['df_infer_scaled'],
            #     outputs=None,
            # ),
            node(
                func=evaluate,
                name=r'Evaluate',
                inputs=['model',
                        'capacity_factors_daily_noanomaly',
                        'inference_test_splits_positions',
                        'params:evaluation',
                        'scaler'],
                outputs=['scores_nodewise', 'scores_averaged'],
            ),
            # node(
            #     func=evaluate_in_kW,
            #     name=r'Evaluate in kW',
            #     inputs=['model',
            #             'capacity_factors_daily_noanomaly',
            #             'inference_test_splits_positions',
            #             'params:evaluation',
            #             'scaler',
            #             'power_installed'],
            #     outputs=['scores_nodewise_kW', 'scores_averaged_kW'],
            # ),
            # node(
            #     func=update_scoreboard,
            #     name=r'Report Scores',
            #     inputs=['scoreboard', 'model_metadata', 'cv_splits_dict'],
            #     outputs=None,  # updates scoreboard
            # ),
        }
    )
