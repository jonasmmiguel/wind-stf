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
    split_modinfer_test,
    define_cvsplits,
    scale,
    train,
    report_scores,
)


def create_pipeline(**kwargs):
    return Pipeline(
        {
            node(
                func=split_modinfer_test,
                name=r'Split Inference-Test',
                inputs=['capacity_factors_daily_2000to2015', 'params:modeling'],
                outputs=['df_infer', 'df_test'],
            ),
            node(
                func=scale,
                name=r'Scale',
                inputs=['df_infer', 'params:modeling'],
                outputs=['df_infer_scaled', 'scaler'],
            ),
            # node(
            #     func=define_cvsplits,
            #     name=r'Define CV Splits',
            #     inputs='params:cv',
            #     outputs='cv_splits_dict',
            # ),

            # node(
            #     func=train,
            #     name=r'Train Model',
            #     inputs=['df_spatiotemporal_preprocessed',
            #             'cv_splits_dict',
            #             'params:modeling'],
            #     outputs='model',
            # ),
            # # node(
            # #     func=predict,
            # #     name=r'Predict',
            # #     inputs=['df_spatiotemporal', 'cv_splits_dict', 'transformation_parameters'],
            # #     outputs=['train_y_hat', 'test_y_hat'],
            # # ),
            # node(
            #     func=report_scores,
            #     name=r'Report Scores',
            #     inputs=['scoreboard', 'model_metadata', 'cv_splits_dict'],
            #     outputs=None,  # updates scoreboard
            # ),
        }
    )
