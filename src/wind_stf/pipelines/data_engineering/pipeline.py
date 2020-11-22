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
    concatenate,
    aggregate_temporally,
    filter_districts,
    convert_kw_to_capfactor,
    build_power_installed_mts,
    handle_anomalous_values,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=concatenate,
                name="Concatenate",
                inputs=[
                    "measurements_hourly_2000",
                    "measurements_hourly_2001",
                    "measurements_hourly_2002",
                    "measurements_hourly_2003",
                    "measurements_hourly_2004",
                    "measurements_hourly_2005",
                    "measurements_hourly_2006",
                    "measurements_hourly_2007",
                    "measurements_hourly_2008",
                    "measurements_hourly_2009",
                    "measurements_hourly_2010",
                    "measurements_hourly_2011",
                    "measurements_hourly_2012",
                    "measurements_hourly_2013",
                    "measurements_hourly_2014",
                    "measurements_hourly_2015",
                ],
                outputs="measurements_hourly_2000to2015",
            ),
            node(
                func=aggregate_temporally,
                name="Downsample",
                inputs="measurements_hourly_2000to2015",
                outputs="measurements_daily_2000to2015",
            ),
            node(
                func=filter_districts,
                name="Filter Districts",
                inputs="measurements_daily_2000to2015",
                outputs="measurements_daily_2000to2015_filtered",
            ),
            # node(
            #     func=build_power_centroids_mts,
            #     name="Get Power Centroids Positions TSs",
            #     inputs="sensors",
            #     outputs="centroids_positions",
            # ),
            node(
                func=build_power_installed_mts,
                name="Get Power Installed TSs",
                inputs="sensors",
                outputs="power_installed",
            ),
            node(
                func=convert_kw_to_capfactor,
                name="Transform kW to CF",
                inputs=["measurements_daily_2000to2015_filtered", "power_installed"],
                outputs="capacity_factors_daily_2000to2015",
            ),
            node(
                func=handle_anomalous_values,
                name="Prepare",
                inputs=["capacity_factors_daily_2000to2015"],
                outputs="capacity_factors_daily_2000to2015_noanomaly",
            ),
        ]
    )
