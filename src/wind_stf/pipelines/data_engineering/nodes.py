from typing import List, Any
import pandas as pd
import nvector as nv
import numpy as np
from src.utils.data_engineering import weighted_average
from pandas.core.window.expanding import _Rolling_and_Expanding
from src.utils.geospatial import build_distances_mx
from src.utils.gwnet_utils import get_adj_mx


_Rolling_and_Expanding.weighted_average = weighted_average

def _get_centroid(lat: pd.Series, lon: pd.Series) -> tuple:
    # TODO: currently getting centroid of turbines installed anytime. Get centroid for each day instead!
    # TODO: currently getting geospatial centroid. Get power-weighted centroid instead!
    points = nv.GeoPoint(
        latitude=lat.values,
        longitude=lon.values,
    )
    vectors = points.to_nvector()
    centroid_vector = vectors.mean()
    centroid = centroid_vector.to_geo_point()
    return (centroid.latitude_deg, centroid.longitude_deg)


# node functions
def concatenate(*data_single_years: List[
    pd.DataFrame]) -> pd.DataFrame:  # using inputs of *args type so that any number of dataframes can be used as input
    data_all_years = pd.concat(data_single_years)
    return data_all_years


def aggregate_temporally(data_hourly: pd.DataFrame, target_freq: str = 'D') -> pd.DataFrame:
    data_daily = data_hourly.resample(rule=target_freq).mean()
    return data_daily


def filter_districts(data_all_districts: pd.DataFrame) -> pd.DataFrame:
    blacklist = [
        'DE409', 'DE40C', 'DE403',  # outliers in spatial correlogram
        'DE24C', 'DE266', 'DEA2C',  # zero installed capacity at 2015-01-01
        'DE257',  # having somehow infinity values after CF transformation
    ]
    # TODO: ensure df_spatial includes the same set of districts as the df_temporal
    # TODO: remove districts where any single turbine represents >10% installed capacity
    data_selected_districts = data_all_districts.drop(blacklist, axis='columns')
    return data_selected_districts


def convert_kw_to_capfactor(
        power_generated_daily: pd.DataFrame,
        power_installed_daily: pd.DataFrame,
) -> pd.DataFrame:
    # calculate capacity factor for each day and district
    capfactors_mts = power_generated_daily / power_installed_daily.loc[
        power_generated_daily.index, power_generated_daily.columns]

    # fill missing values with 0.0
    capfactors_mts_filled = capfactors_mts.fillna(value=0.0)

    # assign names for levels in (1) index & (2) columns, for compatibility with spatial data later on
    # capfactors_mts_filled.columns = pd.MultiIndex.from_product(
    #     [capfactors_mts_filled.columns, ['power']],
    #     names=['nuts_id', 'var']
    # )
    capfactors_mts_filled.index.rename('date', inplace=True)

    return capfactors_mts_filled


def _validate_data(
        df_: pd.DataFrame,
        tolerance: dict ={
            'nan': False,
            'inf': False,
            'value_max': 2.0,
            'value_min': 0.0
        },
    display_culprits: bool = False):
    inconsistencies = {'NaNs': df_.isna().sum().sum(),
                       'infs count': np.isinf(df_).sum().sum(),
                       'value<min count': (df_ < tolerance['value_min']).sum().sum(),
                       'value>max count': (df_ > tolerance['value_max']).sum().sum()}

    if any([inconsistencies[check]>0 for check in inconsistencies.keys()]):
        is_valid = False
    else:
        is_valid = True

    inconsistencies_distribution = {'NaNs': df_.isna().sum(),
                'infs': np.isinf(df_).sum(),
                'value<min': (df_ < tolerance['value_min']).sum(),
                'value>max': (df_ > tolerance['value_max']).sum()}

    if display_culprits:
        for culprit in inconsistencies_distribution['value>max'].index:
            print(df_[culprit].describe()[['50%', '75%', 'max']])

    print(f'Data valid? {is_valid}; Length (years): {len(df_) / 365}; Inconsistency: {inconsistencies}')

    return None


def handle_anomalous_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf], 0)
    _validate_data(df)
    return df


def build_power_installed_mts(sensors: pd.DataFrame) -> pd.DataFrame:
    sensors_sortedbydate = sensors.sort_values('commissioning_date')
    powerdeltas_daily_districtwise = sensors_sortedbydate.groupby(by=['nuts_id', 'commissioning_date'], sort=False).agg(
        {'power': 'sum'})
    power_installed = powerdeltas_daily_districtwise.groupby('nuts_id').agg({'power': 'cumsum'})
    power_installed_unstacked = power_installed.unstack(level='nuts_id')
    power_installed_mts = power_installed_unstacked.asfreq('D').ffill().fillna(value=0, axis='rows')
    power_installed_mts = power_installed_mts.droplevel(level=0,
                                                        axis='columns')  # remove irrelevant column level name 'power'
    return power_installed_mts


def build_power_centroids_mts(sensors: pd.DataFrame) -> pd.DataFrame:
    # TODO: define a cummean and use pd.apply
    # TODO: modularize function for improved readability

    # sort dataframe by commissioning date
    sensors_datesorted = sensors.sort_values('commissioning_date')

    # aggregate same-district, same-day entries
    def _power_weighted_average(coord: pd.Series):
        return np.average(coord, weights=sensors_datesorted.loc[coord.index, 'power'])

    sensors_daily_aggregated = sensors_datesorted \
        .groupby(by=['nuts_id', 'commissioning_date'], sort=False) \
        .agg(
             power=('power', 'sum'),
             lat=('lat', _power_weighted_average),
             lon=('lon', _power_weighted_average),
        ) \
        .reset_index() \
        .set_index('commissioning_date')

    # initialize dataframe
    power_centroids_mts = pd.DataFrame(
        index=pd.date_range(
            start=sensors_daily_aggregated.index.min(),
            end=sensors_daily_aggregated.index.max(),
            freq='D',
            name='commissioning_date'
        ),
        columns=pd.MultiIndex.from_product(
            [sensors_daily_aggregated['nuts_id'].unique(), ['lat', 'lon']],
            names=['nuts_id', 'coords'],
        ),
    )

    # fill with available values: coordinates are cumulative means of turbines, power-weighted
    for district in sensors_daily_aggregated['nuts_id'].unique():
        single_district_data = sensors_daily_aggregated[sensors_daily_aggregated['nuts_id'] == district]
        power_centroids_mts[district] = single_district_data[['lat', 'lon']] \
            .expanding() \
            .weighted_average(weights=single_district_data['power'])

    # fill remaining NaN entries
    power_centroids_mts = power_centroids_mts.ffill().bfill()

    # # simplify dataframe
    # districts = list(power_centroids_mts.columns.get_level_values(level=0))
    #
    # power_centroids_mts = pd.DataFrame(
    #     data={d: list(map(tuple, power_centroids_mts[d][['lat', 'lon']].values)) for d in districts},
    #     index=power_centroids_mts.index,
    # )
    return power_centroids_mts
