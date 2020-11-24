import pandas as pd
import yaml

if __name__ == '__main__':
    cf = pd.read_hdf('/home/jonasmmiguel/Documents/learning/poli/thesis/wind-stf/data/04_feature/capacity_factors_daily_2000to2015_noanomaly.hdf', 'df')
    with open(r'../conf/base/parameters.yml') as file:
        pars = yaml.load(file, Loader=yaml.FullLoader)
    targets = pars['modeling']['targets']
    # targets_new = [col for col in cf.columns if col[:3] in ['DEF', 'DE8', 'DE4']]

    print(targets)

    cf = cf[slice('2005-01-01', '2015-12-31')]
    cf[targets].to_hdf('/home/jonasmmiguel/Documents/learning/poli/thesis/wind-stf/data/04_feature/capacity_factors_daily_2005to2015_noanomaly_targets.hdf', 'w')