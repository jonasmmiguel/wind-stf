import numpy as np

if __name__=='__main__':
    with np.load('/home/jonasmmiguel/Documents/learning/graphwavenet/data/wind/train.npz') as wind:
        x_wind = wind['x']
        y_wind = wind['y']
        yo_wind = wind['y_offsets']
        xo_wind = wind['x_offsets']

    with np.load('/home/jonasmmiguel/Documents/learning/graphwavenet/data_metr-la/METR-LA/train.npz') as metr:
        x_metr = metr['x']
        y_metr = metr['y']
        yo_metr = metr['y_offsets']
        xo_metr = metr['x_offsets']

    # with np.load('/home/jonasmmiguel/Documents/learning/graphwavenet/data/wind/train.npz') as data:
    #     wind = data['wind']
    #
    # wind = np.load('')
    # metr = np.load('/home/jonasmmiguel/Documents/learning/graphwavenet/data_metr-la/METR-LA/train.npz')
    #
    print('done')
