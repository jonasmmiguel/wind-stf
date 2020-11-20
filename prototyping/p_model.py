import pickle

if __name__ == '__main__':
    with open(r'/home/jonasmmiguel/Documents/learning/poli/thesis/wind-stf/data/06_models/model.pkl/2020-11-20T22.58.20.015Z/model.pkl',
              'rb') as f:
        model = pickle.load(f)

    type(model[0].model)
    print('done!')


