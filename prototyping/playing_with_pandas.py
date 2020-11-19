import pandas as pd
from sklearn.datasets import load_iris
import numpy as np


if __name__ == '__main__':
    iris = load_iris()

    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                      columns=iris['feature_names'] + ['target'])

    print(f'Type is: {type(df)}')
    print(df.head(3))

    # [print(type(col)) for col in df]

    print(df[slice(0,4)])
