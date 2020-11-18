if __name__ == '__main__':
    y = pd.DataFrame(
        {
            'y1': [100, 101, 102, 103, 104],
            'y2': [200, 201, 202, 203, 204],
            'y3': [300, 301, 302, 303, 304],
         },
        index=pd.date_range(start='2015-01-01', end='2015-01-05', freq='D')
    )

    myarray = np.median(y, axis=0, keepdims=True)

    timerange = ['2015-01-06', '2015-01-10']
    freq = y.index.freq
    index = pd.date_range(start=timerange[0], end=timerange[1], freq=freq)

    df = pd.DataFrame(np.tile(myarray, reps=[len(index), 1]), columns=y.columns, index=index)
    print(df)