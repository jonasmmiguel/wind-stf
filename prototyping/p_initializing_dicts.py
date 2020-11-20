if __name__ == '__main__':
    mydict = {{}}
    for i in range(5):
        # mydict[i] = {}
        for j in range(10):
            mydict[i][j] = str(i+j+1)

    print(mydict)