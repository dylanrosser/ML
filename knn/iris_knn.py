import numpy as np
import pandas as pd
from scipy.stats import mode
import datetime as dt

norm = 1

train = pd.read_csv('train.dat', sep = '_', header=None)
test = pd.read_csv( 'test.dat', sep = '_', header=None )
results = []
for k in range(1, 25):
    num_correct = 0
    for i in test.index:
    #for i in range(1):
        a = np.int(test[1][i], 16)
        b = np.int(test[2][i], 16)
        c = np.int(test[3][i], 16)
        d = np.int(test[4][i], 16)

        ar1 = np.array([a,b,c,d])
        nn = pd.DataFrame(data=None)


        for j in train.index:
                e = np.int(train[1][j], 16)
                f = np.int(train[2][j], 16)
                g = np.int(train[3][j], 16)
                h = np.int(train[4][j], 16)

                ar2 = np.array([e,f,g,h])
                dist = np.linalg.norm(ar1-ar2, ord=norm)

                dic = {'idx' : j, 'dist' : dist , 'label': train[0][j]}
                nn = nn.append(dic, ignore_index=True)

        knn = nn.sort_values('dist').head(k)

        m = mode(knn.label.values)
        if m.count > 1:
            res = int(m.mode)
        else:
            res = int(knn.label.values[0])

        if (res == test[0][i]):
            num_correct = num_correct+1

    accuracy = num_correct/40*100
    print(k, accuracy, '%')
    results.append({'k': k, 'norm': norm, 'accuracy': accuracy})

now = dt.datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
r = pd.DataFrame(results)
r.to_csv("results_"+now+".csv")
