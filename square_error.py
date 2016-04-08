#! -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame

from numpy.random import normal

def create_dataset(tsv):

    data_set = DataFrame(columns=["x", "t"])

    data = sp.genfromtxt(tsv, delimiter=",")
    for d in data:
        data_set = data_set.append(Series(d, index=["x", "t"]),ignore_index=True)

    return data_set

def least_square(dataset, n):
    """
    最小二乗法で解を求める
    """
    # phi を求める
    phi = DataFrame()
    for i in range(0,n+1):
        p = dataset.x**i
        p.name = "x**{0}".format(i)
        phi = pd.concat([phi,p], axis=1)

    w = np.dot(np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T), dataset.t)

    def f(x):
        y = 0
        for i, wt in enumerate(w):
            y += wt * (x ** i)
        return y

    return (f, w)

if __name__ == '__main__':

    # 訓練データ作成
    train_data = create_dataset("./access_data.csv")
    # テストデータ作成
    test_data = create_dataset("./access_data.csv")

    N = 5

    (f, w) = least_square(train_data, N)

    plt.scatter(test_data.x, f(test_data.x))
    plt.scatter(test_data.x, test_data.t)
    plt.show()
