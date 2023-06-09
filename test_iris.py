from pmlb import fetch_data
from sklearn.model_selection import train_test_split
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import LGP
import MAP_Elites
import pickle

dataX, dataY = fetch_data('iris', return_X_y=True, local_cache_dir='..\data')


def combine_labels(dataY):
    new_labels = []
    uniqueY = np.unique(dataY)
    for class1 in range(uniqueY.size):
        for class2 in range(class1 + 1, uniqueY.size):
            newY = np.copy(dataY)
            newY[newY == class1] = class2
            newY_unique = np.unique(newY)
            newY[newY == newY_unique[0]] = -1
            newY[newY == newY_unique[1]] = 1
            new_labels.append(newY)
    return new_labels


def plot_data(dataX, trueLabels, newLabels):
    fig, axs = plt.subplots(2,3)

    for x in range(dataX.shape[1]):
        for y in range(x+1, dataX.shape[1]):
            if x == 0:
                sns.scatterplot(ax=axs[x,y-1],x = dataX[:,x], y = dataX[:, y], hue=newLabels, style=trueLabels,
                                palette='deep', legend=False)
            elif x == 1:
                sns.scatterplot(ax=axs[x, y - 2], x=dataX[:, x], y=dataX[:, y], hue=newLabels, style=trueLabels,
                                palette='deep', legend=False)
            else:
                sns.scatterplot(ax=axs[x-1, y - 1], x=dataX[:, x], y=dataX[:, y], hue=newLabels, style=trueLabels,
                                palette='deep', legend=False)

def run_regular_lgp(dataX, dataY, num_generation, pop_size):
    pass


def run_cos_cvt():
    pass


def main():

    labels = combine_labels(dataY)
    # for newLabels in labels:
    #     plot_data(dataX, dataY, newLabels)
    #
    # plt.show()

    # testy = LGP.LGP(5)
    #
    # testy.initialize()
    # print(testy.instructions)
    # with open('test_pickle.pkl','wb') as f:
    #     pickle.dump(testy, f)
    #[[4, -3, 2, 3], [7, 0, -2, 4], [1, 0, -3, 1], [9, 11, 1, 1], [8, -7, 4, 0], [7, -2, 13, 3], [2, 6, 1, 0], [9, 10, 3, 3], [0, -3, 2, 4], [2, 9, 9, 1], [1, 2, -9, 0], [3, 8, -6, 2], [6, 14, -5, 2], [5, -5, 11, 1], [6, -3, 6, 4]]
    # with open('test_pickle.pkl', 'rb') as f:
    #     testy = pickle.load(f)
    # print(type(testy))
    rng = default_rng(seed = 1)
    pop = []
    for i in range(100):
        ind = LGP.LGP(5, rng)
        ind.initialize()
        ind.evaluate(np.array([[1,2,3,4,5],[1,1,1,1,1],[-5,-3,2,1,5],[7,8,1,-2,3],[-1,-2,-3,-4,-5]]), np.array([1,-1,1,1,-1]))
        pop.append(ind)

    pop2 = rng.choice(pop, size=10, replace=False)
    for i in pop2:
        print(i.fitness)

    print(pop2)
    pop2 = sorted(pop2,key=lambda x:x.fitness, reverse=True)[0]
    print(pop2)

    # print(testy2.instructions)
    #
    # child1, child2 = testy.recombine_child(testy2, np.array([[1,2,3,4,5]]), np.array([1]))
    #
    # print(child1.instructions)
    # print(child2.instructions)
    #
    # print(testy == testy)
    # print(testy == testy2)
    # print(testy == testy.make_copy())

if __name__ == '__main__':
    main()