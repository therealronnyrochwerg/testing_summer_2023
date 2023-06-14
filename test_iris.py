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
from matplotlib.colors import Colormap




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


def plot_data(dataX, trueLabels, newLabels, palette=None, legened = False):
    fig, axs = plt.subplots(2,3)

    for x in range(dataX.shape[1]):
        for y in range(x+1, dataX.shape[1]):
            if x == 0:
                sns.scatterplot(ax=axs[x,y-1],x = dataX[:,x], y = dataX[:, y], hue=newLabels, style=trueLabels,
                                palette=palette, legend=legened)
            elif x == 1:
                sns.scatterplot(ax=axs[x, y - 2], x=dataX[:, x], y=dataX[:, y], hue=newLabels, style=trueLabels,
                                palette=palette, legend=legened)
            else:
                sns.scatterplot(ax=axs[x-1, y - 1], x=dataX[:, x], y=dataX[:, y], hue=newLabels, style=trueLabels,
                                palette=palette, legend=legened)

def run_regular_lgp(dataX, dataY, num_generation, pop_size, tourney_size, recom_rate, mut_rate):
    rng = default_rng(seed = 1)
    params = LGP.Parameters(dataX.shape[1], rng)
    population = []
    # initialization of population
    for name in range(pop_size):
        ind = LGP.LGP(params)
        ind.initialize(dataX, dataY, name = name)
        population.append(ind)

    for gen in range(num_generation):

        for _ in range(pop_size//2):
            winners, losers = LGP.tourney_selection(population,tourney_size, rng)
            child1 = population[winners[0]].make_copy()
            child2 = population[winners[1]].make_copy()
            if rng.random() < recom_rate:
                child1.recombine(child2)
            elif rng.random() < mut_rate:
                child1.mutate()
                child2.mutate()

            child1.evaluate(dataX, dataY)
            child2.evaluate(dataX, dataY)
            population[losers[0]] = child1
            population[losers[1]] = child2

        highest = sorted([x.fitness for x in population])[-1]
        average = np.mean([x.fitness for x in population])
        median = np.median([x.fitness for x in population])

        if int(gen % (num_generation//50)) == 0:
            print("finished generation: {}, fitness: highest {}, average {}, median {} \n".format(
                  gen, highest, average, median))

        if highest == 1:
            print("finished generation: {}, fitness: highest {}, average {}, median {} \n".format(
                  gen, highest, average, median))
            return sorted(population, key=lambda x: x.fitness)[-1]
        # print("highest fitness is: ", sorted(population, key=lambda x: x.fitness)[-1])
        # print("average fitness is: ", np.mean([x.fitness for x in population]))
        # print("median fitness is: ", np.median([x.fitness for x in population]))
    return sorted(population, key=lambda x: x.fitness)[-1]

def run_cos_cvt():
    pass


def main():
    Cmap = sns.color_palette("viridis", as_cmap=True)
    dataX, dataY = fetch_data('iris', return_X_y=True, local_cache_dir='..\data')

    labels = combine_labels(dataY)

    plot_data(dataX, dataY, labels[2], palette=Cmap) #palette = 'deep'

    highest_ind = run_regular_lgp(dataX, labels[2], 500, 500, 5, 0.9, 1)

    print("regular LGP highest individual")
    highest_ind.print_program() # PUT IN A PRINT FUNCTION FOR LGP
    highest_ind.print_program(effective=True)

    plot_data(dataX, dataY, highest_ind.predictions, palette=Cmap, legened=True)

    plt.show()
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


if __name__ == '__main__':
    main()