from pmlb import fetch_data
from sklearn.model_selection import train_test_split
import numpy as np
from numpy.random import default_rng, shuffle, permutation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import LGP
import MAP_Elites
import pickle
from matplotlib.colors import Colormap
import time




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


def plot_data(dataX, trueLabels, newLabels, palette=None, legend = False):
    fig, axs = plt.subplots(2,2)

    for x in range(dataX.shape[1]):
        for y in range(x+1, dataX.shape[1]):
            if x == 0:
                sns.scatterplot(ax=axs[x,y-1],x = dataX[:,x], y = dataX[:, y], hue=newLabels, style=trueLabels,
                                palette=palette, legend=legend)
            elif x == 1:
                sns.scatterplot(ax=axs[x, y - 2], x=dataX[:, x], y=dataX[:, y], hue=newLabels, style=trueLabels,
                                palette=palette, legend=legend)
            else:
                sns.scatterplot(ax=axs[x-1, y - 1], x=dataX[:, x], y=dataX[:, y], hue=newLabels, style=trueLabels,
                                palette=palette, legend=legend)

def gen_points(dataX,num_points):
    min1 = np.min(dataX[:,0])
    min2 = np.min(dataX[:,1])
    max1 = np.max(dataX[:, 0])
    max2 = np.max(dataX[:, 1])

    return np.mgrid[min1:max1:complex(imag=np.sqrt(num_points)), min2:max2:complex(imag=np.sqrt(num_points))].reshape(2,-1).T

def plot_models(dataX, models, true_labels, total_columns, palette=None):
    num_plots = len(models)*2
    total_cols = total_columns
    plots = [None] * num_plots
    if num_plots % total_cols == 0:
        total_rows = num_plots // total_cols
    else:
        total_rows = num_plots // total_cols + 1
    fig, axs = plt.subplots(nrows=total_rows, ncols=total_cols,
                            figsize=(5 * total_cols, 5 * total_rows), constrained_layout=True)

    data_span = gen_points(dataX, 10000)

    if total_rows > 1:
        for i, model in enumerate(models):
            row1 = (i*2) // total_cols
            pos1 = (i*2) % total_cols
            row2 = (i*2+1) // total_cols
            pos2 = (i*2+1) % total_cols

            norm = plt.Normalize(min(model.predictions), max(model.predictions))
            sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
            sm.set_array([])

            hue_map = model.predictions
            hue_map[0] = hue_map[0] - 0.000001
            sns.scatterplot(ax=axs[row1, pos1], x=dataX[:, 0], y=dataX[:, 1], hue=hue_map, style=true_labels,
                            palette=palette, legend=False)

            fig.colorbar(sm, ax=axs[row1,pos1])

            model_behaviour = model.predict(data_span)
            norm = plt.Normalize(min(model_behaviour), max(model_behaviour))
            sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
            sm.set_array([])

            hue_map = model_behaviour
            hue_map[0] = hue_map[0] - 0.000001
            sns.scatterplot(ax=axs[row2, pos2], x=data_span[:,0], y=data_span[:, 1], hue=hue_map,
                            palette=palette, legend=False)
            fig.colorbar(sm, ax=axs[row2,pos2])
    else:
        for i, model in enumerate(models):
            pos1 = i*2
            pos2 = i*2+1

            norm = plt.Normalize(min(model.predictions), max(model.predictions))
            sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
            sm.set_array([])

            sns.scatterplot(ax=axs[pos1], x=dataX[:, 0], y=dataX[:, 1], hue=model.predictions, style=true_labels,
                            palette=palette, legend=False)
            fig.colorbar(sm, ax=axs[pos1])


            model_behaviour = model.predict(data_span)
            norm = plt.Normalize(min(model_behaviour), max(model_behaviour))
            sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
            sm.set_array([])

            sns.scatterplot(ax=axs[pos2], x=data_span[:, 0], y=data_span[:, 1], hue=model_behaviour,
                            palette=palette, legend=False)
            fig.colorbar(sm, ax=axs[pos2])

def plot_behaviour(dataX, behaviours, true_labels, total_columns, palette=None):
    num_plots = len(behaviours)
    total_cols = total_columns
    if num_plots % total_cols == 0:
        total_rows = num_plots // total_cols
    else:
        total_rows = num_plots // total_cols + 1
    fig, axs = plt.subplots(nrows=total_rows, ncols=total_cols,
                            figsize=(5 * total_cols, 5 * total_rows), constrained_layout=True)

    if total_rows > 1:
        for i, behaviour in enumerate(behaviours):
            row1 = i // total_cols
            pos1 = i % total_cols

            norm = plt.Normalize(min(behaviour), max(behaviour))
            sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
            sm.set_array([])

            sns.scatterplot(ax=axs[row1, pos1], x=dataX[:, 0], y=dataX[:, 1], hue=behaviour, style=true_labels,
                            palette=palette, legend=False)
            fig.colorbar(sm, ax=axs[row1,pos1])

    else:
        for i, behaviour in enumerate(behaviours):
            pos1 = i

            norm = plt.Normalize(min(behaviour), max(behaviour))
            sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
            sm.set_array([])

            sns.scatterplot(ax=axs[pos1], x=dataX[:, 0], y=dataX[:, 1], hue=behaviour, style=true_labels,
                            palette=palette, legend=False)
            fig.colorbar(sm, ax=axs[pos1])


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

        if int(gen % max((num_generation//50),1)) == 0:
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

def run_cos_cvt(dataX, dataY, num_generation, init_pop_size, num_per_gen, recom_rate, mut_rate, num_niches, rng, palette = None):
    model_param = LGP.Parameters(dataX.shape[1], rng)
    cvt_param = MAP_Elites.Parameters(dataX, dataY, dataX.shape[0], rng, LGP.LGP, model_param)
    cvt_param.pop_init_amount = init_pop_size
    cvt_param.generation_amount = num_per_gen
    cvt_param.recom_rate = recom_rate
    cvt_param.mut_rate = mut_rate
    cvt_param.num_niches = num_niches

    CVT = MAP_Elites.MAPElites(cvt_param)
    CVT.initialize_centroids()
    CVT.initialize_population()

    for gen in range(num_generation):
        CVT.sim_generation()

        if int(gen % max((num_generation // 10),1)) == 0:
            CVT.print_map()
            print(CVT.niche_popularity)
    return CVT


def unison_shuffled_copies(a, b, rng):
    assert len(a) == len(b)
    p = rng.permutation(len(a))
    return a[p], b[p]

def main():
    Cmap = sns.color_palette("viridis", as_cmap=True)
    dataX_iris, dataY_iris = fetch_data('iris', return_X_y=True, local_cache_dir='..\data')
    spiral_data = np.loadtxt('../data/spiral_data.txt')
    dataX_spiral = spiral_data[:,0:2]
    dataY_spiral = spiral_data[:,2]

    labels_iris = combine_labels(dataY_iris)
    dataX_iris = dataX_iris[:,(0,2)]
    testing_data = dataX_iris
    testing_labels = labels_iris[0]


    plot_data(dataX_iris, dataY_iris, labels_iris[0], palette=Cmap) #palette = 'deep'
    plt.show()
    exit()
    st_reg = time.time()
    highest_ind = run_regular_lgp(testing_data, testing_labels,505, 1000, 5, 0.9, 1)
    et_reg = time.time()
    #
    plot_models(testing_data,[highest_ind],testing_labels,2,Cmap)

    print("regular LGP highest individual")
    # highest_ind.print_program() # PUT IN A PRINT FUNCTION FOR LGP
    highest_ind.print_program(effective=True)

    st_cvt = time.time()
    CVT = run_cos_cvt(testing_data, testing_labels, 500, 5000, 1000, 0.9, 1, 6, default_rng(seed=1),palette=Cmap)
    et_cvt = time.time()

    plot_behaviour(testing_data, CVT.gen_centroids, testing_labels, 4, Cmap)

    cvt_models = [x for x in CVT.mapE.values() if x]

    for model in cvt_models:
        model.print_program(effective=True)

    plot_models(testing_data, cvt_models, testing_labels,4,Cmap)


    plt.show()

    print("time taken for regular:", et_reg - st_reg, 'seconds')
    print("time taken for cvt:", et_cvt - st_cvt, 'seconds')



if __name__ == '__main__':
    main()