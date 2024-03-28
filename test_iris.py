import ctypes
import glob
import shutil
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
import numpy as np
from numpy.random import default_rng, shuffle, permutation
import matplotlib.pyplot as plt
from matplotlib import cm as CM
import seaborn as sns
import pandas as pd
import LGP
import MAP_Elites
import pickle
from matplotlib.colors import Colormap
import time
from k_means_constrained import KMeansConstrained as kmeans
import Population_Evolution as Population
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import set_random_seed
from keras.callbacks import EarlyStopping
from scipy.spatial import distance
import os
import tracemalloc



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

# to plot scatterplot data with different colors and marker
# color should be a list of values with length same as data
# markers should be a list of integers form 0-3 same length as data
def plot_data(x_axis, y_axis, fld_name, plt_name, color=None, markers=None, cmap='viridis', newfig=True, vmin=None, vmax=None, alphas=None):
    # fig, axs = plt.subplots(2,2)
    if newfig:
        fig = plt.figure(figsize=[6.4*3, 4.8*3])
    marker_values = ['o', 'x', '^', 'd']
    if markers is not None:
        for marker in set(markers):
            temp_x_axis = [v for k,v in enumerate(x_axis) if markers[k] == marker]
            temp_y_axis = [v for k,v in enumerate(y_axis) if markers[k] == marker]
            if alphas is not None:
                temp_alpha = [v for k, v in enumerate(alphas) if markers[k] == marker]
            else:
                temp_alpha = alphas
            if type(color) != type(""):
                temp_color = [v for k, v in enumerate(color) if markers[k] == marker]
            else:
                temp_color = color
            plt.scatter(temp_x_axis, temp_y_axis, c=temp_color,cmap=cmap, marker=marker_values[int(marker)], s=72, vmin=vmin, vmax=vmax, alpha=temp_alpha)
    else:
        plt.scatter(x_axis, y_axis, c=color, cmap=cmap, s=72, vmin=vmin, vmax=vmax, alphas=alphas)

    if fld_name and plt_name:
        save_path = fld_name + '/' + plt_name + '.png'
        plt.savefig(save_path, bbox_inches='tight')

    # axs = plt.axes()
    # sns.scatterplot(ax=axs, x=dataX[:, 0], y=dataX[:, 1], hue=newLabels, style=trueLabels, palette=palette, legend=legend)
    # for x in range(dataX.shape[1]):
    #     for y in range(x+1, dataX.shape[1]):
    #         if x == 0:
    #             sns.scatterplot(ax=axs[x,y-1],x = dataX[:,x], y = dataX[:, y], hue=newLabels, style=trueLabels,
    #                             palette=palette, legend=legend)
    #         elif x == 1:
    #             sns.scatterplot(ax=axs[x, y - 2], x=dataX[:, x], y=dataX[:, y], hue=newLabels, style=trueLabels,
    #                             palette=palette, legend=legend)
    #         else:
    #             sns.scatterplot(ax=axs[x-1, y - 1], x=dataX[:, x], y=dataX[:, y], hue=newLabels, style=trueLabels,
    #                             palette=palette, legend=legend)

def plot_model_heatmap(data_span, data_x, data_y, model, fld_name, plt_name, weights=None):
    # plt.figure()
    # min1 = np.min(data_x[:, 0])
    # min2 = np.min(data_x[:, 1])
    # max1 = np.max(data_x[:, 0])
    # max2 = np.max(data_x[:, 1])
    # xx, yy = np.meshgrid(
    #     np.linspace(min1, max1, 1000),
    #     np.linspace(min2, max2, 1000))
    #
    # zz = np.zeros(xx.shape)
    # for i in range(xx.shape[0]):
    #     for j in range(xx.shape[1]):
    #         zz[i, j] = model.predict([[xx[i, j], yy[i, j]]], verbose=0)[0]
    #
    # plt.pcolormesh(xx, yy, zz)
    # cb = plt.colorbar()

    model_behaviour = model.predict(data_span)

    # fig = plt.figure()
    # palette = sns.color_palette("viridis", as_cmap=True)
    #
    #
    # norm = plt.Normalize(min(model_behaviour), max(model_behaviour))
    # sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    # sm.set_array([])
    #
    # hue_map = model_behaviour
    # hue_map[0] = hue_map[0] - 0.000001
    # sns.scatterplot(x=data_span[:, 0], y=data_span[:, 1], hue=hue_map,
    #                 palette=palette, legend=False)
    #
    # fig.colorbar(sm)

    plt.figure(figsize=[6.4*3.5, 4.8*3])

    plt.hexbin(data_span[:,0], data_span[:,1], C=model_behaviour, gridsize=120, bins=None)
    plt.axis([data_span[:,0].min(), data_span[:,0].max(), data_span[:,1].min(), data_span[:,1].max()])
    cb1 = plt.colorbar()
    cb1.set_label('mean value')

    if weights is None:
        plot_data(data_x[:,0], data_x[:,1], None, None, color='red', markers=data_y, newfig=False)
    else:
        if max(weights) > 1:
            alphas = [i/max(weights) for i in weights]
        else: alphas = weights.copy()
        plot_data(data_x[:, 0], data_x[:, 1], None, None, color='red', markers=data_y, newfig=False, alphas=alphas)
        # cb2 = plt.colorbar()
        # cb2.set_label('weight')




    save_path = fld_name + '/' + plt_name + '.png'
    plt.savefig(save_path, bbox_inches='tight')

def gen_points(dataX,num_points):
    min1 = np.min(dataX[:,0]) - 0.5
    min2 = np.min(dataX[:,1]) - 0.5
    max1 = np.max(dataX[:, 0]) + 0.5
    max2 = np.max(dataX[:, 1]) + 0.5

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

def run_regular_lgp(dataX, dataY, num_generation, pop_size, tourney_size, recom_rate, mut_rate, rng):
    params = LGP.Parameters(dataX.shape[1], rng)
    params.operators = [0,1,2,3,4] #setting usable operators (+,-,x,/,<)
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
            if rng.random() < mut_rate:
                child1.mutate()
                child2.mutate()
            elif rng.random() < recom_rate:
                child1.recombine(child2)

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

# finding the points close to the boundary using a complex model (threshold is the % of points to take)
def find_boundary_points(data_x, model, threshold):
    predictions = np.ravel(model.predict(data_x, verbose=0)) #getting predictions
    sorted_predictions = np.argsort([abs(i - 0.5) for i in predictions]) #getting the indices of the sorted predictions
    threshold_val = int(len(predictions) * threshold) #getting how many points to include


    boundary_points = sorted_predictions[:threshold_val] #indices of boundary points
    boundary_points_data = data_x[sorted_predictions[:threshold_val]] #data of boundary points
    return boundary_points, boundary_points_data #returning the data of the boundary points as well

def find_clusters(k_clusters, boundary_data, data_x, rng, size_min = None, size_max = None):
    clf = kmeans(n_clusters=k_clusters, size_min=size_min,size_max=size_max, random_state=rng)

    boundary_clusters = clf.fit_predict(boundary_data) + 1

    clusters = clf.predict(data_x, size_min=None, size_max=None) + 1

    cluster_centers = clf.cluster_centers_

    return boundary_clusters, clusters, cluster_centers

def find_data_weights(chosen_cluster, data_x, clusters, cluster_center, cluster_weight, non_cluster_weight, distance_based=False):
    weights = []
    if distance_based:
        distances = np.ravel(distance.cdist(data_x, [cluster_center], 'euclidean'))
        for i in range(len(clusters)):
            if chosen_cluster == clusters[i]:
                weights.append(distances[i] * cluster_weight)
            else:
                weights.append(distances[i] * non_cluster_weight)
    else:
        for i in range(len(clusters)):
            if chosen_cluster == clusters[i]:
                weights.append(cluster_weight)
            else:
                weights.append(non_cluster_weight)

    return weights

'''Old main from cvt Map Elites'''
# def main():
#     Cmap = sns.color_palette("viridis", as_cmap=True)
#     dataX_iris, dataY_iris = fetch_data('iris', return_X_y=True, local_cache_dir='..\data')
#     spiral_data = np.loadtxt('../data/spiral_data.txt')
#     dataX_spiral = spiral_data[:,0:2]
#     dataY_spiral = spiral_data[:,2]
#
#     labels_iris = combine_labels(dataY_iris)
#     dataX_iris = dataX_iris[:,(0,2)]
#     testing_data = dataX_iris
#     testing_labels = labels_iris[0]
#
#
#     plot_data(dataX_iris, dataY_iris, labels_iris[0], palette=Cmap) #palette = 'deep'
#     plt.show()
#     exit()
#     st_reg = time.time()
#     highest_ind = run_regular_lgp(testing_data, testing_labels,505, 1000, 5, 0.9, 1)
#     et_reg = time.time()
#     #
#     plot_models(testing_data,[highest_ind],testing_labels,2,Cmap)
#
#     print("regular LGP highest individual")
#     # highest_ind.print_program() # PUT IN A PRINT FUNCTION FOR LGP
#     highest_ind.print_program(effective=True)
#
#     st_cvt = time.time()
#     CVT = run_cos_cvt(testing_data, testing_labels, 500, 5000, 1000, 0.9, 1, 6, default_rng(seed=1),palette=Cmap)
#     et_cvt = time.time()
#
#     plot_behaviour(testing_data, CVT.gen_centroids, testing_labels, 4, Cmap)
#
#     cvt_models = [x for x in CVT.mapE.values() if x]
#
#     for model in cvt_models:
#         model.print_program(effective=True)
#
#     plot_models(testing_data, cvt_models, testing_labels,4,Cmap)
#
#
#     plt.show()
#
#     print("time taken for regular:", et_reg - st_reg, 'seconds')
#     print("time taken for cvt:", et_cvt - st_cvt, 'seconds')

'''main for kmeans boundary stuff (using iris data)'''
# def main():
#     rng = default_rng(seed=1)
#
#     Cmap = sns.color_palette("viridis", as_cmap=True)
#     dataX_iris, dataY_iris = fetch_data('iris', return_X_y=True, local_cache_dir='..\data')
#     spiral_data = np.loadtxt('../data/spiral_data.txt')
#     dataX_spiral = spiral_data[:,0:2]
#     dataY_spiral = spiral_data[:,2]
#
#     labels_iris = combine_labels(dataY_iris)
#     dataX_iris = dataX_iris[:,(0,2)]
#     testing_data = dataX_iris
#     testing_labels = labels_iris[0]
#
#     model_param = LGP.Parameters(dataX_iris.shape[1], rng)
#     model_param.operators = [0, 1, 2, 3, 4]
#     population_param = Population.Parameters(rng, LGP.LGP, model_param, testing_data, testing_labels)
#
#     population = Population.Population(population_param)
#     population.initialize_run()
#     population.run_evolution(10)
#     highest_ind = population.return_best()
#
#     plot_data(dataX_iris, dataY_iris, labels_iris[0], palette=Cmap) #palette = 'deep'
#
#     # st_reg = time.time()
#     # highest_ind = run_regular_lgp(testing_data, testing_labels,100, 1000, 5, 0.9, 1, rng)
#     # et_reg = time.time()
#     # print("time taken for regular:", et_reg - st_reg, 'seconds')
#
#     plot_models(testing_data, [highest_ind], testing_labels, 2, Cmap)
#
#     boundary_points_ind, boundary_points_data, = find_boundary_points(dataX_iris, highest_ind, 0.3)
#
#     boundary_coloring = np.zeros(len(testing_labels), dtype=int)
#     boundary_coloring[boundary_points_ind] = 1
#
#     plot_data(dataX_iris, labels_iris[0], boundary_coloring)
#
#     boundary_clusters, clusters = find_clusters(boundary_points_data, dataX_iris, rng=0)
#
#     boundary_coloring[boundary_points_ind] = boundary_clusters
#
#
#     plot_data(dataX_iris, labels_iris[0], boundary_coloring, palette=Cmap)
#
#     plot_data(dataX_iris, labels_iris[0], clusters, palette=Cmap)
#
#     plt.show()

'''main for kmeans boundary stuff (using synthetic data)'''
# def main():
#     # setting rng for run
#     rng = default_rng(seed=1)
#
#     # choosing what dataset to use
#     dataset_name = 'dataset_02.pkl'
#     fld_dataset_name = dataset_name.removesuffix('.pkl')
#     fld_dataset_path = "output/" + fld_dataset_name
#
#     # choosing whether to remove previous runs and data of dataset
#     rmv_prev_dataset_data = False
#
#     # if we want to overwrite the previous run with this one
#     overwrite_prev_run = True
#
#     try:
#         os.makedirs(fld_dataset_path)
#     except FileExistsError:
#         if rmv_prev_dataset_data:
#             shutil.rmtree(fld_dataset_path)
#             os.makedirs(fld_dataset_path)
#
#
#     # loading the pickled dataset
#     with open("data/" + dataset_name, 'rb') as f:
#         data = pickle.load(f)
#
#     # splitting data into variables and target
#     data_x = data[:,0:-1]
#     data_y = data[:,-1]
#     num_samples, num_var = data_x.shape
#     data_span = gen_points(data_x, 50000) #used for heatmaps
#     #plotting the raw data
#     if rmv_prev_dataset_data:
#         plot_data(data_x[:,0], data_x[:,1], fld_dataset_path, 'raw_data', color=data_y)
#
#
#     # the run parameters
#     percentage_boundary_points = 0.4 #percent of points to choose when selecting boundary points
#     amount_boundary_points = int(len(data_y) * percentage_boundary_points)
#     cluster_values = [2] # different cluster values to test
#     # possible [cluster, non-cluster] weights to choose from
#     weight_pairs = [
#         [1, 0],  # only points in cluster are relevant
#         # [1, 1],  # equivalent to ensemble on all data
#         # [0.9, 0.1],
#         # [0.7, 0.3]
#     ]
#
#     distance_based = False #whether weights are determined by point distance to clusters
#
#     # minimum and maximum sizes of clusters
#     min_cluster_size_percent = 0.7 # percent of boundary points that must be evenly distributed among the clusters
#     cluster_size_min = [int((min_cluster_size_percent/ k) * amount_boundary_points) for k in cluster_values]
#     cluster_size_max = [None for k in cluster_values]
#
#     for i in range(len(weight_pairs)):
#         chosen_weight = i
#         run_parameters = {
#             'percentage_boundary_points': percentage_boundary_points,
#             'amount_boundary_points': amount_boundary_points,
#             'cluster_values': cluster_values,
#             'min_cluster_size_percent': min_cluster_size_percent,
#             'cluster_size_min': cluster_size_min,
#             'cluster_size_max': cluster_size_max,
#             'weight_pairs': weight_pairs,
#             'chosen_weight': chosen_weight,
#             'distance_based': distance_based
#         }
#         if len(glob.glob('run*', root_dir=fld_dataset_path)) == 0:
#             fld_run_name = 'run_01'
#         else:
#             if overwrite_prev_run:
#                 fld_run_name = sorted(glob.glob('run*', root_dir=fld_dataset_path), reverse=True, key=lambda x: int(x[-2:]))[0]
#             else:
#                 fld_run_name = 'run_{:02d}'.format(int(
#                     sorted(glob.glob('run*', root_dir=fld_dataset_path), reverse=True, key=lambda x: int(x[-2:]))[0][-2:]) + 1)
#         # run folder path
#         fld_run_path = fld_dataset_path + '/' + fld_run_name
#
#         # creating the folder or deleting the previous one
#         try:
#             os.makedirs(fld_run_path)
#         except FileExistsError:
#             shutil.rmtree(fld_run_path)
#             os.makedirs(fld_run_path)
#
#         with open(fld_run_path + '/run_details.txt', 'w') as f:
#             for k, v in run_parameters.items():
#                 print('{}: \t {}'.format(k, v), file=f)
#
#
#         # Building a complex model to find the decision boundary
#         set_random_seed(1)
#         model = Sequential()
#         model.add(Dense(128, activation='relu'))  # Input layer with 2 features, 32 neurons, ReLU activation
#         model.add(Dense(64, activation='relu'))  # Hidden layer with 16 neurons, ReLU activation
#         model.add(Dense(32, activation='relu'))  # Hidden layer with 16 neurons, ReLU activation
#         model.add(
#         Dense(1, activation='sigmoid'))  # Output layer with 1 neuron, sigmoid activation for binary classification
#
#         # adding an early stopping criteria for the model
#         # if want to use, add 'callbacks=[callback]' to model.fit
#         # callback = EarlyStopping(monitor='accuracy', patience=20, start_from_epoch=50)
#         # Compile the model
#         model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#         # Train the model
#         model.fit(data_x, data_y, epochs=1000, batch_size=32)
#
#         # plotting the complex model heatmap
#         plot_model_heatmap(data_span, data_x, data_y, model, fld_run_path, 'complex_model_heatmap')
#
#         # saving complex model information for specific run
#         with open(fld_run_path + '/complex_model_code.txt', 'w') as f:
#             f.write('''
#         # Building a complex model to find the decision boundary
#         set_random_seed(1)
#         model = Sequential()
#         model.add(Dense(128, activation='relu'))  # Input layer with 2 features, 32 neurons, ReLU activation
#         model.add(Dense(64, activation='relu'))  # Hidden layer with 16 neurons, ReLU activation
#         model.add(Dense(32, activation='relu'))  # Hidden layer with 16 neurons, ReLU activation
#         model.add(
#         Dense(1, activation='sigmoid'))  # Output layer with 1 neuron, sigmoid activation for binary classification
#
#         # adding an early stopping criteria for the model
#         # if want to use, add 'callbacks=[callback]' to model.fit
#         # callback = EarlyStopping(monitor='accuracy', patience=20, start_from_epoch=50)
#         # Compile the model
#         model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#         # Train the model
#         model.fit(data_x, data_y, epochs=1000, batch_size=32)
#         '''
#         )
#
#
#
#         # finding the data points along to boundary
#         boundary_points_ind, boundary_points_data, = find_boundary_points(data_x, model, percentage_boundary_points)
#
#         # plotting the points along the boundary
#         boundary_coloring = np.zeros(len(data_y), dtype=int)
#         boundary_coloring[boundary_points_ind] = 1
#
#         plot_data(data_x[:, 0], data_x[:, 1], fld_run_path, 'boundary_points', color=boundary_coloring, markers=data_y)
#
#
#
#
#         cluster_weight, non_cluster_weight = weight_pairs[chosen_weight]
#
#         # going through each k value and creating models
#         for cluster_val in range(len(cluster_values)):
#             k = cluster_values[cluster_val]
#             size_min = cluster_size_min[cluster_val]
#             size_max = cluster_size_max[cluster_val]
#             # making sure there is a folder for the k value to save the output
#             k_fld_path = fld_run_path + '/' + repr(k) + '_cluster'
#             try:
#                 os.makedirs(k_fld_path)
#             except FileExistsError:
#                 pass
#
#             # finding the boundary and full clusters according the k value
#             boundary_clusters, clusters, cluster_centers = find_clusters(k, boundary_points_data, data_x, size_min=size_min,size_max=size_max, rng=0)
#
#             # plotting the boundary and full clusters
#             boundary_coloring[boundary_points_ind] = boundary_clusters
#             plot_data(data_x[:, 0], data_x[:, 1], k_fld_path, 'boundary_clusters', color=boundary_coloring, markers=data_y)
#             plot_data(data_x[:, 0], data_x[:, 1], k_fld_path, 'full_clusters', color=clusters, markers=data_y)
#
#             # setting up the lgp models to be used on each cluster
#             model_param = LGP.Parameters(num_var, rng)
#             model_param.operators = [0, 1, 2, 3, 4] #limiting operators to +,-,*,/
#             model_param.init_length = list(range(5,11))
#             model_param.max_length = 25
#             model_param.min_length = 4
#             model_param.max_dc = 5 # maximum crossover distance
#             # need to fix effective stuff
#             # (initialization can only work with input sep = false, macro mut should just run
#             #   intron itself instead of keeping hold of eff reg)
#             # model_param.effective_mutation = True
#             # model_param.effective_initialization = True
#             # model_param.effective_recombination = True
#
#             with open(k_fld_path + '/LGP_model_param.txt', 'w') as f:
#                 model_param.print_attributes(file=f)
#
#
#             # building a model for each cluster
#             for cluster in set(clusters):
#                 # cluster_data_x = data_x[clusters == cluster]
#                 # cluster_data_y = data_y[clusters == cluster]
#                 cluster_center = cluster_centers[cluster-1]
#                 cluster_weights = find_data_weights(cluster,data_x,clusters,cluster_center,cluster_weight,non_cluster_weight,distance_based)
#
#                 population_param = Population.Parameters(rng, LGP.LGP, model_param, data_x, data_y, weights=cluster_weights)
#                 population_param.recomb_rate = 0.5
#                 #population_param.num_eval_per_gen = 5
#                 population = Population.Population(population_param)
#                 population.initialize_run()
#                 with open(k_fld_path + '/Population_{}_model_param.txt'.format(cluster), 'w') as f:
#                     population_param.print_attributes(file=f)
#
#
#                 population.run_evolution(500)
#
#                 with open(k_fld_path + '/model_' + repr(cluster) + '.txt', 'w') as f:
#                     highest_ind = population.return_best()
#                     highest_ind.print_program(effective=True, file=f)
#                 plot_model_heatmap(data_span, data_x, data_y, highest_ind, k_fld_path, 'model_' + repr(cluster) + '_heatmap', weights=cluster_weights)
#
#     # plt.show()

'''main for MOCK clustering stuff (using new and improved synthetic data)'''
def main():

def test():
    rng = default_rng(seed=1)
    var = LGP.Parameters(2, rng)

    print(var)

    var2 = LGP.LGP(var)

    print(var2)
    print(ctypes.cast(id(var2), ctypes.py_object))
    location = id(var2)
    print(location)

    var2.initialize(np.array([[1,2],[3,4]]),np.array([1,0]))
    var3 = var2.make_copy()
    var4 = LGP.LGP(var)
    print(id(var2))
    a = ctypes.cast(location, ctypes.py_object).value
    # print(var3)
    # print(var3.param)
    print(ctypes.cast(id(var2), ctypes.py_object))
    print(a)




if __name__ == '__main__':
    # test()
    main()