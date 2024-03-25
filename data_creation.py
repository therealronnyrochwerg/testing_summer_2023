import numpy as np
import pickle
import matplotlib.pyplot as plt


def piecewise_lin_boundary(x_range=(0,1), y_range=(0,1), partitions=3, rng=np.random.default_rng()):
    chosen_points = [(x_range[0], rng.uniform(y_range[0], y_range[1]))]
    partition_points = np.linspace(x_range[0], x_range[1], num=partitions)
    partition_ranges_x = [(partition_points[i], partition_points[i + 1]) for i in range(partitions-1)]
    partition_ranges_x.append((x_range[1], x_range[1]))
    prev_slope = None
    current_partition = 0
    while len(chosen_points) <= partitions:
        prev_point = chosen_points[-1]
        current_partition_range = partition_ranges_x[current_partition]
        next_point_x = rng.uniform(current_partition_range[0], current_partition_range[1])
        next_point_y = rng.uniform(y_range[0], y_range[1])
        current_slope = round((next_point_y - prev_point[1]) / (next_point_x - prev_point[0]), ndigits=1)

        while prev_slope is not None and prev_slope - 0.5 <= current_slope <= prev_slope + 0.5:
            next_point_x = rng.uniform(current_partition_range[0], current_partition_range[1])
            next_point_y = rng.uniform(y_range[0], y_range[1])
            current_slope = round((next_point_y - prev_point[1]) / (next_point_x - prev_point[0]), ndigits=1)

        chosen_points.append((next_point_x, next_point_y))
        prev_slope = current_slope
        current_partition += 1
    return chosen_points

def create_data(change_points, data_per_partition, y_range=(0,1), rng=np.random.default_rng()):
    data = []
    for current_partition in range(len(data_per_partition)):
        data_class_0 = data_per_partition[current_partition][0]
        data_class_1 = data_per_partition[current_partition][1]
        points_class_0 = 0
        points_class_1 = 0
        start_point = change_points[current_partition]
        end_point = change_points[current_partition+1]
        slope = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])
        b = start_point[1] - (slope * start_point[0])
        while points_class_0 < data_class_0:
            x = rng.uniform(start_point[0], end_point[0])
            y = rng.uniform(y_range[0], slope*x + b)
            data.append([x,y,0])
            points_class_0 += 1
        while points_class_1 < data_class_1:
            x = rng.uniform(start_point[0], end_point[0])
            y = rng.uniform(slope*x + b, y_range[1])
            data.append([x,y,1])
            points_class_1 += 1
    return np.array(data)

        #add datapoints to the partition until reaching the number needed, do it class by class


def main():
    rng = np.random.default_rng(seed=3)
    change_points = piecewise_lin_boundary(rng=rng)
    data_per_partition = [[25,25],[25,25],[25,25]]
    data = create_data(change_points, data_per_partition, rng=rng)

    plt.plot([x[0] for x in change_points], [y[1] for y in change_points])
    plt.scatter(data[:,0], data[:,1], c=data[:,2])


    # plt.plot([x[0] for x in change_points], [y[1] for y in change_points])
    # plt.show()
    #
    dataset_name = 'dataset_02'
    save_path = "data" + '/' + dataset_name + '.png'
    plt.savefig(save_path, bbox_inches='tight')
    with open("data/" + dataset_name + ".pkl", 'wb') as f:
        pickle.dump(data, f)



if __name__ == '__main__':
    main()

#
# rng = np.random.default_rng(seed=2)
#
# x1 = rng.uniform(0,10,size=100)
# x2 = rng.uniform(0,10,size=100)
#
# target = []
# for i in range(len(x1)):
#     t_x1 = x1[i]
#     t_x2 = x2[i]
#     if t_x1 <=5:
#         if -(t_x1-3)**2 + 7 - t_x2 <= 0:
#             target.append(0)
#         else:
#             target.append(1)
#     else:
#         if t_x1 -2 - t_x2 <= 0:
#             target.append(0)
#         else:
#             target.append(1)
#
# target = np.asarray(target)
# plt.scatter(x1,x2, c=target)
# plt.show()
#
#
#
# dataset = np.column_stack([x1,x2,target])
# with open("data/dataset_01.pkl", 'wb') as f:
#     pickle.dump(dataset, f)
