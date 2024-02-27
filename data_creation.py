import numpy as np
import pickle
import matplotlib.pyplot as plt


rng = np.random.default_rng(seed=2)

x1 = rng.uniform(0,10,size=100)
x2 = rng.uniform(0,10,size=100)

target = []
for i in range(len(x1)):
    t_x1 = x1[i]
    t_x2 = x2[i]
    if t_x1 <=5:
        if -(t_x1-3)**2 + 7 - t_x2 <= 0:
            target.append(0)
        else:
            target.append(1)
    else:
        if t_x1 -2 - t_x2 <= 0:
            target.append(0)
        else:
            target.append(1)

target = np.asarray(target)   
plt.scatter(x1,x2, c=target)
plt.show()

dataset = np.column_stack([x1,x2,target])
with open("data/dataset_01.pkl", 'wb') as f:
    pickle.dump(dataset, f)
