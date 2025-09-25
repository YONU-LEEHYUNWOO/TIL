!pip install hdbscan


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as data
import hdbscan


#plt 와 sns Setting
%matplotlib inline
sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0}
plt.rcParams["figure.figsize"] = [9,7]


# Sample Data 만들기
num=100
moons, _ = data.make_moons(n_samples=num, noise=0.01)
blobs, _ = data.make_blobs(n_samples=num, centers=[(-0.75,2.25), (1.0, -2.0)], cluster_std=0.25)
blobs2, _ = data.make_blobs(n_samples=num, centers=[(2,2.25), (-1, -2.0)], cluster_std=0.4)
test_data = np.vstack([moons, blobs,blobs2])
plt.scatter(test_data.T[0], test_data.T[1], color='b', **plot_kwds)
plt.show()

#[HDBSCAN]
#  - Hyperparameter Tuning using for Loop

#[HDBSCAN Parameters]
#  - Packge : https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html
#  - min_cluster : Cluster 안에 적어도 몇개가 있어야 하는지
#  - cluster_selection_epsilon : combining HDBSACN with DBSCAN

minsize = [3, 5, 10, 15, 20, 30]
for m in minsize:
    print("min_cluster_size : {}".format(m))
    db = hdbscan.HDBSCAN(min_cluster_size=m).fit(test_data)
    palette = sns.color_palette()
    cluster_colors = [palette[col]
                    if col >= 0 else (0.5, 0.5, 0.5) for col in
                    db.labels_]
    plt.scatter(test_data.T[0], test_data.T[1], c=cluster_colors, **plot_kwds)
    plt.show()
