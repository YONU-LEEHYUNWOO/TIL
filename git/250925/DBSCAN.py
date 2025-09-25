import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as data
from sklearn.cluster import DBSCAN

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


#[DBSCAN]
#  - Hyperparameter Tuning using for Loop

#[DBSCAN Parameters]
#  - Packge : https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
#  - eps : 이웃을 판단하는 거리
#  - metric : 거리를 계산할 때 사용하는 방법
#    - default : euclidean
#  - min_samples : eps안에 적어도 몇개 들어와야 하는지 이웃의 숫자


epsilon = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
minPls = [5, 10, 15, 20]

for e in epsilon:
    for m in minPls:
        print("epsilon : {}, minPls : {}".format(e, m))
        db = DBSCAN(eps=e, min_samples=m).fit(test_data)
        palette = sns.color_palette()
        cluster_colors = [palette[col]
                        if col >= 0 else (0.5, 0.5, 0.5) for col in
                        db.labels_]
        plt.scatter(test_data.T[0], test_data.T[1], c=cluster_colors, **plot_kwds)
        plt.show()








