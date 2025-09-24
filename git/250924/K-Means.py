import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

#loading the iris dataset
data = load_iris()

print(data['DESCR'])

# X's & Y Split
Y = pd.DataFrame(data['target'], columns = ['Target'])
X = pd.DataFrame(data['data'], columns = data['feature_names'])

X.shape

data= pd.concat([X, Y], axis=1)

data.shape

data

#[Clustering 전 Scaling]
#  - Clustering은 Distance를 구하는 작업이 필요함
# - Feature들의 Scale이 다르면 Distance를 구하는데 가중치가 들어가게 됨
#  - 따라서, Distance 기반의 Clustering의 경우 Scaling이 필수


# Scaling
scaler = MinMaxScaler().fit(X)
X_scal = scaler.transform(X)
X_scal = pd.DataFrame(X_scal, columns=X.columns)


pca = PCA(n_components=2).fit(X)
X_PCA = pca.fit_transform(X)
X_EMM = pd.DataFrame(X_PCA, columns=['AXIS1','AXIS2'])
print(">>>> PCA Variance : {}".format(pca.explained_variance_ratio_))




# - K-means는 Step2에서 '초기 중심점 설정'이라는 작업을 하는데, 초기 중심점을 셋팅하는 것에 따라 군집의 Quality가 달라짐
# - 따라서 여러번 시도해 보는것 
# - default = 10
# - max_iter : 몇번 Round를 진행할 것 인지
# - Round
[O# - Step 4: 중심점 재설정
# - Step 5: 데이터를 군집에 재할당
# - 이러한 Round를 최대 몇번까지 돌것인가?
[I# - default = 300
# - 300번 안에 중심점 움직임이 멈추지 않으면 그냥 STOP





# K-means Modeling
for cluster in list(range(2, 6)):
    Cluster = KMeans(n_clusters=cluster).fit(X_scal)
    labels = Cluster.predict(X_scal)

    # label Add to DataFrame
    data['{} label'.format(cluster)] = labels
    labels = pd.DataFrame(labels, columns=['labels'])
    # Plot Data Setting
    plot_data = pd.concat([X_EMM, labels], axis=1)
    groups = plot_data.groupby('labels')

    mar = ['o', '+', '*', 'D', ',', 'h', '1', '2', '3', '4', 's', '<', '>']
    colo = ['red', 'orange', 'green', 'blue', 'cyan', 'magenta', 'black', 'yellow', 'grey', 'orchid', 'lightpink']

    fig, ax = plt.subplots(figsize=(10,10))
    for j, (name, group) in enumerate(groups):
        ax.plot(group['AXIS1'], 
                group['AXIS2'], 
                marker=mar[j],
                linestyle='',
                label=name,
                c = colo[j],
                ms=10)
        ax.legend(fontsize=12, loc='upper right') # legend position
    plt.title('Scatter Plot', fontsize=20)
    plt.xlabel('AXIS1', fontsize=14)
    plt.ylabel('AXIS2', fontsize=14)
    plt.show()
    print("---------------------------------------------------------------------------------------------------")

    gc.collect()


# Confusion Matrix 확인
cm = confusion_matrix(data['Target'], data['3 label'])
print(cm)
