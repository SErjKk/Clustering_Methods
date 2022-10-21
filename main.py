import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs
%matplotlib inline
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (18,8)

# Генерация данных
X,y = make_blobs(n_samples = 200, random_state = 1,centers = 5)


# Метод К-средних
def k_means():
    fig, axes = plt.subplots(1, 2)
    kmeansModel = KMeans(n_clusters = 5)
    kmeansModel.fit(X)
    labels = kmeansModel.labels_
    axes[0].scatter(X[:, 0], X[:, 1], c = labels)
    axes[0].set_title("Prediction")
    axes[1].scatter(X[:, 0], X[:, 1], c = y)
    axes[1].set_title("Generated data")
    fig.suptitle("K-Means Clustering", fontsize = 16)


# Метод локтя для определения оптимального кол-ва кластеров
def elbow_method():
    criteries = []
    for k in range(2, 10):
        kmeansModel = KMeans(n_clusters = k, random_state = 3)
        kmeansModel.fit(X)
        criteries.append(kmeansModel.inertia_)
    plt.plot(range(2, 10), criteries)


# Плотностный метод - DBSCAN
def dbscan():
    fig, axes = plt.subplots(1, 2)
    dbscanModel = DBSCAN(eps = 0.9, min_samples = 4)
    clustering = dbscanModel.fit_predict(X)
    axes[0].scatter(X[:, 0], X[:, 1], c = clustering)
    axes[0].set_title("Prediction")
    axes[1].scatter(X[:, 0], X[:, 1], c = y)
    axes[1].set_title("Generated data")
    fig.suptitle("DBSCAN Clustering", fontsize = 16)