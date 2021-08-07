import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

# Входной файл, содержащий данные
input_file = 'mean_shift_data.txt'
# Загрузка данных из входного файла
X = np.loadtxt(input_file, delimiter=',')

# Оценка ширины окна для X
bandwidth_X = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))

# Кластеризация данных методом сдвига среднего
meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
meanshift_model.fit(X)

# Извлечение центров кластеров
cluster_centers = meanshift_model.cluster_centers_
print('\nCenters of clusters:\n', cluster_centers)

# Оценка количества кластеров
labels = meanshift_model.labels_
num_clusters = len(np.unique(labels))
print('\nNumber of clusters in input data =', num_clusters)

# Отображение на графике точек и центров кластеров
plt.figure()
markers = 'o*xvs'
for i, marker in zip(range(num_clusters), markers):
    # Отображение на графике точек, принадлежащих текущему кластеру
    plt.scatter(X[labels == i, 0], X[labels == i, 1], marker=marker, color='black')
    # Отображение на графике центра кластера
    cluster_center = cluster_centers[i]
    plt.plot(cluster_center[0], cluster_center[1], marker='o',
             markerfacecolor='black', markeredgecolor='black', markersize=15)
plt.title('Clusters')
plt.savefig('mean_shift_results/clusters.png')
plt.show()
