import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Входной файл, содержащий данные
input_file = 'kmeans_data.txt'
# Загрузка данных из входного файла
X = np.loadtxt(input_file, delimiter=',')

num_clusters = 5

# Построение входных данных
plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none',
            edgecolors='black', s=80)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
plt.title('Input data')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# Создание объекта KMeans
kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
# Обучение модели KMeans
kmeans.fit(X)

# Определение шага сетки
step_size = 0.01
# Определение сетки точек для отображения границ
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, step_size),
                             np.arange(y_min, y_max, step_size))
# Предсказание выходных меток для всех точек сетки
output = kmeans.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
# Графическое отображение областей и выделение их цветом
output = output.reshape(x_vals.shape)
plt.figure()
plt.clf()
plt.imshow(output, interpolation='nearest',
           extent=(x_vals.min(), x_vals.max(),
                   y_vals.min(), y_vals.max()),
           cmap=plt.cm.Paired,
           aspect='auto',
           origin='lower')
# Отображение входных точек
plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none',
            edgecolors='black', s=80)
# Отображение центров кластеров
cluster_centers = kmeans.cluster_centers_
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
            marker='o', s=210, linewidths=4, color='black',
            zorder=12, facecolors='black')
plt.title('Boundaries of clusters')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.savefig('kmeans_results/boundaries of clusters.png')
plt.show()
