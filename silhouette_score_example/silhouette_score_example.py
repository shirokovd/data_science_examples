import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics


# Входной файл, содержащий данные
input_file = 'silhouette_score_data.txt'
# Загрузка данных из входного файла
X = np.loadtxt(input_file, delimiter=',')

# Инициализация переменных
scores = []
values = np.arange(2, 10)

# Определение шага сетки
step_size = 0.01
# Определение сетки точек для отображения границ
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, step_size),
                             np.arange(y_min, y_max, step_size))

# Итерирование в определенном диапазоне значений
for num_clusters in values:
    # Создание объекта KMeans
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    # Обучение модели KMeans
    kmeans.fit(X)
    score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean', sample_size=len(X))
    print("\nNumber of clusters = ", num_clusters)
    print("\nSilhouette score = ", score)
    scores.append(score)
    # Предсказание выходных меток для всех точек сетки
    output = kmeans.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    # Графическое отображение областей и выделение их цветом
    output = output.reshape(x_vals.shape)
    plt.figure()
    plt.clf()
    plt.imshow(output, interpolation='nearest', extent=(x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()),
               cmap=plt.cm.Paired, aspect='auto', origin='lower')
    # Отображение входных точек
    plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none', edgecolors='black', s=80)
    # Отображение центров кластеров
    cluster_centers = kmeans.cluster_centers_
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='o', s=210, linewidths=4, color='black',
                zorder=12, facecolors='black')
    plt.title("Число кластеров = " + str(num_clusters))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.savefig("silhouette_score_results/Число кластеров = " + str(num_clusters) + '.png')

# Отображение силуэтных оценок на графике
plt.figure()
plt.bar(values, scores, width=0.7, color='black', align='center')
plt.title('Зависимость силуэтной оценки от количества кластеров')
plt.savefig('silhouette_score_results/Зависимость силуэтной оценки от количества кластеров.png')

# Извлечение наилучшей оценки и оптимального количества кластеров
num_clusters = np.argmax(scores) + values[0]
print('\nOptimal number of clusters = ', num_clusters)

# Отображение данных на графике
plt.figure()
plt.scatter(X[:, 0], X[:, 1], color='black', s=80, marker='o', facecolors='none')
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
plt.title('Входные данные')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.savefig('silhouette_score_results/Входные данные.png')
plt.show()
