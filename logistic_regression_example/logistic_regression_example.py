import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def visualize_classifier(classifier, X, y):
    # Определение для X и Y минимального и максимального значений, которые будут использоваться при построении сетки
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    # Определение величины шага для построения сетки
    mesh_step_size = 0.01
    # Определение сетки для значений X и Y
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size), np.arange(min_y, max_y, mesh_step_size))
    # Выполнение классификатора на сетке данных
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    # Переформирование выходного массива
    output = output.reshape(x_vals.shape)
    # Создание графика
    plt.figure()
    # Выбор цветовой схемы для графика
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)
    # Размещение тренировочных точек на графике
    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', linewidths=1, cmap=plt.cm.Paired)
    # Определение границ графика
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())
    # Определение делений на осях X и Y
    plt.xticks((np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1), 1.0)))
    plt.yticks((np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1), 1.0)))


# Входной файл, содержащий данные
input_file = 'logistic_regression_data.txt'
# Загрузка данных из входного файла
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Разбивка результатов на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
classifier = LogisticRegression(solver='liblinear', C=100)
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)
accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print("Accuracy of classifier = ", round(accuracy, 2), "%")
visualize_classifier(classifier, X_test, y_test)
plt.savefig('logistic_regression_results/logistic regression result.png')
plt.show()
