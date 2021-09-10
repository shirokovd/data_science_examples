import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


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


def perform_classifier(X_train, X_test, y_train, y_test, method, classifier_name):
    classifier = method(n_estimators=100, max_depth=4, random_state=0)
    classifier.fit(X_train, y_train)
    y_test_pred = classifier.predict(X_test)
    visualize_classifier(classifier, X_test, y_test)
    plt.savefig('random_forests_results/' + classifier_name + '.png')

    # Оценка классификатора
    class_names = ['Class-0', 'Class-1', 'Class-2']
    print("\n" + "#"*40)
    print("\nClassifier performance on training dataset\n")
    print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
    print("#"*40 + "\n")
    print("#"*40)
    print("\nClassifier performance on test dataset\n")
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    print("#"*40 + "\n")


# Загрузка данных из входного файла
input_file = 'random_forests_data.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]
# Разбивка данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
# Выполнение классификатора
perform_classifier(X_train, X_test, y_train, y_test, RandomForestClassifier, 'random forests classifier')
perform_classifier(X_train, X_test, y_train, y_test, ExtraTreesClassifier, 'extremely random forests classifier')
plt.show()
