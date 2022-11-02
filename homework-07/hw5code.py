import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator

def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух соседних (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """

    def get_index_gini(thr, feature=feature_vector, target=target_vector):
        # Получим значения индексов таргета по разделенным поддеревьям
        R_l_index = feature_vector > thr
        R_r_index = feature_vector < thr
        R_l = target_vector[R_l_index]
        R_r = target_vector[R_r_index]
        # Рассчитаем длину поддеревьев
        R_l_len = len(R_l)
        R_r_len = len(R_r)
        # Вернем заглушку, если одно из поддеревьев пустое
        if R_l_len == 0 or R_r_len == 0:
            return [-np.inf, -np.inf]
        # Найдем общую длину
        R_len = R_l_len + R_r_len
        # Вычислим долю классов в каждом из поддеревьев
        p_1_l = (sum(R_l == 1)) / R_l_len
        p_0_l = 1 - p_1_l

        p_1_r = (sum(R_r == 1)) / R_r_len
        p_0_r = 1 - p_1_r
        # Рассчитаем джини
        gini = -(R_l_len / R_len) * (1 - np.square(p_1_l) - np.square(p_0_l)) - (R_r_len / R_len) * (
                    1 - np.square(p_1_r) - np.square(p_0_r))
        return [thr, gini]

    # Отсортируем вектор признака
    sort_feature = np.sort(feature_vector)
    # Создадим два сдвига для вычисления среднего порога
    lag_plus = sort_feature[1:]
    lag_minus = sort_feature[:-1]
    # Вычислим пороги
    total_lag = (lag_plus + lag_minus) / 2
    # Вычислим пороги и значения джини
    unsorted_return = map(get_index_gini, np.unique(total_lag))
    # Подготовим результат к выдачи
    thr_and_gini = np.array(sorted(unsorted_return, key=lambda x: x[0]))
    thr_and_gini = thr_and_gini[thr_and_gini != [-np.inf, -np.inf]]
    thr_and_gini = thr_and_gini.reshape(len(thr_and_gini) // 2, 2)

    if thr_and_gini.shape[0] != 0:
        # Выделим необходимые для результата начения
        thresholds = thr_and_gini[:, 0]

        ginis = thr_and_gini[:, 1]
        max_gini_index = np.argmax(ginis)
        gini_best = ginis[max_gini_index]
        threshold_best = thresholds[max_gini_index]
        return thresholds, ginis, threshold_best, gini_best
    else:
        return [], [], -np.inf, -np.inf

    pass


class DecisionTree(BaseEstimator):
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y != sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(1, sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_count / current_click
                sorted_categories = list(map(lambda x: x[1], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(map(lambda x: categories_map[x], sub_X[:, feature]))
            else:
                raise ValueError

            if len(feature_vector) == 3:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "Categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[split], node["right_child"])

    def _predict_node(self, x, node):
        # Приведем к numpy
        x = np.array(x)
        # Созададим критерий остановки
        if node['type'] == 'terminal':
            return node['class']
        else:
            # Если код не остановлен, то получаем новые поддеревья согласно созданному дереву
            if 'categories_split' in node:
                condition = node['categories_split'][0]
                feature = node["feature_split"]
                split = x[feature] == condition
            elif "threshold" in node:
                condition = node['threshold']
                feature = node["feature_split"]
                split = int(x[feature]) < condition
            else:
                raise ValueError
            # Продолжим движение по дереву
            if split:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
