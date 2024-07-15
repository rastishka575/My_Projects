import numpy as np
import cv2

# модуль и направление градиента
def compute_gradient(img):

    # вычисляем производные
    g_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_CONSTANT)
    g_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_CONSTANT)

    # направление градиента
    theta = np.arctan2(g_y, g_x)

    # аппроксимируем углы с шагом 45 градусов
    theta /= np.pi/4
    theta = np.around(theta)
    theta *= np.pi/4

    # диапазон от 0 до pi
    theta_mask = theta < 0
    theta[theta_mask] += np.pi
    theta = abs(theta)

    # модуль градиента
    g = np.sqrt(g_x**2+g_y**2)

    return g, theta


# non maximum suppression по трём точкам соответсвующих направлению градиента
def non_maximum_suppression(gradient_check, theta):

    gradient = np.zeros(gradient_check.shape)

    for i in range(1, gradient.shape[0] - 1):
        for j in range(1, gradient.shape[1] - 1):
            # максимальное значение соседа
            max_neighbor = 0
            if theta[i, j] == 0 or theta[i, j] == np.pi:
                max_neighbor = max(gradient_check[i, j-1], gradient_check[i, j+1])

            if theta[i, j] == np.pi/4:
                max_neighbor = max(gradient_check[i-1, j-1], gradient_check[i+1, j+1])

            if theta[i, j] == np.pi/2:
                max_neighbor = max(gradient_check[i-1, j], gradient_check[i+1, j])

            if theta[i, j] == 3*np.pi/4:
                max_neighbor = max(gradient_check[i-1, j+1], gradient_check[i+1, j-1])

            # сохраняем максимальные значения модуля градиента
            if gradient_check[i, j] >= max_neighbor:
                gradient[i, j] = gradient_check[i, j]

    return gradient


# класс для алгоритма Connected Components Labeling
class SSL(object):
    def __init__(self, img, shape):
        self.parent = np.empty((shape[0], shape[1], 2), dtype=int)  # родитель
        self.rank = np.zeros(shape)
        self.index = np.zeros(shape)  # метки
        self.ind = 1  # наибольшая метка
        self.max_value = []  # максимальное значение в связности
        self.img = img

    # добавление меток
    def makeset(self, i, j):
        self.parent[i, j, :] = [int(i), int(j)]  # инициализируем корень связности
        self.rank[i, j] = 0
        self.index[i, j] = self.ind
        self.ind += 1
        self.max_value.append(self.img[i, j])

    # добавление родителя соседу
    def add_neighbor(self, i, j, m, n):
        self.parent[i, j, :] = [int(m), int(n)]
        self.rank[m, n] += 1
        self.index[i, j] = self.index[m, n]
        #  максимальное значение в связности
        if self.max_value[int(self.index[i, j])-1] < self.img[i, j]:
            self.max_value[int(self.index[i, j])-1] = self.img[i, j]

    # поиск корня дерева
    def find(self, i, j):
        if self.parent[i, j, 0] != i or self.parent[i, j, 1] != j:
            self.parent[i, j, :], self.index[i, j] = self.find(self.parent[i, j, 0], self.parent[i, j, 1])
            self.rank[i, j] = 0

        if self.max_value[int(self.index[i, j])-1] < self.img[i, j]:
            self.max_value[int(self.index[i, j])-1] = self.img[i, j]

        return self.parent[i, j, :], self.index[i, j]

    # объединение двух множеств
    def union(self, i, j, m, n):
        # корни множеств
        parent_a, _ = self.find(i, j)
        parent_b, _ = self.find(m, n)

        if parent_a[0] == parent_b[0] and parent_a[1] == parent_b[1]:
            return

        # минимальная метка
        index = min(self.index[parent_a[0], parent_a[1]],
                    self.index[parent_b[0], parent_b[1]])

        # нахождение корня дерева
        if self.rank[parent_a[0], parent_a[1]] >= self.rank[parent_b[0], parent_b[1]]:
            self.parent[parent_b[0], parent_b[1], :] = parent_a
            self.index[parent_a[0], parent_a[1]] = index

            if self.rank[parent_a[0], parent_a[1]] == self.rank[parent_b[0], parent_b[1]]:
                self.rank[parent_a[0], parent_a[1]] += 1

        else:
            self.parent[parent_a[0], parent_a[1], :] = parent_b
            self.index[parent_b[0], parent_b[1]] = index


# поиск соседей с минимальной меткой
def check_neighbor(index, m, n):
    min_index = index.shape[0]*index.shape[1]
    min_i, min_j = 0, 0
    neighbor = False
    for i in range(m - 1, m + 2):
        for j in range(n - 1, n + 2):
            if i >= 0 and i < index.shape[0] and j >= 0 and j < index.shape[1]:
                if (i != m or j != n) and index[i, j] != 0:
                    if index[i, j] < min_index:
                        neighbor = True
                        min_index, min_i, min_j = index[i, j], i, j

    if neighbor == False:
        return neighbor
    else:
        return min_i, min_j


# поиск соседних множеств
def union_neighbor(sets, m, n):
    for i in range(m - 1, m + 2):
        for j in range(n - 1, n + 2):
            if i >= 0 and i < sets.index.shape[0] and j >= 0 and j < sets.index.shape[1]:
                if (i != m or j != n) and sets.index[i, j] != 0:
                    if sets.index[i, j] != sets.index[m, n]:
                        sets.union(i, j, m, n)

    return sets


# алгоритм Connected Components Labeling
def hysteresis_threshold(img, high_threshold, low_threshold):
    # инициализируем класс для алгоритма
    sets = SSL(img, img.shape)
    # first pass
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            # не фон
            if img[i, j] != 0:
                # поиск соседей
                neighbor = check_neighbor(sets.index, i, j)
                if neighbor != False:
                    # добавление соседа с минимальной меткой
                    sets.add_neighbor(i, j, neighbor[0], neighbor[1])
                else:
                    # если соседей нет, то инициализируем новую метку
                    sets.makeset(i, j)
                # объединение двух соседних множеств
                sets = union_neighbor(sets, i, j)

    # second pass
    # инициализается корня для каждого множества
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i, j] != 0:
                sets.find(i, j)

    # отсеивание по нижнему порогу
    low_mask = img < low_threshold

    # уточнение связностей по верхнему порогу
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i, j] != 0:
                # если максимальное значение в связности меньше, чем верхний порог
                if sets.max_value[int(sets.index[i, j])-1] > high_threshold:
                    img[i, j] = 255

    img[low_mask] = 0

    return img


# уточнение границ по верхнему и нижнему порогу
def function_threshold(img, low_threshold=10, high_threshold=90):
    # алгоритм Connected Components Labeling и отсеивание средних значений по связности
    img = hysteresis_threshold(img, high_threshold, low_threshold)
    # бинаризация
    binary_mask = img != 255
    img[binary_mask] = 0
    return img


if __name__ == "__main__":

    # загружаем изображение
    path = 'emma.jpg'
    image = cv2.imread(path)

    # перевод в градации серого
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # фильтр гаусса
    img = cv2.GaussianBlur(img, (5, 5), sigmaX=1, sigmaY=1)

    # модуль и направление градиента
    img, theta = compute_gradient(img)

    # non maximum suppression по трём точкам
    img = non_maximum_suppression(img, theta)

    # уточнение границ по верхнему и нижнему порогу
    img = function_threshold(img, 30, 90)

    # сохраняем
    cv2.imwrite('img_canny.png', img)

