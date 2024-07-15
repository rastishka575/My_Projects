import numpy as np
import cv2

# вычисляем медианное значение в списке
def median_select(mas):
    return median_search(mas, len(mas)//2)


# каждый раз случайно подбираем медиану
# если это значение не медианное,то переходим в тот список, где лежит медиана (либо левее, либо правее) через индекс
def median_search(mas, index):

    if len(mas) == 1:
        return mas[0]

    median = np.random.choice(mas)

    lefts = mas < median
    rights = mas > median
    medians = mas == median

    left = mas[lefts]
    med = len(mas[medians])

    if index < len(left):
        return median_search(left, index)

    if index < len(left) + med:
        return median

    return median_search(mas[rights], index - len(left) - med)


# горизонтальный BAG
def horizontal_bag_extraction(img, R):

    block = 16      # 16*2+1=33 - блок
    threshold = 30  # порог E

    # вычисляем апроксимацию второй производной, одновременно обнуляем значение, где R(x, y) = 1
    # d(y, x) = |2*s(y,x) - s(y-1, x) - s(y+1, x)|
    img_d = np.zeros((img.shape[0] + 2 * block, img.shape[1] + 2 * block))

    for i in range(1, img.shape[0] - 1):
        for j in range(0, img.shape[1]):
            if R[i, j] != 1:
                d = abs(2 * img[i, j] - img[i - 1, j] - img[i + 1, j])
                img_d[i + block, j + block] = d

    # на границах
    for j in range(0, img.shape[1]):
        if R[0, j] != 1:
            d = abs(2 * img[0, j] - img[1, j])
            img_d[block, j + block] = d

    for j in range(0, img.shape[1]):
        if R[img.shape[0] - 1, j] != 1:
            d = abs(2 * img[img.shape[0] - 1, j] - img[img.shape[0] - 2, j])
            img_d[img.shape[0] - 1 + block, j + block] = d

    # если d > 30, то обнуляем d
    mask_d = img_d > threshold
    img_d[mask_d] = 0

    # первый этап усиления BAG
    # вычисляем фильтром 1 на 33 суммы
    e_sum = np.zeros((img.shape[0] + 2 * block, img.shape[1] + 2 * block))

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            e_sum[i + block, j + block] = img_d[i + block, j:j + 2 * block + 1].sum()

    # итоговое знчение первого этапа
    # вычисляем разность между значением и медианой фильтра 33 на 1
    e_h = np.zeros((img.shape[0] + 2 * block, img.shape[1] + 2 * block))
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            mid_e = median_select(e_sum[i:i + 2 * block + 1, j + block])
            e_h[i + block, j + block] = e_sum[i + block, j + block] - mid_e

    # второй этап усиления BAG
    # вычисляем медианное значение среди 5 значений фильтра 33 на 1
    g = np.zeros(img.shape)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            h = np.array([e_h[i, j + block],
                 e_h[i + block//2, j + block],
                 e_h[i + block, j + block],
                 e_h[i + block + block//2, j + block],
                 e_h[i + 2*block, j + block]])
            g_h = median_select(h)
            g[i, j] = g_h

    return g


def bag_extraction(img, R):
    # по горизонтали
    g_h = horizontal_bag_extraction(img, R)
    # по вертикали
    g_w = horizontal_bag_extraction(img.T, R.T)
    g_w = g_w.T
    # итоговая маска
    g = g_h + g_w
    return g


# подсчёт anomaly score
# значения блока 8 на 8 равно max(сумма внутренних строк без крайних столбов)-min(суммы краев строк без крайних столбов)
# + max(сумма внутренних столбцов без крайних строк)-min(суммы краев столбцов без крайних строк)
def anomaly_score(img):
    a = np.zeros(img.shape)
    bl = 8
    for i in range(0, img.shape[0] - bl + 1, bl):
        for j in range(0, img.shape[1] - bl + 1, bl):
            x_max = img[i + 1:i + bl - 1, j + 1].sum()
            y_max = img[i + 1, j + 1:j + bl - 1].sum()
            for k in range(1, bl - 1):
                x_sum = img[i + 1:i + bl - 1, j + k].sum()
                y_sum = img[i + k, j + 1:j + bl - 1].sum()
                if x_sum > x_max:
                    x_max = x_sum
                if y_sum > y_max:
                    y_max = y_sum

            x_min = img[i + 1:i + bl - 1, j].sum()
            y_min = img[i, j + 1:j + bl - 1].sum()
            x_8 = img[i + 1:i + bl - 1, j + bl - 1].sum()
            y_8 = img[i + bl - 1, j + 1:j + bl - 1].sum()
            if x_8 < x_min:
                x_min = x_8
            if y_8 < y_min:
                y_min = y_8
            a[i:i + bl, j:j + bl] = x_max - x_min + y_max - y_min
    return a


# предобработка
def preprocessing(img):
    th = 10  # порог угла

    # маски Собеля по x и y
    g_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    g_x = g_y.T
    mask = len(g_y)  # размер маски
    img_m = np.zeros((img.shape[0]+2, img.shape[1]+2))
    img_m[1:img.shape[0]+1, 1:img.shape[1]+1] = img
    # углы
    theta = np.zeros(img.shape)

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            x = img_m[i:i + mask, j:j + mask] * g_x
            y = img_m[i:i + mask, j:j + mask] * g_y
            x = abs(x.sum())
            y = abs(y.sum())
            theta[i, j] = np.arctan2(y, x)

    R = np.zeros(img.shape)
    # в градусах
    theta = np.degrees(theta)

    # вычисляем R согласно условию
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (theta[i,j] >= 0 and theta[i,j] <= th) or (theta[i,j] >= 90 - th and theta[i,j] <= 90 + th) or (theta[i,j] >= 180 - th and theta[i,j] < 180):
                R[i, j] = 0
            else:
                R[i, j] = 1

    return R


if __name__ == "__main__":

    # считываем изображение
    path = 'planes_forg_2.jpg'
    img = cv2.imread(path)
    # перевод в градации серого
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # предобработка
    R = preprocessing(img)

    # вычисляем сетку
    img = bag_extraction(img, R)

    # вычисляем anomaly score
    img = anomaly_score(img)

    # нормируем
    im = img < 0
    img[im] = 0
    img = img/np.max(img)
    img *= 255

    # сохраняем
    cv2.imwrite('img.png', img)





