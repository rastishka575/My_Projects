import numpy as np
import cv2

# интегральное изображение
def integral_image(img):

    # добавляем нули слева и сверху
    integ_img = np.zeros((img.shape[0]+1, img.shape[1] + 1))

    # по строкам
    for i in range(0, len(img)):
        integ_img[i + 1, 1:] = img[i].cumsum()

    # по столбцам и строкам
    for i in range(0, integ_img.shape[1]):
        integ_img[:, i] = integ_img[:, i].cumsum()

    return integ_img


# бинаризация Сауволы
def binary_sauvola(img):

    # параметры
    w = 5 #31  # окно w на w [14, 60], w - нечетное число
    k = 0.2 #0.21  # [0.2, 0.5]
    R = 128  # значение 128 для изображений в градациях серого с 256 оттенками

    img_binary = np.zeros(img.shape)

    # вычисляем интегральное изображение s1(x[i]) и s2(x[i]^2) соответственно
    # делим на 255 во избежании переполняемости
    integ_img = integral_image(img/255)
    integ_img2 = integral_image((img/255)**2)

    # проходим по пиксельно с окном w на w c центром i, j
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):

            # границы окна
            x1 = i - w//2
            x2 = i + w//2
            y1 = j - w//2
            y2 = j + w//2

            # при выходе окна за границы изображения
            if x1 < 0:
                x1 = 0
            if x2 >= img.shape[0]:
                x2 = img.shape[0] - 1
            if y1 < 0:
                y1 = 0
            if y2 >= img.shape[1]:
                y2 = img.shape[1] - 1

            # количество элементов в окне
            n = (x2 - x1 + 1) * (y2 - y1 + 1)

            # сумма пикселей в окне через интегральное изображение s1 D - C - B + A
            average = integ_img[x2 + 1, y2 + 1] - integ_img[x2 + 1, y1] - integ_img[x1, y2 + 1] + integ_img[x1, y1]
            # среднее значение пикселя в окне
            average /= n

            # сумма квадратов пикселей в окне через интегральное изображение s2 D - C - B + A
            s_2 = integ_img2[x2 + 1, y2 + 1] - integ_img2[x2 + 1, y1] - integ_img2[x1, y2 + 1] + integ_img2[x1, y1]
            # дисперсия
            dispersion = s_2 / n - average**2
            dispersion = np.sqrt(abs(dispersion))


            # от 0-1 переходим в 0-255
            average *= 255
            dispersion *= 255

            # вычисляем соотвествующий порог для пикселя
            threshold = average * (1 + k * (dispersion / R - 1))

            # бинаризуем
            if img[i, j] > threshold:
                img_binary[i, j] = 255

    return img_binary


# бинаризация Отсу
def binary_otsu(img):

    histogram, bin_edges = np.histogram(img.flatten(), range=(0, 256), bins=256)

    # максимальная межклассовая дисперсия и его порог
    max_dispersion_between = -1
    threshold = 0  # T

    # количество пикселей - в изображении и объекте
    sum_pixel = histogram.sum()
    sum_pixel_object = 0

    # сумма всех пикселей - в изображении и объекте
    sum_img = histogram * np.arange(256)
    sum_img = sum_img.sum()

    sum_object = 0

    # подбираем порог для изображения
    for i in range(1, len(histogram)):
        # сумма всех пискелей в объекте до T - 1
        sum_object += (i - 1) * histogram[i - 1]
        # количество пикселей в объекте до T - 1
        sum_pixel_object += histogram[i - 1]
        # сумма вероятностей наличия пикселя в объекте
        f_object = sum_pixel_object / sum_pixel
        # разница средних значений пискеля между фоном и объектом
        average = (sum_img - sum_object) / (sum_pixel - sum_pixel_object) - sum_object / sum_pixel_object
        # межклассовая дисперсия с порогом i
        dispersion_between = f_object * (1.0 - f_object) * average**2

        # подбираем такой порог T, что при исходном пороге значение межклассовой дисперсии максимальна
        if dispersion_between > max_dispersion_between:
            max_dispersion_between = dispersion_between
            threshold = i - 1  # так как f(x, y) > T

    # бинаризуем изображение строго с порогом T
    img_binary = img > threshold
    img_binary = img_binary.astype(int)
    img_binary *= 255

    return img_binary


if __name__ == "__main__":

    # считываем файл с путем path
    # path = 'ex.jpg'
    path = 'Houses.jpg'
    img = cv2.imread(path)

    # переводим изображение в градации серого
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Отсу
    img_otsu = binary_otsu(img)
    # Саувола
    img_sauvola = binary_sauvola(img)

    # сохраняем изображения в формате png
    cv2.imwrite('res_otsu.png', img_otsu)
    cv2.imwrite('res_sauvola.png', img_sauvola)
