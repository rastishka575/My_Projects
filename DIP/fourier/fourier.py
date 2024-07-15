import numpy as np
import cv2
from scipy import ndimage


# вычисляем угол наклона изображения через гистограмму
def compute_direction_histogram(img):
    mask = img > 0
    # гистограмма углов поворота от 0 до 360
    theta_img = np.zeros(361)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                # arctan отношений индексов высоты к индексам ширины в градусах
                th = int(np.degrees(np.arctan2(i, j) + np.pi))
                theta_img[th] += 1
    # угол, за который проголосовало большинство оставшихся точек
    theta = theta_img.argmax() - 180
    return theta


# вычисляем угол наклона изображения через преобразование Хафа
def compute_direction_hough(img):
    img = img.astype(dtype=np.uint8)
    # все линии через преобразование Хафа
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 3)
    # оставляем линию с наибольшей длиной
    max_length = 0
    max_lines = None
    for i in range(lines.shape[0]):
        for x1, y1, x2, y2 in lines[i]:
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length > max_length:
                max_length = length
                max_lines = lines[i]

    # угол на который необходимо повернуть изображение
    theta = 0
    for x1, y1, x2, y2 in max_lines:
        theta = np.arctan2(x2 - x1, y2 - y1)
        theta = np.degrees(theta)
        theta = np.round(theta)
        theta = 90 - theta
        if x1 == x2:
            theta = 0
        if y1 == y2:
            theta = 90

    return theta


# визуализация действительной части двумерного преобразования Фурье
def fourier_2d(img):
    # вычисляем спектр смещенного двумерного преобразования Фурье
    img = np.fft.fft2(img)
    # Сдвиг DC коэффициента в центр изображения
    img = np.fft.fftshift(img)
    # избавляемся от мнимой части
    img = np.abs(img)
    # логарифмируем
    img = 10 * np.log(img)
    return img


if __name__ == "__main__":

    # загружаем изображение
    path = 'text.jpg'

    image = cv2.imread(path)

    # перевод в градации серого
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # визуализация действительной части двумерного преобразования Фурье
    img = fourier_2d(img)

    # подавляем часть коэффициентов с помощью порога
    threshold = 125  # порог
    mask_threshold = img < threshold
    img[mask_threshold] = 0

    # вычисляем угол наклона изображения

    # 1 способ
    # вычисляем угол наклона изображения через гистограмму
    # работает плохо, потому что угол лежит от 0 до np.pi/2
    # theta_his = compute_direction_histogram(img)

    # 2 способ
    # вычисляем угол наклона изображения через преобразование Хафа
    theta_hough = compute_direction_hough(img)

    # поворот изображения
    image = ndimage.rotate(image, theta_hough, reshape=False)

    # сохраняем
    cv2.imwrite('image_result.jpg', image)
