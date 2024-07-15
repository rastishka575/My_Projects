import numpy as np
import cv2


#шаг 3
def draw_border(img, cur_i, cur_j, nbd, outer):
    #для обхода по окружности
    clock = [[-1,0], [-1, 1],
             [0,1], [1,1],[1,0],[1,-1],
             [0, -1], [-1, -1]]

    #внутренний и внешний контуры
    cl = 0
    if outer:
        cl = 0
    else:
        cl = 4

    # поиск рабочей точки 
    work_i, work_j = 0, 0
    for i in range(len(clock)):
        work_i = cur_i + clock[cl][1]
        work_j = cur_j + clock[cl][0]
        if work_j >= 0 and work_i >= 0 and work_j < len(img[0]) and work_i < len(img) and img[work_i, work_j] != 0:
            break
        cl += 1
        if cl >= len(clock):
            cl = 0

    # если не нашли рабочую точку
    if ((work_j < 0 or work_j >= len(img[0])) or (work_i < 0 or work_i >= len(img))) or img[work_i, work_j] == 0:
        img[cur_i, cur_j] = - nbd
        return img

    i2, j2 = work_i, work_j
    i3, j3 = cur_i, cur_j

    while True:
    #задаем nbd для точек
        if j3 < len(img[0]) - 1  and j3 >= 0: 
            if img[i3, j3 + 1] == 0:
                img[i3, j3] = -nbd
            if img[i3, j3 + 1] != 0 and img[i3, j3] == 1:
                img[i3, j3] = nbd

        i_st = i2 - i3
        j_st = j2 - j3

        #находим местоположение рабочей точки относительно текущей
        cl = 0
        for i in range(len(clock)):
            if i_st == clock[i][1] and j_st == clock[i][0]:
                cl = i
                break

        cl -= 1

        #ищем новую текущую точку
        for i in range(len(clock)-1):
            i_st = i3 + clock[cl][1]
            j_st = j3 + clock[cl][0]
            if j_st < len(img[0]) and i_st < len(img) and j_st >= 0 and i_st >= 0 and img[i_st, j_st] != 0:
                break
            cl -= 1
            if cl < 0:
                cl = len(clock) - 1

        if ((j_st < 0 or j_st >= len(img[0])) or (i_st < 0 or i_st >= len(img))) or img[i_st, j_st] == 0:
          break

        #присваиваем новые координаты для точек
        i2, j2 = i3, j3
        i3, j3 = i_st, j_st

        #условие выхода из контура
        if i2 == work_i and j2 == work_j and i3 == cur_i and j3 == cur_j:
            break

    return img



if __name__ == "__main__":

    #файл
    path = 'segment.jpg'
    img = cv2.imread(path)

    img_res = img

    #перевод в формат hsv
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #диапазон маски
    low_green = np.array([32, 5, 0])
    high_green = np.array([90, 255, 220])

    #маска
    mask = cv2.inRange(img, low_green, high_green)

    mask = mask/255

    #бинарное изображение
    img = mask

    #шаг 1
    nbd = 1

    #разделение на внешний и внутренний контур
    for i in range(len(img)):
        outer = False
        for j in range(len(img[0])):
            if img[i, j] == 1 and (j == 0 or img[i, j-1] == 0):
                nbd += 1
                outer = True
                img = draw_border(img, i, j, nbd, outer)
            else:
                if img[i, j] >= 1 and (j == len(img[0]) - 1 or img[i, j + 1] == 0):
                    nbd += 1
                    outer = False
                    img = draw_border(img, i, j, nbd, outer)

    #bounding box
    img = abs(img)
    cnt = []
    cnt2 = []

    #проходимся один раз по изображению и считываем для каждого nbd: значение nbd для cnt2 и координаты 
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i, j] >=2:
              cnt.append([[j, i]])
              cnt2.append(img[i, j])

    #сортируем по возрастанию nbd
    cnt = np.asarray(cnt)
    cnt2 = np.asarray(cnt2)
    tx = np.argsort(cnt2)
    cnt = cnt[tx]
    cnt2 = cnt2[tx]

    #отрисовываем bounding box на изображении с фильтром
    k1 = 0
    k2 = 0
    for i in range(nbd+1):
        while k2 < len(cnt2) and cnt2[k2] == i:
          k2 += 1
        if len(cnt[k1:k2]) >= 4:
          x, y, w, h = cv2.boundingRect(cnt[k1:k2])
          if w*h >= 135000:
              img_res = cv2.rectangle(img_res, (x, y), (x + w, y + h), (0, 255, 0), 10)
        k1 = k2

    cv2.imwrite('img_res.png', img_res)
