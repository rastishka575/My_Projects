import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import pickle

#для сохранения моделей
def pickle_it(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def unpickle_it(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
#стандартизируем изображение

def standartization(img):
    mean = img.mean()
    d = np.std(img)
    if d == 0:
        d = 1
    img = (img-mean)/d
    return img
#считываем изображения

def reading(directory):
    files = os.listdir(directory)
    im = []
    for i in range(len(files)):
        img = cv.imread(directory+'\\'+files[i])
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = standartization(img)
        im.append(img)

    return im

#вычисляем integral image для изображения
def integral_image(img):
    rown = np.zeros(img.shape[1])
    column = np.zeros((img.shape[0]+1, 1))
    intr = np.row_stack((rown, img))
    int = np.column_stack((column, intr))
    int = np.cumsum(int, axis=1)
    int = np.cumsum(int, axis=0)
    return int

#две вертикальные полосы
def first(img):
    height = 1
    width = 2
    sum = np.zeros(1)
    while width < img.shape[1]/2:
        while height < img.shape[0]:
            i = 0
            while i+width < img.shape[1]:
                j = 0
                while j+height < img.shape[0]:
                    white = img[j+height, int(i+width/2)]
                    black = img[j+height, i+width]-img[j+height, int(i+width/2)]
                    s = white - black
                    sum = np.append(sum, s)
                    j = j + 1

                i = i + 1

            height = height + 1

        width = width + 2

    sum = np.delete(sum, 0)
    return sum

#три вертикальные полосы
def second(img):
    height = 1
    width = 3
    sum = np.zeros(1)
    while width < img.shape[1]/3:
        while height < img.shape[0]:
            i = 0
            while i+width < img.shape[1]:
                j = 0
                while j+height < img.shape[0]:
                    white1 = img[j+height, int(i+width/3)]
                    white2 = img[j+height, i+width]-img[j+height, int(i+2*width/3)]
                    black = img[j+height, int(i+2*width/3)]-img[j+height, int(i+width/3)]
                    s = white1 + white2 - black
                    sum = np.append(sum, s)
                    j = j + 1

                i = i + 1

            height = height + 1

        width = width + 3

    sum = np.delete(sum, 0)
    return sum

#две горизонтальные полосы
def third(img):
    height = 2
    width = 1
    sum = np.zeros(1)
    while width < img.shape[1]:
        while height < img.shape[0]/2:
            i = 0
            while i+width < img.shape[1]:
                j = 0
                while j+height < img.shape[0]:
                    white = img[int(j+height/2), i+width]
                    black = img[j+height, i+width]-img[int(j+height/2), i+width]
                    s = white - black
                    sum = np.append(sum, s)
                    j = j + 1

                i = i + 1

            height = height + 2

        width = width + 1

    sum = np.delete(sum, 0)
    return sum

#три горизонтальные полосы
def four(img):
    height = 3
    width = 1
    sum = np.zeros(1)
    while width < img.shape[1]:
        while height < img.shape[0]/3:
            i = 0
            while i+width < img.shape[1]:
                j = 0
                while j+height < img.shape[0]:
                    white1 = img[int(j+height/3), i+width]
                    white2 = img[j+height, i+width]-img[int(j+2*height/3), i+width]
                    black = img[int(j+2*height/3), i+width]-img[int(j+height/3), i+width]
                    s = white1 + white2 - black
                    sum = np.append(sum, s)
                    j = j + 1

                i = i + 1

            height = height + 3

        width = width + 1

    sum = np.delete(sum, 0)
    return sum

#четыре квадрата
def five(img):
    height = 2
    width = 2
    sum = np.zeros(1)
    while width < img.shape[1]/2:
        while height < img.shape[0]/2:
            i = 0
            while i+width < img.shape[1]:
                j = 0
                while j+height < img.shape[0]:
                    white1 = img[int(j+height/2), int(i+width/2)]
                    black1 = img[j+height, int(i+width/2)]-img[int(j+height/2), int(i+width/2)]
                    black2 = img[int(j+height/2), i+width]-img[int(j+height/2), int(i+width/2)]
                    white2 = img[j+height, i+width] + white1-black1-black2
                    s = white1+white2 - black1-black2
                    sum = np.append(sum, s)
                    j = j + 1

                i = i + 1

            height = height + 2

        width = width + 2

    sum = np.delete(sum, 0)
    return sum

#вычисляем haar-like features для 5 фигур
def haar_like_features(img):
    img = np.array(integral_image(img))
    im = []
    h1 = first(img)
    im.append(h1)
    h2 = second(img)
    im.append(h2)
    h3 = third(img)
    im.append(h3)
    h4 = four(img)
    im.append(h4)
    h5 = five(img)
    im.append(h5)

    return im

#слабый классификтор
#класс пни решений
class Stump:
    threshold = None
    polarity = None
    figur_th = None
    value_th = None

    def __init__(self, polarity, figur_th, threshold, value_th):
        self.threshold = threshold
        self.polarity = polarity
        self.figur_th = figur_th
        self.value_th = value_th

    #для определения класса в слабом классификаторе
    def compire(self, img):
        if(self.value_th>=img[self.figur_th][self.threshold]):
            return self.polarity
        else:
            return (-1)*self.polarity

#подбирается лучшая граница и полярность для слабого классификатора 
#исолзуется алгоритм random node optimization
def best_stump(train , labels):
    figur = np.random.random_integers(0, 4, 3)
    features = []
    for i in range(len(figur)):
        f = np.random.random_integers(0, len(train[0][figur[i]])-1, int(np.sqrt(len(train[0][figur[i]]))))
        features.append(f)
    t_p = 0
    t_n = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            t_p = t_p + 1
        else:
            t_n = t_n + 1

    e_min = len(labels)
    figur_th_best = 0
    feat_th_best = 0
    polar = 1
    val_best = 0

    for i in range(len(figur)):
        for j in range(len(features[i])):
            val = np.random.random_integers(0, len(train)-1, int(len(train)/3))
            for m in val:
                for v in range(len(train)):
                    s_p = 0
                    s_n = 0
                    if(train[m][figur[i]][features[i][j]] >= train[v][figur[i]][features[i][j]]):
                        if(labels[v] == 1):
                            s_p = s_p + 1
                        else:
                            s_n = s_n + 1
                    e = min(s_n + t_p - s_p, s_p + t_n - s_n)
                    if e < e_min:
                        e_min = e
                        if e == s_p + t_n - s_n:
                            polar = -1
                        figur_th_best = figur[i]
                        feat_th_best = features[i][j]
                        val_best = train[m][figur[i]][features[i][j]]

    return Stump(polar, figur_th_best, feat_th_best, val_best)

#сильный классификатор 
class Adaboost:
    alpha = None
    stump = None

    def __init__(self, alpha, stump):
        self.alpha = alpha
        self.stump = stump

#построение адабуста
def adab(train, labels):
    alpha = np.ones(11)
    stump = []
    for i in range(11):
        s = best_stump(train, labels)
        stump.append(s)

    return Adaboost(alpha, stump)

#определение класса в сильном классификаторе
def adab_class(boost, img):
    classes = 1
    sum = 0
    for i in range(len(boost.stump)):
        sum = sum + boost.alpha[i]*boost.stump[i].compire(img)
    if sum != 0:
        classes = np.sign(sum)
    return classes

#обучение адабуста 
def adab_learn(boost, train, labels):
    w = np.ones(len(train))
    w = w / len(train)
    eps_diff_prev = 0.5
    while True:
        for j in range(len(boost.stump)):
            eps = 0
            for i in range(len(train)):
                eps = eps + w[i]*boost.stump.compire(train[i])*labels[i]
            boost.alpha[j] = np.log((1-eps)/eps)

            z = 0
            for i in range(len(train)):
                z = z + np.exp(boost.alpha[i]*boost.stump.compire(train[i])*labels[i])

            for i in range(len(train)):
                w[i] = np.exp(boost.alpha[i]*boost.stump.compire(train[i])*labels[i])/z

            if (eps == 0.5):
                return boost

            eps_diff = abs(0.5 - eps)
            if (eps_diff_prev > eps_diff):
                eps_diff_prev = eps_diff
            else:
                return boost

#вычисление точности(accuracy) и true positive и false negative
def accur(boost, train, labels):
    a = 0
    tp = 0
    for i in range(len(train)):
        c = adab_class(boost, train[i])
        if c == labels[i]:
            a = a + 1
            if c == 1:
            	tp = tp + 1
    fp = a-tp
    accur = a/len(train)

    return accur, tp, fp

#main

'''
#считываем с помощью os, стандартизируем и сохраняем 
directory_train_face = 'train\\face'
directory_train_non_face = 'train\\non-face'
directory_test_face = 'test\\face'
directory_test_non_face = 'test\\non-face'

train_face = reading(directory_train_face)
train_non = reading(directory_train_non_face)
test_face = reading(directory_test_face)
test_non = reading(directory_test_non_face)

pickle_it(train_face, "train_face.pickle")
pickle_it(train_non, "train_non.pickle")
pickle_it(test_face, "test_face.pickle")
pickle_it(test_non, "test_non.pickle")
'''

train_face = unpickle_it("train_face.pickle")
train_non = unpickle_it("train_non.pickle")
test_face = unpickle_it("test_face.pickle")
test_non = unpickle_it("test_non.pickle")

#определения классов на -1 и 1 (non face and face)
labels_train_face = np.ones(len(train_face))
labels_train_non = np.zeros(int(len(train_non)/2))-1
labels_test_face = np.ones(len(test_face))
labels_test_non = np.zeros(int(len(test_non)/10))-1

'''
#определяем вектора характеристик и сохраняем (где то по частям)
characteristic_train_face = []
characteristic_train_non = []
characteristic_test_face = []
characteristic_test_non = []

for i in range(len(train_face)):
    im = haar_like_features(train_face[i])
    characteristic_train_face.append(im)

pickle_it(characteristic_train_face, "characteristic_train_face.pickle")

for i in range(int(len(train_non)/2)):
    im = haar_like_features(train_non[i])
    characteristic_train_non.append(im)

pickle_it(characteristic_train_non, "characteristic_train_non1.pickle")

for i in range(int(len(train_non)/2), len(train_non)):
    im = haar_like_features(train_non[i])
    characteristic_train_non.append(im)
    
pickle_it(characteristic_train_non, "characteristic_train_non2.pickle")


for i in range(len(test_face)):
    im = haar_like_features(test_face[i])
    characteristic_test_face.append(im)

pickle_it(characteristic_test_face, "characteristic_test_face.pickle")


for j in range(0, 10):
    for i in range(int(j*len(test_non)/10), int((j+1)*len(test_non)/10)):
        im = haar_like_features(test_non[i])
        characteristic_test_non.append(im)

    pickle_it(characteristic_test_non, "characteristic_test_non"+str(j)+".pickle")
    characteristic_test_non = []
'''

characteristic_train_face = unpickle_it("characteristic_train_face.pickle")
#characteristic_train_face = np.array(characteristic_train_face)

#characteristic_train_non+str(non1, non2)
characteristic_train_non = unpickle_it("characteristic_train_non1.pickle")
#characteristic_train_non = np.array(characteristic_train_non)

#объединяем два класса и перемешиваем
train = characteristic_train_face+characteristic_train_non
train = np.array(train)
labels = np.hstack((labels_train_face, labels_train_non))
labels = labels.ravel()
ind = list(range(len(train)))
np.random.shuffle(ind) 
train = train[ind]
labels = labels[ind]
#создаем адабуст и сохраняем
b = adab(train, labels)
pickle_it(b, "adaboost.pickle")
#обучаем адабуст и получаем сильный классификатор и сохраняем
b = adab_learn(b, train, labels)

pickle_it(b, "adaboost1.pickle")
#b=unpickle_it("adaboost.pickle")
'''
#вычисляем точность и TP и FN для тестовой выборки
characteristic_test_face = unpickle_it("characteristic_test_face.pickle")
characteristic_test_face = np.array(characteristic_test_face)

#characteristic_test_non+str(non0, non1, non2, non3, non4, non5, non6, non7, non8, non9)
characteristic_test_non = unpickle_it("characteristic_test_non0.pickle")
characteristic_test_non = np.array(characteristic_test_non)
test = characteristic_test_face + characteristic_test_non
labels_t = np.hstack((labels_test_face, labels_test_non))
labels_t = labels_t.ravel()
a = accur(b, test, labels_t)
print(a)
'''
'''
#проверка изображения после стандартизации
imgn = cv.normalize(train_face[0], 0, 255, cv.NORM_MINMAX)
plt.figure()
plt.imshow(imgn)
plt.show()
'''