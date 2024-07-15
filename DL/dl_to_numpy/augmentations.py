import numpy as np
import cv2 as cv
from scipy import ndimage
from registry import Registry


REGISTRY_TYPE = Registry()


@REGISTRY_TYPE.register_module
class Pad(object):
    def __init__(self, image_size=0, fill=0, mode='constant'):
        """
        :param image_size (int or tuple): размер итогового изображения. Если одно число, на выходе будет
        квадратное изображение. Если 2 числа - прямоугольное.
        :param fill (int or tuple): значение, которым будет заполнены поля. Если одно число, все каналы будут заполнены
        этим числом. Если 3 - соответственно по каналам.
        :param mode (string): тип заполнения:
        constant: все поля будут заполнены значение fill;
        edge: все поля будут заполнены пикселями на границе;
        reflect: отображение изображения по краям (прим. [1, 2, 3, 4] => [3, 2, 1, 2, 3, 4, 3, 2])
        symmetric: симметричное отображение изображения по краям (прим. [1, 2, 3, 4] => [2, 1, 1, 2, 3, 4, 4, 3])
        """
        self.image_size = image_size
        self.fill = np.array([fill])
        self.mode = mode

    def __call__(self, image):
        if len(image) > len(self.fill):
            s = self.fill
            for i in range(len(image) - len(s)):
                self.fill = np.concatenate((self.fill, s))
        images = []
        for i in range(len(image)):
            img = np.pad(image[i], self.image_size, self.mode, constant_values=self.fill[i])
            images.append(img)
        return np.asarray(images)


@REGISTRY_TYPE.register_module
class Translate(object):
    def __init__(self, shift=10, direction='right', roll=True):
        """
        :param shift (int): количество пикселей, на которое необходимо сдвинуть изображение
        :param direction (string): направление (['right', 'left', 'down', 'up'])
        :param roll (bool): Если False, не заполняем оставшуюся часть. Если True, заполняем оставшимся краем.
        (прим. False: [1, 2, 3]=>[0, 1, 2]; True: [1, 2, 3] => [3, 1, 2])
        """
        self.shift = shift
        self.direction = direction
        self.roll = roll
        self.dir = dict(right=shift, left=-shift, down=shift, up=-shift)
        self.axes = dict(right=1, left=1, down=0, up=0)

    def __call__(self, image):
        images = []
        for i in range(len(image)):
            img = np.roll(image[i], self.dir[self.direction], axis=self.axes[self.direction])
            images.append(img)
        images = np.asarray(images)
        if not self.roll:
            if self.direction == 'right':
                images[:, :, :self.dir[self.direction]] = 0
            elif self.direction == 'left':
                images[:, :,  self.dir[self.direction]:] = 0
            elif self.direction == 'down':
                images[:, :self.dir[self.direction], :] = 0
            elif self.direction == 'up':
                images[:, self.dir[self.direction]:, :] = 0

        return images


@REGISTRY_TYPE.register_module
class Scale(object):
    def __init__(self, image_size, scale):
        """
        :param image_size (int): размер вырезанного изображения (по центру).
        :param scale (float): во сколько раз увеличить изображение.
        """
        self.image_size = image_size
        self.scale = scale

    def __call__(self, image):
        center = np.array(image[0].shape)
        center = center // 2
        width = int(self.image_size * self.scale / 100)
        height = int(self.image_size * self.scale / 100)

        dim = (width, height)

        up = max(0, int(center[0] - self.image_size // 2))
        down = min(image.shape[1], int(center[0] + self.image_size // 2))
        left = max(0, int(center[1] - self.image_size // 2))
        right = min(image.shape[2], int(center[1] + self.image_size // 2))

        if len(image) == 1:
            img = image[0, up:down, left:right]
            images = cv.resize(img, dim)
            images = images.reshape(1, images.shape[0], images.shape[1])
        else:
            image = np.transpose(image, (1, 2, 0))
            images = cv.resize(image[up:down, left:right, :], dim)
            images = np.transpose(images, (2, 0, 1))
        return images


@REGISTRY_TYPE.register_module
class RandomCrop(object):
    def __init__(self, crop_size):
        """
        :param crop_size (int or tuple): размер вырезанного изображения.
        """
        self.crop_size = np.array([crop_size])
        if len(self.crop_size.shape) == 1:
            self.crop_size = np.concatenate((self.crop_size,  self.crop_size))
        else:
            self.crop_size = self.crop_size[0]

    def __call__(self, image):
        max_y = max(1, image[0].shape[0] - self.crop_size[0])
        max_x = max(1, image[0].shape[1] - self.crop_size[1])

        x = np.random.randint(low=0, high=max_x)
        y = np.random.randint(low=0, high=max_y)

        img = image[:, y: y + self.crop_size[0], x: x + self.crop_size[1]]
        return img


@REGISTRY_TYPE.register_module
class CenterCrop(object):
    def __init__(self, crop_size):
        """
        :param crop_size (int or tuple): размер вырезанного изображения (вырезать по центру).
        """
        self.crop_size = np.array([crop_size])
        if len(self.crop_size.shape) == 1:
            self.crop_size = np.concatenate((self.crop_size,  self.crop_size))
        else:
            self.crop_size = self.crop_size[0]

    def __call__(self, image):
        center = np.array(image[0].shape)
        center = center // 2
        up = max(0, int(center[0] - self.crop_size[0] // 2))
        down = min(image.shape[1], int(center[0] + self.crop_size[0] // 2))
        left = max(0, int(center[1] - self.crop_size[1] // 2))
        right = min(image.shape[2], int(center[1] + self.crop_size[1] // 2))
        img = image[:, up:down, left:right]
        return img


@REGISTRY_TYPE.register_module
class RandomRotateImage(object):
    def __init__(self, min_angle, max_angle):
        """
        :param min_angle (int): минимальный угол поворота.
        :param max_angle (int): максимальный угол поворота.
        Угол поворота должен быть выбран равномерно из заданного промежутка.
        """
        self.min_angle = min_angle
        self.max_angle = max_angle

    def __call__(self, image):
        rotate = np.random.randint(self.min_angle, self.max_angle)
        image = np.transpose(image, (1, 2, 0))
        img = ndimage.rotate(image, rotate, reshape=False)
        img = np.transpose(img, (2, 0, 1))
        return img


@REGISTRY_TYPE.register_module
class GaussianNoise(object):
    def __init__(self, mean=0, sigma=0.03, by_channel=False):
        """
        :param mean (int): среднее значение.
        :param sigma (int): максимальное значение ско. Итоговое значение должно быть выбрано равномерно в промежутке
        [0, sigma].
        :param by_channel (bool): если True, то по каналам независимо.
        """
        self.mean = mean
        self.sigma = sigma
        self.by_channel = by_channel

    def __call__(self, image):
        if self.by_channel:
            for i in range(len(image)):
                gauss = np.random.normal(self.mean, self.sigma, image[i].shape)
                image[i] += gauss
        else:
            gauss = np.random.normal(self.mean, self.sigma, image[0].shape)
            image = image.astype(dtype='float')
            image += gauss
            image = image.astype(np.uint8)
        return image


@REGISTRY_TYPE.register_module
class Salt(object):
    def __init__(self, prob, by_channel=False):
        """
        :param prob (float): вероятность, с которой пиксели будут заполнены белым.
        :param by_channel (bool): если True, то по каналам независимо.
        """
        self.prob = prob
        self.by_channel = by_channel

    def __call__(self, image):
        if self.by_channel:
            num_salt = np.ceil(image.size * self.prob)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            image[coords[0], coords[1], coords[2]] = 255
        else:
            num_salt = np.ceil(image[0].size * self.prob)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image[0].shape]
            image[:, coords[0], coords[1]] = 255

        return image


@REGISTRY_TYPE.register_module
class Pepper(object):
    def __init__(self, prob, by_channel=False):
        """
        :param prob (float): вероятность, с которой пиксели будут заполнены черным.
        :param by_channel (bool): если True, то по каналам независимо.
        """
        self.prob = prob
        self.by_channel = by_channel

    def __call__(self, image):
        if self.by_channel:
            num_pepper = np.ceil(image.size * self.prob)
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            image[coords[0], coords[1], coords[2]] = 0
        else:
            num_pepper = np.ceil(image[0].size * self.prob)
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image[0].shape]
            image[:, coords[0], coords[1]] = 0
        return image


@REGISTRY_TYPE.register_module
class ChangeBrightness(object):
    def __init__(self, value=30, type='brightness'):
        """
        :param value (int): насколько изменить яркость. Аналогично hue, contrast, saturation.
        :param type (string): один из [brightness, hue, contrast, saturation].
        """
        self.value = value
        self.type = type

    def __call__(self, image):
        if len(image) <= 1:
            return None

        image = np.transpose(image, (1, 2, 0))
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        if self.type == 'hue':
            hsv[0] += self.value
            img_m = hsv > 255
            hsv[img_m] = 255
        elif self.type == 'saturation':
            hsv[1] += self.value
            img_m = hsv > 255
            hsv[img_m] = 255
        elif self.type == 'brightness':
            hsv[2] += self.value
            img_m = hsv > 255
            hsv[img_m] = 255

        img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        img = np.transpose(img, (2, 0, 1))

        if self.type == 'contrast':
            img *= self.value
            img_m = img > 255
            img[img_m] = 255

        return img


@REGISTRY_TYPE.register_module
class GaussianBlur(object):
    def __init__(self, ksize=(5, 5)):
        """
        :param ksize (tuple): размер фильтра.
        """
        self.ksize = tuple(ksize)

    def __call__(self, image):
        image = cv.GaussianBlur(image, self.ksize, 1)
        return image


@REGISTRY_TYPE.register_module
class Normalize(object):
    def __init__(self, mean=128, var=255):
        """
        :param mean (int or tuple): среднее значение (пикселя), которое необходимо вычесть.
        :param var (int): значение, на которое необходимо поделить.
        """
        self.mean = np.array([mean])
        self.var = var

    def __call__(self, image):
        if len(self.mean) == 1:
            image -= self.mean[0]
        else:
            self.mean = self.mean.reshape(image.shape[0], 1, 1)
            image -= self.mean

        img_zeros = image < 0
        image[img_zeros] = 0

        image = image.astype(dtype='float')
        image /= self.var
        image *= 255
        image[image > 255] = 255
        image = image.astype(np.uint8)
        return image
