import numpy as np
import gzip
import tarfile
import pickle
import matplotlib.pyplot as plt


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class Dataset_MNIST(object):
    def __init__(self, data_path, dataset_type, transforms, nrof_classes, random_aug=False, aug_prob=None):
        """
        :param data_path (string): путь до файла с данными.
        :param dataset_type (string): (['train', 'test']).
        :param transforms (list): список необходимых преобразований изображений.
        :param nrof_classes (int): количество классов в датасете.
        """

        self.data_path = data_path
        self.dataset_type = dataset_type
        self.transforms = transforms
        self.nrof_classes = nrof_classes
        self.data = None
        self.label = None
        self.random_aug = random_aug
        self.aug_prob = aug_prob

    def read_data(self):
        """
        Считывание данных по заданному пути+вывод статистики.
        """
        if self.dataset_type == 'train':

            data = gzip.open(self.data_path + '//train-images-idx3-ubyte.gz', 'rb')
            labels = gzip.open(self.data_path + '//train-labels-idx1-ubyte.gz', 'rb')
            size_data = 60000

        else:
            data = gzip.open(self.data_path + '//t10k-images-idx3-ubyte.gz', 'rb')
            labels = gzip.open(self.data_path + '//t10k-labels-idx1-ubyte.gz', 'rb')
            size_data = 10000

        data.read(16)
        labels.read(8)
        image_size = 28

        buf_im = data.read(size_data * image_size * image_size)
        data_im = np.frombuffer(buf_im, dtype=np.uint8).astype(np.float32)

        buf_label = labels.read(size_data)
        data_label = np.frombuffer(buf_label, dtype=np.uint8).astype(np.float32)

        self.data = data_im.reshape(size_data, 1, image_size, image_size)
        self.label = data_label

        indx = np.argsort(self.label)
        self.label = self.label[indx]
        self.data = self.data[indx]

        data.close()
        labels.close()

    def __len__(self):
        """
        :return: размер выборки
        """
        return len(self.label)

    def one_hot_labels(self, label):
        """
        :param label: метка класса
        :return: one-hot encoding вектор
        """
        one_hot = np.zeros(self.nrof_classes)
        one_hot[int(label)] = 1
        return one_hot

    def __transforms(self, img):
        if len(self.transforms) > 0:
            if not self.random_aug:
                for func in self.transforms:
                    img = func(img)
            else:
                i = np.random.choice(len(self.transforms), 1, p=self.aug_prob)
                img = self.transforms[int(i)](img)
        return img

    def __getitem__(self, idx):
        """
        :param idx: индекс элемента в выборке
        :return: preprocessed image and label
        """
        if isinstance(idx, np.ndarray):
            img = np.array(list(map(lambda i: self.__transforms(self.data[i]), idx)))
            label = np.array(list(map(lambda i: self.one_hot_labels(self.label[i]), idx)))
            return img, label

        img = self.__transforms(self.data[idx])
        label = self.one_hot_labels(self.label[idx])
        return img, label

    def show_statistics(self):
        """
        Необходимо вывести количество элементов в датасете, количество классов и количество элементов в каждом классе
        """
        class_size, _ = np.histogram(self.label, bins=np.arange(self.nrof_classes+1))
        print("Size dataset: {} \nNumber of classes: {} \nNumber of items in each class: {}".format(
               self.__len__(), self.nrof_classes, class_size))

    def show_transforms(self, transforms, idx):
        images = []
        labels = []
        for func in transforms:
            img = self.data[idx].copy()
            img = func(img)
            images.append(img)
            labels.append(self.label[idx])

        fig = plt.figure(figsize=(28, 28))
        columns = np.ceil(len(images) / 2)
        for i in range(0, len(images)):
            img = images[i]
            label = labels[i]
            img = img.astype(dtype='int')
            fig.add_subplot(2, columns, i + 1)
            # plt.title(str(label))
            plt.title(transforms[i].__class__.__name__)
            if len(img) == 1:
                plt.imshow(img[0])
            else:
                plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.show()


class Dataset_CIFAR10(object):
    def __init__(self, data_path, dataset_type, transforms, nrof_classes, random_aug=False, aug_prob=None):
        """
        :param data_path (string): путь до файла с данными.
        :param dataset_type (string): (['train', 'valid', 'test']).
        :param transforms (list): список необходимых преобразований изображений.
        :param nrof_classes (int): количество классов в датасете.
        """
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.transforms = transforms
        self.nrof_classes = nrof_classes
        self.data = None
        self.label = None
        self.random_aug = random_aug
        self.aug_prob = aug_prob

    def read_data(self):
        """
        Считывание данных по заданному пути+вывод статистики.
        """

        tar = tarfile.open(self.data_path + "//cifar-10-python.tar.gz")
        tar.extractall(path=self.data_path)
        tar.close()

        image_size = 32

        for i in range(1, 6):
            dataset = unpickle(self.data_path + '//cifar-10-batches-py//data_batch_' + str(i))
            label = dataset[b'labels']
            data = dataset[b'data']
            size = len(data)
            data = data.reshape(size, 3, image_size, image_size)

            if self.data is None:
                self.data = data
                self.label = label
            else:
                self.data = np.concatenate((self.data, data), axis=0)
                self.label = np.concatenate((self.label, label), axis=0)

        self.label = np.asarray(self.label)
        self.data = np.asarray(self.data)

        indx = np.argsort(self.label)
        self.label = self.label[indx]
        self.data = self.data[indx]

    def __len__(self):
        """
        :return: размер выборки
        """
        return len(self.label)

    def one_hot_labels(self, label):
        """
        :param label: метка класса
        :return: one-hot encoding вектор
        """
        one_hot = np.zeros(self.nrof_classes)
        one_hot[int(label)] = 1
        return one_hot

    def __transforms(self, img):
        if len(self.transforms) > 0:
            if not self.random_aug:
                for func in self.transforms:
                    img = func(img)
            else:
                i = np.random.choice(len(self.transforms), 1, p=self.aug_prob)
                img = self.transforms[int(i)](img)
        return img

    def __getitem__(self, idx):
        """
        :param idx: индекс элемента в выборке
        :return: preprocessed image and label
        """
        if isinstance(idx, np.ndarray):
            img = np.array(list(map(lambda i: self.__transforms(self.data[i]), idx)))
            label = np.array(list(map(lambda i: self.one_hot_labels(self.label[i]), idx)))
            return img, label

        img = self.__transforms(self.data[idx])
        label = self.one_hot_labels(self.label[idx])
        return img, label

    def show_statistics(self):
        """
        Необходимо вывести количество элементов в датасете, количество классов и количество элементов в каждом классе
        """
        class_size, _ = np.histogram(self.label, bins=np.arange(self.nrof_classes+1))
        print("Size dataset: {} \nNumber of classes: {} \nNumber of items in each class: {}".format(
               self.__len__(), self.nrof_classes, class_size))

    def show_transforms(self, transforms, idx):
        images = []
        labels = []
        for func in transforms:
            img = self.data[idx].copy()
            img = func(img)
            images.append(img)
            labels.append(self.label[idx])

        fig = plt.figure(figsize=(28, 28))
        columns = np.ceil(len(images) / 2)
        for i in range(0, len(images)):
            img = images[i]
            label = labels[i]
            img = img.astype(dtype='int')
            fig.add_subplot(2, columns, i + 1)
            # plt.title(str(label))
            plt.title(transforms[i].__class__.__name__)
            if len(img) == 1:
                plt.imshow(img[0])
            else:
                plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.show()
