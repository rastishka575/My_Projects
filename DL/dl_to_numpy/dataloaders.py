import datasets
import numpy as np
import matplotlib.pyplot as plt


class DataLoader(object):
    def __init__(self, dataset, nrof_classes, dataset_type, shuffle, batch_size,
                 sample_type, epoch_size=None, probabilities=None):
        """
        :param dataset (Dataset): объект класса Dataset.
        :param nrof_classes (int): количество классов в датасете.
        :param dataset_type (string): (['train', 'test']).
        :param shuffle (bool): нужно ли перемешивать данные после очередной эпохи.
        :param batch_size (int): размер батча.
        :param sample_type (string): (['default' - берем последовательно все данные, 'balanced' - сбалансированно,
        'prob' - сэмплирем с учетом указанных вероятностей])
        :param epoch_size (int or None): размер эпохи. Если None, необходимо посчитать размер эпохи (=размеру обучающей выюорки/batch_size)
        :param probabilities (array or None): в случае sample_type='prob' вероятности, с которыми будут выбраны элементы из каждого класса.
        """
        self.dataset = dataset
        self.nrof_classes = nrof_classes
        self.dataset_type = dataset_type
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_type = sample_type
        if epoch_size is None:
            self.epoch_size = np.ceil(len(self.dataset)/self.batch_size)
        else:
            self.epoch_size = epoch_size
        self.probabilities = probabilities

        if sample_type == 'default':
            self.index = np.arange(len(self.dataset))

        elif sample_type == 'upsample':
            self.index = np.arange(len(self.dataset))
            class_size, _ = np.histogram(self.dataset.label, bins=np.arange(self.nrof_classes + 1))
            number_max = class_size.max()
            step = 0
            for i in range(nrof_classes):
                class_plus = number_max - class_size[i]
                if class_plus != 0:
                    index = np.arange(class_size[i]) + step
                    step += class_size[i]
                    ind = np.random.choice(index, class_plus)
                    self.index = np.concatenate((self.index, ind), axis=0)

        elif sample_type == 'downsample':
            class_size, _ = np.histogram(self.dataset.label, bins=np.arange(self.nrof_classes + 1))
            class_step = np.cumsum(class_size)
            number_min = class_size.min()
            self.index = np.arange(number_min)
            for i in range(1, nrof_classes):
                index = np.arange(class_size[i]) + class_step[i-1]
                ind = index[:number_min]
                self.index = np.concatenate((self.index, ind), axis=0)

        elif sample_type == 'prob':
            if probabilities is None:
                probabilities, _ = np.histogram(self.dataset.label, bins=np.arange(self.nrof_classes + 1))
                probabilities = probabilities.astype(dtype='float')
                probabilities /= len(self.dataset)
                self.probabilities = np.round(probabilities, 2)
                print(self.probabilities)
            ind = np.random.choice(nrof_classes, len(self.dataset), p=probabilities)
            class_size, _ = np.histogram(dataset.label, bins=np.arange(self.nrof_classes + 1))
            class_step = np.cumsum(class_size)
            step = np.zeros(nrof_classes)
            index = []
            for i in range(len(ind)):
                index.append((class_step[ind[i]]+step[ind[i]])%class_step[ind[i]])
                step[ind[i]] += 1
            self.index = np.asarray(index)
            self.index = self.index.astype(dtype='int')

    def batch_generator(self):
        """
        Создание батчей на эпоху с учетом указанного размера эпохи и типа сэмплирования.
        """
        if self.shuffle:
            order = np.random.permutation(len(self.index))
            self.index = self.index[order]

        i = 0
        for start_batch in range(0, len(self.dataset), self.batch_size):
            if i == self.epoch_size:
                break
            i += 1
            index = self.index[start_batch:start_batch+self.batch_size]
            self.batch_index = index
            yield self.dataset[index]

    def show_batch(self):
        """
        Необходимо визуализировать и сохранить изображения в батче (один батч - одно окно). Предварительно привести значение в промежуток
        [0, 255) и типу к uint8
        :return:
        """
        fig = plt.figure(figsize=(28, 28))
        columns = np.ceil(self.batch_size/2)
        for i in range(0, self.batch_size):
            img, label = self.dataset[self.batch_index[i]]
            #img = img.astype(dtype='float')
            #img /= img.max()
            #img *= 255
            #img = img.astype(np.uint8)
            fig.add_subplot(2, columns, i+1)
            plt.title(str(label))
            if len(img) == 1:
                plt.imshow(img[0])
            else:
                plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.show()
