import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import pickle


class MnistData:
    def __init__(self, model_train=True, nb_classes=10):

        self.nb_classes = nb_classes

        if model_train:
            train_data = pd.read_csv('dataset/mnist_train.csv').values

            shuffle_index = np.arange(train_data.shape[0])
            np.random.shuffle(shuffle_index)
            split_index = int(0.8 * shuffle_index.shape[0])

            self.train_images = self.normalization(train_data[:split_index, 1:])
            self.train_labels = train_data[:split_index, 0]

            self.valid_images = self.normalization(train_data[split_index:, 1:])
            self.valid_labels = train_data[split_index:, 0]

            self.train_labels = self.one_hot_encoding(self.train_labels)
            self.valid_labels = self.one_hot_encoding(self.valid_labels)

        test_data = pd.read_csv('dataset/mnist_test.csv').values
        self.test_images = self.normalization(test_data[:, 1:])
        self.test_labels = test_data[:, 0]
        self.test_labels = self.one_hot_encoding(self.test_labels)

    def one_hot_encoding(self, labels):
        one_hot_targets = np.eye(self.nb_classes)[labels.reshape(-1)]
        return one_hot_targets

    @staticmethod
    def normalization(data):
        return 2 * (data - data.min(axis=1, keepdims=True)) \
            / (data.max(axis=1, keepdims=True) - data.min(axis=1, keepdims=True)) - 1

    def __len__(self):
        return self.test_images.shape[1]


class LogisticRegression:
    def __init__(self, length, nb_classes):
        self.weights = np.random.rand(nb_classes, length)
        self.bias = np.zeros(nb_classes)
        self.weights_gradient = None
        self.bias_gradient = None
        self.accuracy_logs = []

    def forward(self, x):
        return (self.weights@x.T).T + self.bias

    @staticmethod
    def softmax(z):
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    def backward(self, y, target, x):
        self.weights_gradient = ((y - target).reshape(y.shape[0], y.shape[1], 1) @
                                 x.reshape(x.shape[0], 1, x.shape[1])).mean(axis=0)
        self.bias_gradient = (y - target).mean(axis=0)
        return self.weights_gradient, self.bias_gradient

    def gradient_step(self, learning_rate=0.1):
        self.weights = self.weights - learning_rate * self.weights_gradient
        self.bias = self.bias - learning_rate * self.bias_gradient
        return self.weights, self.bias

    def predict(self, x):
        return self.softmax(self.forward(x))

    def fit(self, train_images, train_labels, valid_images, valid_labels, learning_rate=0.1, eps=0.001):
        while True:
            train_predict = self.predict(train_images)
            self.backward(train_predict, train_labels, train_images)
            weights_prev, bias_prev = self.weights, self.bias
            self.gradient_step(learning_rate)
            valid_predict = self.predict(valid_images)
            accuracy_valid = accuracy_compute(valid_predict, valid_labels)
            self.accuracy_logs.append(accuracy_valid)
            if np.linalg.norm(self.weights - weights_prev) < eps * (np.linalg.norm(self.weights) + learning_rate):
                break


def confusion_matrix_compute(predict_one_hot_encoding, label_one_hot_encoding, nb_classes):
    predict = predict_one_hot_encoding.argmax(axis=1)
    label = label_one_hot_encoding.argmax(axis=1)
    confusion_matrix = np.zeros((nb_classes, nb_classes))
    for i in range(nb_classes):
        confusion_classes = label == i
        values, counts = np.unique(predict[confusion_classes], return_counts=True)
        confusion_matrix[i, values] += counts
    return confusion_matrix


def accuracy_compute(predict, label):
    accuracy_score = np.sum(predict.argmax(axis=1) == label.argmax(axis=1))/label.shape[0]
    return accuracy_score.round(2)*100


def precision_compute(confusion_matrix):
    return np.mean(np.diag(confusion_matrix)/np.sum(confusion_matrix, axis=0))


def recall_compute(confusion_matrix):
    return np.mean(np.diag(confusion_matrix)/np.sum(confusion_matrix, axis=1))


def f1_score_compute(precision, recall):
    return 2*precision*recall/(precision+recall)


def dump_model(model, filename_model):
    pickle.dump(model, open(filename_model, 'wb'))


def load_model(filename_model):
    return pickle.load(open(filename_model, 'rb'))


if __name__ == '__main__':

    random.seed(0)
    np.random.seed(0)

    train = False
    filename = 'lr.pickle'

    dataset = MnistData(model_train=train)

    if train:
        model_lr = LogisticRegression(len(dataset), dataset.nb_classes)

        model_lr.fit(dataset.train_images, dataset.train_labels, dataset.valid_images, dataset.valid_labels)
        dump_model(model_lr, filename)
    else:
        model_lr = load_model(filename)

    accuracy_test = accuracy_compute(model_lr.predict(dataset.test_images), dataset.test_labels)
    confusion_matrix_test = confusion_matrix_compute(model_lr.predict(dataset.test_images),
                                                     dataset.test_labels, dataset.nb_classes)
    precision_test = precision_compute(confusion_matrix_test)
    recall_test = recall_compute(confusion_matrix_test)
    f1_score_test = f1_score_compute(precision_test, recall_test)

    print(accuracy_test)
    print(precision_test)
    print(recall_test)
    print(f1_score_test)
    print(confusion_matrix_test)

    fig, ax = plt.subplots()
    ax.plot(model_lr.accuracy_logs)
    ax.set(xlim=(0, len(model_lr.accuracy_logs)),
           ylim=(0, 100))
    plt.show()
