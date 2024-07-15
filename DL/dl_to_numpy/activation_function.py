from abc import ABC, abstractmethod
import numpy as np


class BaseActivationFunctionClass(object):
    def __init__(self):
        self.train = False

    @abstractmethod
    def __call__(self, x):
        return x

    @property
    def trainable(self):
        return False

    def get_grad(self):
        pass

    @abstractmethod
    def backward(self, dy):
        pass

    def zero_grad(self):
        self.grad = 0


class Relu(BaseActivationFunctionClass):
    def __call__(self, x):
        self.grad = x < 0
        x[self.grad] = 0
        return x

    def get_grad(self):
        self.grad = self.grad.astype('int')
        self.grad = 1 - self.grad

    def backward(self, dy):
        return dy * self.grad


class Sigmoid(BaseActivationFunctionClass):
    def __call__(self, x):
        x = np.clip(x, -15, 15)
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def get_grad(self):
        self.grad = (1 - self.output) * self.output

    def backward(self, dy):
        return dy * self.grad


class Tanh(BaseActivationFunctionClass):
    def __call__(self, x):
        x = np.clip(x, -15, 15)
        self.output = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return self.output

    def get_grad(self):
        self.grad = 1 - self.output**2

    def backward(self, dy):
        return dy * self.grad


class Identity(BaseActivationFunctionClass):
    def __call__(self, x):
        return x

    def get_grad(self):
        self.grad = 1

    def backward(self, dy):
        return dy


class Flatten(BaseActivationFunctionClass):
    def __call__(self, x):
        self.shape = x.shape
        return x.reshape(self.shape[0], self.shape[1]*self.shape[2]*self.shape[3])

    def get_grad(self):
        self.grad = 1

    def backward(self, dy):
        return dy.reshape(self.shape[0], self.shape[1], self.shape[2], self.shape[3])


class Softmax(BaseActivationFunctionClass):
    def __call__(self, x):
        e_x = np.exp(np.subtract(x, x.max(axis=1).reshape((len(x), 1))))
        sm = e_x / e_x.sum(axis=1).reshape((len(x), 1))
        return sm

    def get_grad(self):
        pass

    def backward(self, dy):
        return dy