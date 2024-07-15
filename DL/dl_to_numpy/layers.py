from abc import ABC, abstractmethod
import numpy as np


class BaseLayerClass(object):
    @abstractmethod
    def __call__(self, x):
        return x

    @abstractmethod
    def get_grad(self):
        pass

    def backward(self, dy):
        pass

    def update_weights(self, update_func):
        pass

    def get_nrof_trainable_params(self):
        pass

    @property
    def trainable(self):
        return True

    def get_params(self):
        pass

    def zero_grad(self):
        pass


class Linear(BaseLayerClass):
    def __init__(self, input_shape, output_shape, use_bias=True,
                 initialization_type='he', regularization=False, weight_decay=0.001):

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.use_bias = use_bias
        self.initialization_type = initialization_type
        self.regularization = regularization
        self.weight_decay = weight_decay
        self.train = True
        self.v_w = 0
        self.v_b = 0

        self.__initialization_weight()

        if self.use_bias:
            self.__initialization_bias()
        else:
            self.bias = 0

    def __initialization_weight(self):
        sigma = 1
        if self.initialization_type == 'he':
            sigma = np.sqrt(2 / self.input_shape)

        elif self.initialization_type == 'xavier':
            sigma = np.sqrt(1 / self.input_shape)

        elif self.initialization_type == 'glorot':
            sigma = np.sqrt(1 / (self.input_shape + self.output_shape))

        self.weight = np.sqrt(sigma) * np.random.randn(self.output_shape, self.input_shape)

    def __initialization_bias(self):
        l = 1
        if self.initialization_type == 'he':
            l = np.sqrt(6 / self.input_shape)

        elif self.initialization_type == 'xavier':
            l = np.sqrt(3 / self.input_shape)

        elif self.initialization_type == 'glorot':
            l = np.sqrt(6 / (self.input_shape + self.output_shape))

        self.bias = np.random.uniform(-l, l, self.output_shape)

    def __call__(self, x):
        self.input_value = x
        x = x@self.weight.T + self.bias
        return x

    def get_grad(self):
        self.weight_grad = self.input_value
        self.bias_grad = 1
        self.grad = self.weight

    def backward(self, dy):
        self.delta = dy
        return dy @ self.grad

    def update_weights(self, update_func):
        if self.train:
            grad = np.expand_dims(self.delta, axis=1).transpose(0, 2, 1) @ np.expand_dims(self.weight_grad, axis=1)
            #grad = np.array(list(map(lambda i: np.expand_dims(self.delta[i].T, axis=1) @ np.expand_dims(self.weight_grad[i], axis=0), np.arange(len(self.delta)))))
            self.v_w = update_func(grad, self.v_w)
            self.weight += self.v_w
            if self.regularization:
                self.weight += self.weight_decay * np.linalg.norm(self.v_w)**2
            if self.use_bias:
                self.v_b = update_func(self.delta * self.bias_grad, self.v_b)
                self.bias += self.v_b
                if self.regularization:
                    self.bias += self.weight_decay * np.linalg.norm(self.v_b) ** 2

    def get_nrof_trainable_params(self):
        return self.weight.shape

    def trainable(self):
        return True

    def get_params(self):
        if self.use_bias:
            return self.weight, self.bias
        else:
            return self.weight

    def zero_grad(self):
        self.weight_grad = 0
        self.bias_grad = 0
        self.grad = 0
        self.delta = 0


REGISTRY_MODEL = dict({'Linear': Linear})
