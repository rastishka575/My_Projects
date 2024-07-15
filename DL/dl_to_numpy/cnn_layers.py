from layers import *


class Convolution(BaseLayerClass):
    def __init__(self, kernel_size, nrof_filters, kernel_depth=1, zero_pad=0, stride=1,
                 use_bias=False, initialization_form='normal', initialization_type='he'):
        self.kernel_size = kernel_size
        self.nrof_filters = nrof_filters
        self.kernel_depth = kernel_depth
        self.zero_pad = zero_pad
        self.stride = stride
        self.use_bias = use_bias
        self.initialization_form = initialization_form
        self.initialization_type = initialization_type
        self.init_weights()

        if self.use_bias:
            self.bias = np.zeros(self.nrof_filters) if self.nrof_filters > 1 else 0

        self.train = True
        self.v_w = 0
        self.v_b = 0

    def trainable(self):
        return True

    def get_nrof_trainable_params(self):
        return self.weight.shape

    def get_params(self):
        if self.use_bias:
            return self.weight, self.bias
        else:
            return self.weight

    def __initialization_form_normal(self):
        sigma = 1
        if self.initialization_type == 'he':
            sigma = np.sqrt(2 / self.kernel_size * self.kernel_size * self.kernel_depth)

        elif self.initialization_type == 'xavier':
            sigma = np.sqrt(1 / self.kernel_size * self.kernel_size * self.kernel_depth)

        elif self.initialization_type == 'glorot':
            sigma = np.sqrt(1 / (self.kernel_size * self.kernel_size * self.kernel_depth + self.nrof_filters))

        self.weight = np.sqrt(sigma) * np.random.randn(self.nrof_filters, self.kernel_depth, self.kernel_size, self.kernel_size)

    def __initialization_form_uniform(self):
        l = 1
        if self.initialization_type == 'he':
            l = np.sqrt(6 / self.kernel_size * self.kernel_size * self.kernel_depth)

        elif self.initialization_type == 'xavier':
            l = np.sqrt(3 / self.kernel_size * self.kernel_size * self.kernel_depth)

        elif self.initialization_type == 'glorot':
            l = np.sqrt(6 / (self.kernel_size * self.kernel_size * self.kernel_depth + self.nrof_filters))

        self.weight = np.random.uniform(-l, l, (self.nrof_filters, self.kernel_depth, self.kernel_size, self.kernel_size))

    def init_weights(self):
        if self.initialization_form == 'normal':
            self.__initialization_form_normal()
        elif self.initialization_form == 'uniform':
            self.__initialization_form_uniform()

    def __call__(self, input):
        self.input_width = input.shape[3]
        self.input_height = input.shape[2]
        self.input_depth = input.shape[1]

        self.output_width = int((input.shape[3] - self.kernel_size + 2 * self.zero_pad) / self.stride + 1)
        self.output_height = int((input.shape[2] - self.kernel_size + 2 * self.zero_pad) / self.stride + 1)
        self.output_depth = self.nrof_filters
        if self.zero_pad > 0:
            self.input = np.zeros((input.shape[0], input.shape[1],
                                   input.shape[2] + 2*self.zero_pad, input.shape[3] + 2*self.zero_pad))
            self.input[:, :, self.zero_pad: input.shape[2] + self.zero_pad,
                        self.zero_pad: input.shape[3] + self.zero_pad] = input
        else:
            self.input = input

        self.k = np.expand_dims(np.repeat(np.arange(self.input.shape[1]), self.output_width * self.output_height), axis=1)

        self.i = np.repeat(np.expand_dims(np.repeat(np.tile(np.arange(self.kernel_size), self.input.shape[1]),
                                        self.kernel_size), axis=0), self.output_width * self.output_height, axis=0)
        self.i += np.expand_dims(np.repeat(np.arange(0, self.output_height*self.stride, self.stride), self.output_width),
                                 axis=0).T

        self.i = np.tile(self.i, (self.input.shape[1], 1))

        self.j = np.repeat(np.expand_dims(np.tile(np.tile(np.arange(self.kernel_size), self.input.shape[1]),
                                        self.kernel_size), axis=0), self.output_width * self.output_height, axis=0)
        self.j += np.expand_dims(np.tile(np.arange(0, self.output_height*self.stride, self.stride), self.output_width),
                                 axis=0).T
        self.j = np.tile(self.j, (self.input.shape[1], 1))

        self.kernel = self.weight.reshape(self.nrof_filters, self.kernel_depth*self.kernel_size*self.kernel_size)

        self.kernel = np.tile(self.kernel, self.input_depth // self.kernel_depth)

        self.input = self.input[:, self.k, self.i, self.j]

        self.input = self.input.transpose(0, 2, 1)

        self.output = self.kernel @ self.input
        if self.use_bias:
            self.kernel_bias = np.repeat(np.expand_dims(np.expand_dims(self.bias, axis=1), axis=0), input.shape[0], axis=0)
            self.output = self.output + self.kernel_bias

        self.output = self.output.reshape(input.shape[0], self.output_depth, self.input_depth, self.output_height, self.output_width)
        self.output = self.output.sum(2)
        return self.output

    def get_grad(self):
        self.weight_grad = self.input
        self.bias_grad = 1
        self.grad = self.weight.reshape(self.nrof_filters, self.kernel_depth*self.kernel_size*self.kernel_size)

    def backward(self, dy):
        self.delta = dy.reshape(dy.shape[0], self.output_depth, self.output_height * self.output_width)
        grad = self.grad.T @ self.delta
        grad = grad.transpose(0, 2, 1)
        grad = grad.transpose(1, 2, 0)
        dy_grad = np.zeros((self.input_depth, self.input_height, self.input_width, dy.shape[0]))
        np.add.at(dy_grad, [self.k, self.i, self.j], grad)
        dy_grad = dy_grad.transpose(3, 0, 1, 2)
        if self.zero_pad > 0:
            dy_grad = dy_grad[:, :, self.zero_pad:dy_grad.shape[2]-self.zero_pad,
                      self.zero_pad:dy_grad.shape[3]-self.zero_pad]
        return dy_grad

    def update_weights(self, update_func):
        if self.train:
            grad = self.delta @ self.input.transpose(0, 2, 1)
            grad = grad.reshape(grad.shape[0], self.nrof_filters, self.kernel_depth, self.kernel_size, self.kernel_size)
            self.v_w = update_func(grad, self.v_w)
            self.weight += self.v_w
            if self.use_bias:
                self.v_b = update_func(self.delta.mean(0) * self.bias_grad, self.v_b)
                self.bias += self.v_b

    def zero_grad(self):
        self.weight_grad = 0
        self.bias_grad = 0
        self.grad = 0
        self.delta = 0
        self.kernel = 0
        self.input = 0
        self.output = 0
        self.k = 0
        self.i = 0
        self.j = 0
        self.kernel_bias = 0
        self.v_w = 0
        self.v_b = 0


class Pooling(BaseLayerClass):
    def __init__(self, kernel_size, stride, type='Max'):
        self.kernel_size = kernel_size
        self.stride = stride
        self.type = type
        self.train = False

    def trainable(self):
        return False

    def __call__(self, input):
        self.input = input
        self.input_width = input.shape[3]
        self.input_height = input.shape[2]
        self.input_depth = input.shape[1]

        self.output_width = int((input.shape[3] - self.kernel_size) / self.stride + 1)
        self.output_height = int((input.shape[2] - self.kernel_size) / self.stride + 1)
        self.output_depth = input.shape[1]

        self.k = np.expand_dims(np.repeat(np.arange(self.input.shape[1]), self.output_width * self.output_height), axis=1)

        self.i = np.repeat(np.expand_dims(np.repeat(np.tile(np.arange(self.kernel_size), self.input.shape[1]),
                                        self.kernel_size), axis=0), self.output_width * self.output_height, axis=0)
        self.i += np.expand_dims(np.repeat(np.arange(0, self.output_height * self.stride, self.stride), self.output_width),
                                        axis=0).T

        self.i = np.tile(self.i, (self.input.shape[1], 1))

        self.j = np.repeat(np.expand_dims(np.tile(np.tile(np.arange(self.kernel_size), self.input.shape[1]),
                                        self.kernel_size), axis=0), self.output_width * self.output_height, axis=0)
        self.j += np.expand_dims(np.tile(np.arange(0, self.output_height * self.stride, self.stride), self.output_width),
                                        axis=0).T

        self.j = np.tile(self.j, (self.input.shape[1], 1))

        self.input = self.input[:, self.k, self.i, self.j]

        self.input = self.input.transpose(0, 2, 1)

        self.output = self.input.max(1) if self.type == 'Max' else self.input.mean(1)

        self.output = self.output.reshape(self.output.shape[0], self.output_depth, self.output_height, self.output_width)

        return self.output

    def get_grad(self):
        self.input = self.input.transpose(0, 2, 1)
        self.grad = np.zeros_like(self.input)
        if self.type == 'Max':
            index = self.input.argmax(2)
            y = np.expand_dims(np.arange(self.grad.shape[0]), axis=0)
            x = np.expand_dims(np.arange(self.grad.shape[1]), axis=1)
            z = np.expand_dims(index.transpose(1, 0), axis=0)
            self.grad[y, x, z] = 1
        elif self.type == 'Avg':
            self.grad = np.ones_like(self.input)

    def backward(self, dy):
        self.grad = self.grad.transpose(1, 2, 0)
        dy_grad = np.zeros((self.input_depth, self.input_height, self.input_width, dy.shape[0]))
        np.add.at(dy_grad, [self.k, self.i, self.j], self.grad)
        dy_grad = dy_grad.transpose(3, 0, 1, 2)
        if self.type == 'Avg':
            dy_grad /= (self.kernel_size*self.kernel_size)
        dy = np.repeat(np.repeat(dy, self.stride, axis=2), self.stride, axis=3)
        dy *= dy_grad
        return dy

    def zero_grad(self):
        self.grad = 0
        self.kernel = 0
        self.input = 0
        self.output = 0
        self.k = 0
        self.i = 0
        self.j = 0