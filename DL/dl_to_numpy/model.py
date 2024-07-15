import numpy as np
import pickle


class BaseModel(object):
    def __init__(self, parameters):
        pass

    def eval(self):
        pass

    def train(self):
        pass

    def __call__(self, input):
        pass

    def get_parameters(self):
        pass

    def dump_model(self, path):
        pass

    def load_weights(self, path):
        pass


class Mlp(BaseModel):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.parameters = parameters

    def eval(self):
        for layer in self.parameters:
            if layer.trainable:
                layer.train = False

    def train(self):
        for layer in self.parameters:
            if layer.trainable:
                layer.train = True

    def __call__(self, input):
        input = np.reshape(input, (input.shape[0], input.shape[1] * input.shape[2] * input.shape[3]))
        for layer in self.parameters:
            input = layer(input)
        return input

    def get_parameters(self):
        param_list = []
        for layer in self.parameters:
            if layer.trainable:
                param_list.append([layer.__class__.__name__, layer.train, layer.get_nrof_trainable_params()])
            else:
                param_list.append([layer.__class__.__name__, layer.trainable, 0])
        return param_list

    def dump_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.parameters, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_weights(self, path):
        with open(path, 'rb') as f:
            self.parameters = pickle.load(f)


class Cnn(BaseModel):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.parameters = parameters

    def eval(self):
        for layer in self.parameters:
            if layer.trainable:
                layer.train = False

    def train(self):
        for layer in self.parameters:
            if layer.trainable:
                layer.train = True

    def __call__(self, input):
        for layer in self.parameters:
            input = layer(input)
        return input

    def get_parameters(self):
        param_list = []
        for layer in self.parameters:
            if layer.trainable:
                param_list.append([layer.__class__.__name__, layer.train, layer.get_nrof_trainable_params()])
            else:
                param_list.append([layer.__class__.__name__, layer.trainable, 0])
        return param_list

    def dump_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.parameters, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_weights(self, path):
        with open(path, 'rb') as f:
            self.parameters = pickle.load(f)
