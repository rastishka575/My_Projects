import unittest
import numpy as np
from torch.nn import functional
from torch import Tensor
import torch
from layers import *
from activation_function import *
from model import *
from loss_function import *
from torch.autograd import grad


class TestMLP(unittest.TestCase):
    def test_fc_size(self):
        test_input = np.random.normal(size=(1, 28*28))
        fc = Linear(28*28, 64, use_bias=False)
        result = fc(test_input)
        torch_result = functional.linear(Tensor(test_input), Tensor(fc.weight))
        self.__helper(result, torch_result)

    def __helper(self, result, torch_result):
        self.assertEqual(np.array_equal(result.shape, torch_result.numpy().shape), True,
                         (result.shape, torch_result.numpy().shape))
        self.assertEqual(np.allclose(result, torch_result.numpy(), atol=1e-6), True)

    def test_activation_function(self):

        test_input = np.random.normal(size=(1, 28*28))

        ac = Relu()
        result = ac(test_input)
        torch_result = functional.relu(Tensor(test_input))
        self.__helper(result, torch_result)

        ac = Sigmoid()
        result = ac(test_input)
        torch_result = functional.sigmoid(Tensor(test_input))
        self.__helper(result, torch_result)

        ac = Tanh()
        result = ac(test_input)
        torch_result = functional.tanh(Tensor(test_input))
        self.__helper(result, torch_result)

    def __minimize(self, net, dy):
        m = len(net.parameters)
        for i in range(1, m+1):
            net.parameters[m-i].get_grad()
            dy = net.parameters[m-i].backward(dy)
        return dy

    def test_grad_check(self):
        test_input = np.random.normal(size=(1, 28 * 28))
        test_targets = np.zeros((1, 10))
        test_targets[0, 3] = 1

        fc1 = Linear(28 * 28, 64, use_bias=False)
        ac1 = Relu()
        fc2 = Linear(64, 10, use_bias=False)
        sm = Softmax()

        layers = [fc1, ac1, fc2, sm]
        model = Mlp(layers)
        criterion = Cross_entropy()

        prediction = model(np.reshape(test_input, (1, 1, 28, 28)))
        criterion(prediction, test_targets)
        dy = self.__minimize(model, criterion.backward())

        torch_test_input = Tensor(test_input)
        torch_test_input.requires_grad = True
        torch_result = functional.linear(torch_test_input, Tensor(fc1.weight))
        torch_result = functional.relu(torch_result)
        torch_result = functional.linear(torch_result, Tensor(fc2.weight))
        loss = functional.cross_entropy(torch_result, torch.LongTensor(np.array([3], dtype=np.float32)))
        gr = grad(outputs=loss, inputs=torch_test_input)[0]

        self.__helper(dy, gr)


if __name__ == "__main__":
    unittest.main()
