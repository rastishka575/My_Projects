class BaseOptim():
    def __init__(self, learning_rate, net, momentum):
        pass

    def update_rule(self, dW, v):
        pass

    def minimize(self, dy):
        pass

    def update_net(self, net):
        self.net = net


class Sgd(BaseOptim):
    def __init__(self, learning_rate, net, momentum=None):
        self.net = net
        self.learning_rate = learning_rate
        super().__init__(learning_rate, net, 0)

    def update_rule(self, dW, v):
        return (-1) * self.learning_rate * dW.mean(0)

    def minimize(self, dy):
        m = len(self.net.parameters)
        for i in range(1, m+1):
            self.net.parameters[m-i].get_grad()
            dy = self.net.parameters[m-i].backward(dy)
            if self.net.parameters[m-i].train:
                self.net.parameters[m-i].update_weights(self.update_rule)
        return self.net

    def zero_grad(self):
        m = len(self.net.parameters)
        for i in range(1, m + 1):
            self.net.parameters[m-i].zero_grad()
        return self.net


class Msgd(BaseOptim):
    def __init__(self, learning_rate, net, momentum):
        self.net = net
        self.learning_rate = learning_rate
        self.momentum = momentum
        super().__init__(learning_rate, net, momentum)

    def update_rule(self, dW, v):
        v_new = self.momentum*v
        v_new -= self.learning_rate*dW.mean(0)
        return v_new

    def minimize(self, dy):
        m = len(self.net.parameters)
        for i in range(1, m + 1):
            self.net.parameters[m - i].get_grad()
            dy = self.net.parameters[m - i].backward(dy)
            if self.net.parameters[m - i].train:
                self.net.parameters[m - i].update_weights(self.update_rule)
        return self.net

    def zero_grad(self):
        for layer in self.net.parameters:
            layer.zero_grad()
        return self.net
