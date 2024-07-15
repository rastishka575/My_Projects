import numpy as np

class BaseLossesClass():
    def __call__(self, logits, labels, phase='eval'):
        pass

    def get_grad(self):
        pass

    def backward(self, dl=1):
        pass

    def zero_grad(self):
        self.grad = 0


class Cross_entropy(BaseLossesClass):
    def __call__(self, logits, labels, phase='eval'):
        self.grad = logits - labels
        loss = (-1) * labels * np.log(logits + 1e-7)
        loss_mean = max(0, loss.mean())
        return loss_mean

    def get_grad(self):
        pass

    def backward(self, dl=1):
        return self.grad