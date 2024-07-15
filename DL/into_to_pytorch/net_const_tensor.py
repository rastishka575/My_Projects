import torch
import log_reg as _log_reg
from torch.autograd import Function


class LinearFunction(Function):

    @staticmethod
    def forward(ctx, input, constant):
        ctx.constant = constant
        return input*constant

    @staticmethod
    def backward(ctx, grad_output):
        constant = ctx.constant
        return grad_output*constant, None


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lognet = _log_reg.Logreg()
        self.linear = LinearFunction.apply
        self.constant = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, xb):
        return self.linear(self.lognet.forward(xb), self.constant)