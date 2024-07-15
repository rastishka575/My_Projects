import torch

class Logreg(torch.nn.Module):
  def __init__(self):
    super(Logreg, self).__init__()
    self.fc1 = torch.nn.Linear(28*28, 10)

  def forward(self, x):
    x = x.view(x.size(0), x.size(1) * x.size(2)*x.size(3))
    x = self.fc1(x)
    return x
