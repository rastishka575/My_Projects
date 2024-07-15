import torch

class MLP2(torch.nn.Module):
  def __init__(self):
    super(MLP2, self).__init__()
    self.fc1 = torch.nn.Linear(28*28, 64)
    self.act1 = torch.nn.ReLU()
    self.fc2 = torch.nn.Linear(64, 64)
    self.act2 = torch.nn.ReLU()
    self.fc3 = torch.nn.Linear(64, 10)

  def forward(self, x):
    x = x.view(x.size(0), x.size(1) * x.size(2)*x.size(3))
    x = self.fc1(x)
    x = self.act1(x)
    x = self.fc2(x)
    x = self.act2(x)
    x = self.fc3(x)
    return x