import torch
import torchvision.transforms as tf
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets


class Logreg(torch.nn.Module):
  def __init__(self, count_class):
    super(Logreg, self).__init__()
    self.fc1 = torch.nn.Linear(28*28, count_class)

  def forward(self, x):
    x = x.view(x.size(0), x.size(1) * x.size(2)*x.size(3))
    x = self.fc1(x)
    return x


class Loss_total():
    def __init__(self, weight_decay, loss_function):
        self.loss_value = 0
        self.loss_now = 0
        self.iter = 0
        self.loss_total_value = 0
        self.weight_decay = torch.tensor(weight_decay)
        self.loss_function = loss_function

    def reset(self):
        self.loss_value = 0
        self.iter = 0
        self.loss_total_value = 0
        self.loss_now = 0

    def loss(self, pred, label):
        self.iter += 1
        self.loss_now = self.loss_function(pred, label)
        self.loss_value +=  self.loss_now.item()
        return  self.loss_now
    
    def loss_total(self, net):
        l2_reg = None
        for param in net.parameters():
          if l2_reg == None:
            l2_reg = param.norm()**2
          else:
            l2_reg += param.norm()**2
        l2_reg *= weight_decay 
        self.loss_now += l2_reg
        self.loss_total_value +=  self.loss_now.item()
        return self.loss_now

    def value_zero(self):
        self.loss_now = 0
        return 0

    def compute_loss(self):
        return self.loss_value/self.iter

    def compute_loss_total(self):
        return self.loss_total_value/self.iter


class Balanced_accuracy():
    def __init__(self, count_class):
        self.count_class = count_class
        self._num_correct = torch.zeros(self.count_class)
        self._num_examples = torch.zeros(self.count_class)

    def reset(self):
        self._num_correct = torch.zeros(self.count_class)
        self._num_examples = torch.zeros(self.count_class)

    def update(self, pred, label):
        pred = pred.argmax(dim=1)

        pred = torch.bincount(pred[pred==label])

        label = torch.bincount(label)
              
        self._num_correct[:len(pred)] += pred.cpu()
        self._num_examples[:len(label)] += label.cpu()

    def compute(self):
        if self._num_examples.sum() == 0:
          return 0
        acc_bal = self._num_correct / self._num_examples
        acc_bal = acc_bal.sum()
        acc_bal /= self.count_class
        return acc_bal


class Accuracy():
    def __init__(self):
        self._num_correct = 0
        self._num_examples = 0

    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    def update(self, pred, label):
        pred = pred.argmax(dim=1)

        pred = pred[pred==label]

        label = label
              
        self._num_correct += len(pred)
        self._num_examples += len(label)

    def compute(self):
        if self._num_examples == 0:
          return 0
        return self._num_correct / self._num_examples


class Trainer():

  def __init__(self, model, train_loader, valid_loader, weight_decay, loss_function, learning_rate, device, writer, count_class):
    self.model = model
    self.train_loader = train_loader
    self.valid_loader = valid_loader 
    self.loss_total = Loss_total(weight_decay, loss_function)
    self.optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate) 
    self.device = device
    self.writer = writer
    self.balanced_accuracy = Balanced_accuracy(count_class)
    self.accuracy = Accuracy()
    self.epoch = 0

  def reset(self):
    self.accuracy.reset()
    self.balanced_accuracy.reset()
    self.loss_total.reset()

  def train(self):

    self.reset()

    self.model.train()
    self.epoch += 1

    for (x, label) in self.train_loader:

      self.optimizer.zero_grad()

      x = x.to(self.device)
      label = label.to(self.device)

      pred = self.model.forward(x)

      loss = self.loss_total.loss(pred, label)
      loss = self.loss_total.loss_total(self.model)

      self.accuracy.update(pred, label)
      self.balanced_accuracy.update(pred, label)

      loss.backward()
      self.optimizer.step()
    
    self.model.eval()
    
    self.log_writer('Train')
    self.print_result('Train')

  def test(self):

    self.reset()

    self.model.eval()

    for (x, label) in self.valid_loader:

      x = x.to(self.device)
      label = label.to(self.device)

      pred = self.model.forward(x)

      loss = self.loss_total.loss(pred, label)

      self.accuracy.update(pred, label)
      self.balanced_accuracy.update(pred, label)
    
    self.log_writer('Test')
    self.print_result('Test')

  def log_writer(self, data):

    self.writer.add_scalar('Loss/' + data, self.loss_total.compute_loss(), self.epoch)
    self.writer.add_scalar('Accuracy/' + data, self.accuracy.compute(), self.epoch)
    self.writer.add_scalar('Balanced_Accuracy/' + data, self.balanced_accuracy.compute(), self.epoch)

    if data == 'Train':
      self.writer.add_scalar('Loss_Total/Train', self.loss_total.compute_loss_total(), self.epoch)

  
  def print_result(self, data):
    if data == 'Train':
      print('Training Results - Epoch: {}  accuracy: {:.2f} loss: {:.2f} loss_total: {:.2f} balanced_accuracy: {:.2f}'.format(
            self.epoch, self.accuracy.compute(), self.loss_total.compute_loss(), self.loss_total.compute_loss_total(), self.balanced_accuracy.compute()))
    else:
      print('Validation Results - Epoch: {}  accuracy: {:.2f} loss: {:.2f} balanced_accuracy: {:.2f}'.format(
            self.epoch, self.accuracy.compute(), self.loss_total.compute_loss(), self.balanced_accuracy.compute()))

  def train_end(self):
    self.reset()
    self.writer.close()

  def get_model(self):
    return self.model


if __name__ == "__main__":
	date_train = torchvision.datasets.KMNIST('./', download=True, train=True, transform=tf.ToTensor())
	date_test = torchvision.datasets.KMNIST('./', download=True, train=False, transform=tf.ToTensor())

	train_loader = torch.utils.data.DataLoader(dataset=date_train, shuffle=True, batch_size=32)
	test_loader = torch.utils.data.DataLoader(dataset=date_test, batch_size=32)

	lr = 0.001
	weight_decay = 0.001
	max_epoch = 5
	writer = SummaryWriter('logs')
	count_class = 10

	net = Logreg(count_class)

	loss_function = torch.nn.CrossEntropyLoss()
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	net = net.to(device)

	trainer = Trainer(net, train_loader, test_loader, weight_decay, loss_function, lr, device, writer, count_class)

	for i in range(max_epoch):
	  trainer.train()
	  trainer.test()
	trainer.train_end()

	net = trainer.get_model()
	net = net.cpu()

	torch.save(net, 'model.pt')