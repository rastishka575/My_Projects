from datasets import Dataset_MNIST
from augmentations import REGISTRY_TYPE
from dataloaders import DataLoader
from model import *
from layers import *
from loss_function import *
from optimizer import *
from metrics import *
from activation_function import *
from configs_train import cfg
from torch.utils.tensorboard import SummaryWriter


transforms = []
if cfg['transforms_append']:
    for aug in cfg['transforms'].keys():
        transforms.append(REGISTRY_TYPE.get(*(aug, cfg['transforms'][aug])))

mnist_train = Dataset_MNIST(data_path=cfg['data']['path'], dataset_type='train', transforms=transforms,
                            nrof_classes=cfg['data']['nrof_classes'])

mnist_train.read_data()

dataLoader_mnist_train = DataLoader(dataset=mnist_train, nrof_classes=cfg['dataloader']['nrof_classes'],
                                    dataset_type=cfg['dataloader']['type'], shuffle=cfg['dataloader']['shuffle'],
                                    batch_size=cfg['dataloader']['batch_size'], sample_type=cfg['dataloader']['sample_type'])

criterion = eval(cfg['criterion'])()
accuracy = eval(cfg['accuracy'])()

layers = []
for layer in cfg['model'].keys():
    if cfg['model'][layer]['class'] == 'function_activation':
        layers.append(eval(cfg['model'][layer]['type'])())
    else:
        layers.append(REGISTRY_MODEL.get(cfg['model'][layer]['type'], None)(**cfg['model'][layer]['init']))

model = Mlp(layers)

optimizer = eval(cfg['optimizer']['type'])(cfg['optimizer']['learning_rate'], model, cfg['optimizer']['momentum'])
writer = SummaryWriter('runs/' + cfg['experiment_name'])
# tensorboard --logdir=../runs

print(model.get_parameters())

accuracy_total = 0
loss_total = 0

iteration = 0
for epoch in range(cfg['epochs']):
    iteration_epoch = 0
    accuracy_mean = 0
    loss_mean = 0
    for (images, targets) in dataLoader_mnist_train.batch_generator():
        model = optimizer.zero_grad()
        prediction = model(images)
        loss = criterion(prediction, targets)
        dy = criterion.backward()
        optimizer.update_net(model)
        model = optimizer.minimize(dy)

        acc = accuracy(prediction, targets)
        accuracy_mean += acc
        loss_mean += loss
        # print("Training Results - Epoch: {}  Iteration: {}  Accuracy: {:.0f} Loss: {:.2f}".format(
        #   epoch, iteration_epoch, acc, loss))

        # writer.add_scalar('Loss/' + cfg['experiment_name'], loss, iteration)
        # writer.add_scalar('Accuracy/' + cfg['experiment_name'], acc, iteration)

        iteration += 1
        iteration_epoch += 1

        if iteration_epoch % 100 == 0:
            print("Training Results - Epoch: {}  Iteration: {}  Accuracy: {:.0f} Loss: {:.2f}".format(
                epoch, iteration_epoch, accuracy_mean/100, loss_mean/100))
            writer.add_scalar('Loss/' + cfg['experiment_name'], loss_mean/100, iteration)
            writer.add_scalar('Accuracy/' + cfg['experiment_name'], accuracy_mean/100, iteration)
            accuracy_total = accuracy_mean/100
            loss_total = loss_mean/100
            accuracy_mean = 0
            loss_mean = 0

writer.add_hparams({'weight_init': cfg['model']['fc1']['init']['initialization_type'],
                    'optimizer': cfg['optimizer']['type'], 'function_activation': cfg['model']['ac1']['type'],
                    'use_bias': cfg['model']['fc1']['init']['use_bias'],
                    'hidden_nrof': cfg['hidden_nrof'], 'hidden_size': cfg['hidden_size']},
                   {'accuracy': accuracy_total, 'loss': loss_total})

model.dump_model('models/' + cfg['experiment_name'] + '.pickle')
