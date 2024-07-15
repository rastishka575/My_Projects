from datasets import Dataset_MNIST, Dataset_CIFAR10
from augmentations import REGISTRY_TYPE
from dataloaders import DataLoader
from configs import parser


FLAGS = parser.parse_args()
augmentations = [(FLAGS.CenterCrop, {'crop_size': (25, 25)}), (FLAGS.RandomCrop, {'crop_size': 25})]
transforms = []
for aug in augmentations:
    transforms.append(REGISTRY_TYPE.get(*aug))

mnist_train = Dataset_MNIST(data_path=FLAGS.path_mnist, dataset_type='train', transforms=transforms,
                            nrof_classes=FLAGS.nrof_classes)
mnist_test = Dataset_MNIST(data_path=FLAGS.path_mnist, dataset_type='test', transforms=transforms,
                           nrof_classes=FLAGS.nrof_classes)

mnist_train.read_data()
mnist_test.read_data()

mnist_train.show_statistics()
mnist_test.show_statistics()

dataLoader_mnist_train = DataLoader(dataset=mnist_train, nrof_classes=FLAGS.nrof_classes, dataset_type='train',
                                    shuffle=True, batch_size=FLAGS.batch_size, sample_type=FLAGS.sample_type)

dataLoader_mnist_test = DataLoader(dataset=mnist_test, nrof_classes=FLAGS.nrof_classes, dataset_type='test',
                                   shuffle=False, batch_size=FLAGS.batch_size, sample_type=FLAGS.sample_type)


cifar_train = Dataset_CIFAR10(data_path=FLAGS.path_cifar, dataset_type='train', transforms=transforms,
                              nrof_classes=FLAGS.nrof_classes)
cifar_test = Dataset_CIFAR10(data_path=FLAGS.path_cifar, dataset_type='test',
                             nrof_classes=FLAGS.nrof_classes)

cifar_train.read_data()
cifar_test.read_data()

cifar_train.show_statistics()
cifar_test.show_statistics()

dataloader_cifar_train = DataLoader(dataset=cifar_train, nrof_classes=FLAGS.nrof_classes, dataset_type='train',
                                    shuffle=True, batch_size=FLAGS.batch_size, sample_type=FLAGS.sample_type)

dataloader_cifar_test = DataLoader(dataset=cifar_test, nrof_classes=FLAGS.nrof_classes, dataset_type='test',
                                   shuffle=False, batch_size=FLAGS.batch_size, sample_type=FLAGS.sample_type)


n = 0
for (data, label) in dataLoader_mnist_train.batch_generator():
    if n == 2:
        break
    n += 1
    dataLoader_mnist_train.show_batch()

n = 0
for (data, label) in dataloader_cifar_train.batch_generator():
    if n == 2:
        break
    n += 1
    dataloader_cifar_train.show_batch()


augmentations_all_mnist = [
    (FLAGS.Pad, {'image_size': 10, 'fill': 0, 'mode': 'constant'}),
    (FLAGS.Translate, {'shift': 10, 'direction': 'right', 'roll': True}),
    (FLAGS.Scale, {'image_size': 25, 'scale': 10}),
    (FLAGS.RandomCrop, {'crop_size': (10, 12)}),
    (FLAGS.CenterCrop, {'crop_size': (10, 12)}),
    (FLAGS.RandomRotateImage, {'min_angle': 40, 'max_angle': 60}),
    (FLAGS.GaussianNoise, {'mean': 0, 'sigma': 0.03, 'by_channel': False}),
    (FLAGS.Salt, {'prob': 0.05, 'by_channel': False}),
    (FLAGS.Pepper, {'prob': 0.05, 'by_channel': False}),
    (FLAGS.GaussianBlur, {'ksize': (5, 5)}),
    (FLAGS.Normalize, {'mean': 128, 'var': 255})]

augmentations_all_cifar = [
    (FLAGS.Pad, {'image_size': 10, 'fill': 0, 'mode': 'constant'}),
    (FLAGS.Translate, {'shift': 10, 'direction': 'right', 'roll': True}),
    (FLAGS.Scale, {'image_size': 25, 'scale': 10}),
    (FLAGS.RandomCrop, {'crop_size': (10, 12)}),
    (FLAGS.CenterCrop, {'crop_size': (10, 12)}),
    (FLAGS.RandomRotateImage, {'min_angle': 40, 'max_angle': 60}),
    (FLAGS.GaussianNoise, {'mean': 0, 'sigma': 0.03, 'by_channel': False}),
    (FLAGS.Salt, {'prob': 0.05, 'by_channel': False}),
    (FLAGS.Pepper, {'prob': 0.05, 'by_channel': False}),
    (FLAGS.GaussianBlur, {'ksize': (5, 5)}),
    (FLAGS.Normalize, {'mean': 20, 'var': 255}),
    (FLAGS.ChangeBrightness, {'value': 30, 'type': 'brightness'})]

transforms_all_mnist = []
for aug in augmentations_all_mnist:
    transforms_all_mnist.append(REGISTRY_TYPE.get(*aug))

transforms_all_cifar = []
for aug in augmentations_all_cifar:
    transforms_all_cifar.append(REGISTRY_TYPE.get(*aug))

mnist_train.show_transforms(transforms=transforms_all_mnist, idx=0)
cifar_train.show_transforms(transforms=transforms_all_cifar, idx=0)

