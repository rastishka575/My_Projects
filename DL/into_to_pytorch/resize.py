from PIL import Image
import torchvision.transforms as transforms

class Resize():
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return img.resize(self.size, resample=Image.BILINEAR)


tf = transforms.Compose(
    [transforms.ToPILImage(),
    Resize((32, 32)),
    transforms.RandomCrop((28, 28)), transforms.ToTensor()])

