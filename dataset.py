from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import mnist
from torchvision.transforms import transforms

from utils import *

warnings.filterwarnings('ignore')


train_set = mnist.MNIST('./data', train=True, download=True)
test_set = mnist.MNIST('./data', train=False, download=True)

train_imgs = train_set.data.numpy()
train_labels = train_set.targets.numpy()

test_imgs = test_set.data.numpy()
test_labels = test_set.targets.numpy()


class PairedDataset(Dataset):
    def __init__(self, imgs, labels):
        self.x = imgs
        self.y = labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        img0 = self.x[index] / 255
        y0 = self.y[index]
        if np.random.rand() < 0.5:
            idx = np.random.choice(np.arange(len(self.y))[self.y == y0], 1)
        else:
            idx = np.random.choice(np.arange(len(self.y))[self.y != y0], 1)

        img1 = self.x[idx[0]] / 255
        y1 = self.y[idx[0]]
        if y0 == y1:
            label = 1
        else:
            label = 0

        return img0, img1, label


def get_mnist(batch_size_train, batch_size_test, noise=False, sigma=30):
    if noise:
        train_noise = addGaussNoise(train_imgs/255, sigma)
        test_noise = addGaussNoise(test_imgs/255, sigma)
        train_paired = PairedDataset(train_noise*255, train_labels)
        test_paired = PairedDataset(test_noise*255, test_labels)
    else:
        train_paired = PairedDataset(train_imgs, train_labels)
        test_paired = PairedDataset(test_imgs, test_labels)

    train_iter = DataLoader(train_paired, batch_size=batch_size_train, drop_last=True)
    test_iter = DataLoader(test_paired, batch_size=batch_size_test, drop_last=True)

    return train_iter, test_iter

