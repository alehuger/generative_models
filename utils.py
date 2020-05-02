import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, sampler
from torchvision import datasets, transforms
from torchvision.utils import save_image


def denorm(x, channels=None, w=None, h=None, resize=False):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    if resize:
        if channels is None or w is None or h is None:
            print('Number of channels, width and height must be provided for resize.')
        x = x.view(x.size(0), channels, w, h)
    return x


def show(img):
    if torch.cuda.is_available():
        img = img.cpu()
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))


def save_img(fixed_input, path):
    save_image(fixed_input, path)


def load_MNIST(batch_size, data_dir):
    # Modify this line if you need to do any input transformations (optional).
    transform = transforms.Compose([transforms.ToTensor()])

    train_dat = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dat = datasets.MNIST(data_dir, train=False, transform=transform)

    loader_train = DataLoader(train_dat, batch_size, shuffle=True)
    loader_test = DataLoader(test_dat, batch_size, shuffle=False)

    return loader_train, loader_test


def load_CIFAR(batch_size, data_dir, num_train=35000):

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    cifar10_train = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    cifar10_val = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    cifar10_test = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

    loader_train = DataLoader(cifar10_train, batch_size=batch_size,
                              sampler=sampler.SubsetRandomSampler(range(num_train)))
    loader_val = DataLoader(cifar10_val, batch_size=batch_size,
                            sampler=sampler.SubsetRandomSampler(range(num_train, 50000)))
    loader_test = DataLoader(cifar10_test, batch_size=batch_size)

    return loader_train, loader_val, loader_test


def plot(training_loss, testing_loss, save_path):

    plt.plot(training_loss.sum(axis=1), marker='o', linestyle='--', label='training')
    plt.plot(testing_loss.sum(axis=1), marker='o', linestyle='--', label='testing')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title("Total Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

