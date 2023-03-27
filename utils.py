import warnings

import numpy as np
import torch
import torch.nn as nn
from skimage.util import random_noise

warnings.filterwarnings('ignore')


def sigmoid(x):
    s = 1 / (1 + torch.exp(-x))
    return s


def weight_init(net):
    if isinstance(net, nn.Conv2d):
        nn.init.kaiming_uniform_(net.weight, mode='fan_in', nonlinearity='leaky_relu')


def gauss_noise(data, mu, sd):
    noise_data = np.zeros_like(data)
    for i in range(len(data)):
        noise = np.random.normal(loc=mu, scale=sd, size=data[i].shape)
        noise_data[i] = data[i] + noise
    return noise_data

def addGaussNoise(data, sigma):
    sigma2 = sigma**2 / (255 ** 2)
    noise = random_noise(data, mode='gaussian', var=sigma2, clip=True)
    return noise