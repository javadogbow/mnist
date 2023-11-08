import os
import sys
import numpy as np
import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.models
import torchvision.transforms as transforms
from core.save_image import save_image

########## CONFIG ##########
DATA_FOLDER = './data'
BATCH_SIZE = 1

test_dataset = MNIST(root=DATA_FOLDER, train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

Iter = iter(test_loader)
data, label = next(Iter)

save_image(data, 'mnist_1.png')