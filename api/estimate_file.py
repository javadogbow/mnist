import torch.cuda
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from torchvision.datasets import MNIST
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from core.NeuralNetwork import NeuralNetwork

device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = transforms.Compose([transforms.ToTensor()])

model = NeuralNetwork().to(device)
model.load_params()
model.eval()

image = Image.open('./api/mnist_1.png')
image = image.convert('L').resize((28, 28))
image = transform(image).unsqueeze(0)

output = model(image.to(device))
pred = output.data.max(1, keepdim=True)[1]

print('predicate: {}'.format(pred.item()))
