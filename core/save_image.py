import os
import sys
import numpy as np
import torch
import torchvision.datasets
import torchvision.models
import torchvision.transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

def save_image(data, file_name):
    image = make_grid(data.view(-1, 1, 28, 28).data)
    image = np.transpose(image.numpy(),(1,2,0))

    plt.imshow(image)
    plt.gray()
    plt.imsave('./api/'+file_name, image)
    plt.close()