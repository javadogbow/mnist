from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class MNISTDatasetManager:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.data_folder = './backend/data'
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def train_data(self):
        data = MNIST(root=self.data_folder, train=True, download=True, transform=self.transform)
        return DataLoader(data, batch_size=self.batch_size, shuffle=True)

    def test_data(self):
        data = MNIST(root=self.data_folder, train=False, download=True, transform=self.transform)
        return DataLoader(data, batch_size=self.batch_size, shuffle=True)
