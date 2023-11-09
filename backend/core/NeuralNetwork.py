import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optimizer

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optimizer.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def load_params(self):
        state_dict = torch.load('./backend/outputs/parameters.pth')
        self.load_state_dict(state_dict)

    def save_params(self):
        torch.save(self.state_dict(), './backend/outputs/parameters.pth')

