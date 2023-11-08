import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optimizer

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28 * 28, 100)
        self.layer2 = nn.Linear(100, 50)
        self.layer3 = nn.Linear(50, 10)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optimizer.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def load_params(self):
        state_dict = torch.load('./backend/outputs/parameters.pth')
        self.load_state_dict(state_dict)

    def forward(self, input_data):
        input_data = input_data.view(-1, 28 * 28)
        layer1 = self.layer1(input_data)
        output_layer1 = self.act1(layer1)
        layer2 = self.layer2(output_layer1)
        output_layer2 = self.act2(layer2)
        return self.layer3(output_layer2)

    def save_params(self):
        torch.save(self.state_dict(), './backend/outputs/parameters.pth')

