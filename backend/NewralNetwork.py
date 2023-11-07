import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer

class NewralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28 * 28, 600)
        self.layer2 = nn.Linear(600, 500)
        self.layer3 = nn.Linear(500, 400)
        self.layer4 = nn.Linear(400, 10)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optimizer.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def forward(self, input_data):
        input_data = input_data.view(-1, 28 * 28)
        layer1 = self.layer1(input_data)
        output_layer1 = self.act1(layer1)
        layer2 = self.layer2(output_layer1)
        output_layer2 = self.act2(layer2)
        layer3 = self.layer3(output_layer2)
        output_layer3 = self.act3(layer3)
        return self.layer4(output_layer3)
