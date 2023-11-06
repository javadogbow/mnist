import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

class NewralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28 * 28, 100)
        self.layer2 = nn.Linear(100, 50)
        self.layer3 = nn.Linear(50, 10)

    def forward(self, input_data):
        input_data = input_data.view(-1, 28 * 28)
        output_layer1 = self.layer1(input_data)
        output_layer2 = self.layer2(output_layer1)
        output_layer3 = self.layer3(output_layer2)
        return output_layer3
