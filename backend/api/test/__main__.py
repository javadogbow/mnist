from backend.core.NeuralNetwork import NeuralNetwork
from torch.autograd import Variable
import torch
from backend.core.MNISTDataSetManager import MNISTDatasetManager

#################################################### CONFIG ####################################################
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = NeuralNetwork().to(DEVICE)
BATCH_SIZE = 32
EPOCH = 30
data_loader = MNISTDatasetManager(BATCH_SIZE).test_data()
################################################################################################################

print('---------- test accuracy ----------')
total = 0
count_when_correct = 0

for data in data_loader:
    test_data, teacher_labels = data
    input_data = Variable(test_data).to(DEVICE)
    results = model(input_data)
    _, predicted = torch.max(results.data, 1)
    total += teacher_labels.size(0)
    count_when_correct += (predicted == teacher_labels.to(DEVICE)).sum()

print('accuracy: %f' % (float(count_when_correct) / float(total)))
