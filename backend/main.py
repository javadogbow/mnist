from core.NewralNetwork import NewralNetwork
from .plot_loss import LossPloter
from torchvision.datasets import MNIST
from torch.autograd import Variable
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
EPOCH = 3
data_folder = './data'
transform = transforms.Compose([transforms.ToTensor()])

# 学習データ
train_data_with_labels = MNIST(data_folder, train=True, download=True, transform=transform)
train_data_loader = DataLoader(train_data_with_labels, batch_size=BATCH_SIZE, shuffle=True)

# 検証データ
test_data_with_labels = MNIST(data_folder, train=False, download=True, transform=transform)
test_data_loader = DataLoader(test_data_with_labels, batch_size=BATCH_SIZE, shuffle=True)

model = NewralNetwork().to(DEVICE)
loss_array = []
loss_ploter = LossPloter('loss_graph')

print('---------- lerning ----------')
for epoch in range(EPOCH):
    total_loss = 0.0
    num_train = 0
    for i, data in enumerate(train_data_loader):
        train_data, teacher_labels = data
        train_data, teacher_labels = Variable(train_data).to(DEVICE), Variable(teacher_labels).to(DEVICE)
        num_train += len(teacher_labels)

        # 勾配を0に初期化
        model.optimizer.zero_grad()

        # forwardした結果を取得
        outputs = model(train_data)

        # backwardしてパラメータを調整
        loss = model.loss(outputs, teacher_labels)
        loss.backward()
        model.optimizer.step()
        
        total_loss += loss.item()

    loss_array.append(total_loss / num_train)
    loss_ploter.plot(epoch + 1, loss_array)
    model.save_params()

    print('epoch: %d, loss: %f' % (epoch + 1, total_loss/num_train))



print('---------- test accuracy ----------')
total = 0
count_when_correct = 0

for data in test_data_loader:
    test_data, teacher_labels = data
    input_data = Variable(test_data).to(DEVICE)
    results = model(input_data)
    _, predicted = torch.max(results.data, 1)
    total += teacher_labels.size(0)
    count_when_correct += (predicted == teacher_labels.to(DEVICE)).sum()

print('accuracy: %f' % (float(count_when_correct) / float(total)))
