from NewralNetwork import NewralNetwork
from torchvision.datasets import MNIST
from torch.autograd import Variable
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optimizer

model = NewralNetwork()

# ミニバッチのバッチサイズ
BATCH_SIZE = 32
# 最大学習回数
MAX_EPOCH = 100

data_folder = '~/data'
transform = transforms.Compose([
    # データの型をTensorに変換する
    transforms.ToTensor()
])

# 学習データ
train_data_with_labels = MNIST(data_folder, train=True, download=True, transform=transform)

train_data_loader = DataLoader(train_data_with_labels, batch_size=BATCH_SIZE, shuffle=True)

# 検証データ
test_data_with_labels = MNIST(data_folder, train=False, download=True, transform=transform)
test_data_loader = DataLoader(test_data_with_labels, batch_size=BATCH_SIZE, shuffle=True)


lossResult = nn.CrossEntropyLoss()
param_loaded_model = model.load_state_dict(torch.load('model_weight.pth', map_location="cpu"))
optimizer = optimizer.SGD(param_loaded_model, lr=0.01)

print('---------- 学習開始 ----------')
for epoch in range(MAX_EPOCH):
    total_loss = 0.0
    num_train = 0
    for i, data in enumerate(train_data_loader):
        train_data, teacher_labels = data
        train_data, teacher_labels = Variable(train_data), Variable(teacher_labels)
        num_train += len(teacher_labels)
        optimizer.zero_grad()
        outputs = model(train_data)
        loss = lossResult(outputs, teacher_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print('学習進捗：[EPOCH:%d] 学習誤差（loss）: %.3f' % (epoch + 1, total_loss/num_train))

    torch.save(model.state_dict(), 'model_weight.pth')


print('---------- 正解率 ----------')
total = 0
count_when_correct = 0

for data in test_data_loader:
    test_data, teacher_labels = data
    results = model(Variable(test_data))
    _, predicted = torch.max(results.data, 1)
    total += teacher_labels.size(0)
    count_when_correct += (predicted == teacher_labels).sum()

print('count_when_correct:%d' % (count_when_correct))
print('total:%d' % (total))

print('正解率：%d / %d = %f' % (count_when_correct, total, int(count_when_correct) / int(total)))
