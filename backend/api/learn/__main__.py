from backend.core.NeuralNetwork import NeuralNetwork
from .plot_loss import LossPlotter
from torch.autograd import Variable
import torch
from backend.core.MNISTDataSetManager import MNISTDatasetManager

#################################################### CONFIG ####################################################
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = NeuralNetwork().to(DEVICE)
BATCH_SIZE = 32
EPOCH = 30
data_loader = MNISTDatasetManager(BATCH_SIZE).train_data()
################################################################################################################

loss_array = []
loss_plotter = LossPlotter('loss_graph')

print('---------- lerning ----------')
for epoch in range(EPOCH):
    total_loss = 0.0
    num_train = 0
    for i, data in enumerate(data_loader):
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
    loss_plotter.plot(epoch + 1, loss_array)
    model.save_params()

    print('epoch: %d, loss: %f' % (epoch + 1, total_loss/num_train))
