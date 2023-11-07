import matplotlib.pyplot as plt

class LossPloter():
    def __init__(self, file_name):
        self.file_name = file_name

    def plot(self, epoch, loss):
        plt.plot(list(range(epoch)), loss)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig(self.file_name)
        plt.close()