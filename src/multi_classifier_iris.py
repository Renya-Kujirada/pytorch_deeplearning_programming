import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchviz import make_dot
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class Net(nn.Module):
    def __init__(self, dim_input, dim_output):
        super().__init__()
        self.l1 = nn.Linear(dim_input, dim_output)
        self.l1.weight.data.fill_(1.0)
        self.l1.bias.data.fill_(1.0)

    def forward(self, x):
        x1 = self.l1(x)
        return x1


def load_dataset():
    iris = load_iris()
    x_input, y_label = iris.data, iris.target
    return x_input, y_label


def split_dataset(x_input, y_label, seed):
    x_train, x_test, y_train, y_test = train_test_split(
        x_input, y_label, train_size=0.5, test_size=0.5, random_state=seed
    )
    return x_train, x_test, y_train, y_test


def preprocess(seed):
    x_input, y_label = load_dataset()
    x_train, x_test, y_train, y_test = split_dataset(x_input, y_label, seed)
    return x_train, x_test, y_train, y_test


def train(net, inputs, labels, criterion, optimizer, epoch):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(loss.item())


def main():
    seed = 123
    lr = 0.01
    num_epochs = 10000
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    x_train, x_test, y_train, y_test = preprocess(seed)
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    dim_input = x_train.shape[1]
    dim_output = len(list(set(y_train)))
    # print(f'n_input: {n_input}  n_output: {n_output}')

    net = Net(dim_input, dim_output).to(device)
    summary(net, (dim_input,))

    criterion = nn.CrossEntropyLoss()  # 損失関数：交差エントロピー
    optimizer = optim.SGD(net.parameters(), lr=lr)  # 最適化関数：勾配降下法
    history = np.zeros((0, 5))

    inputs = torch.tensor(x_train).float().to(device)
    labels = torch.tensor(y_train).long().to(device)
    for epoch in range(num_epochs):
        train(net, inputs, labels, criterion, optimizer, epoch)


if __name__ == '__main__':
    main()
