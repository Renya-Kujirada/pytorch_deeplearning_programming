import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from preprocess import Preprocess
from model import CNN
from trainer import Trainer


def main():
    seed = 123
    batch_size = 100
    mu = 0.5
    sigma = 0.5
    dataset_path = "./data"
    result_path = "./src/model_tuning/result.csv"
    batch_size = 100
    n_hidden = 128  # no of hidden layer's nodes
    n_output = 10
    lr = 0.01
    num_epochs = 50

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"use device {device}")

    # prepare datasets
    preprocess = Preprocess(mu, sigma, dataset_path)
    train_set, test_set = preprocess.prepare_dataset()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # initialize random seed
    preprocess.torch_seed(seed)

    # prepare model
    net = CNN(n_output).to(device)

    # loss function
    criterion = nn.CrossEntoropyLoss()

    # optimize function
    optimizer = optim.SGD(net.parameters(), lr=lr)

    # for record metrics
    history = np.zeros((0, 5))

    # training and validation
    trainer = Trainer()
    history = trainer.fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history)
    np.savetxt(result_path. history, delimiter=",")


if __name__ == "__main__":
    main()
