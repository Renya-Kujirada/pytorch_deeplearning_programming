import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from preprocess import Preprocess
from model import CNN, CNN_with_Dropout, CNN_with_Dropout_BatchNorm
from trainer import Trainer


def main():
    seed = 123
    batch_size = 100
    mu = 0.5
    sigma = 0.5
    dataset_path = "./data"
    # result_dir = "./src/model_tuning/result_optimizer_SGD"
    # result_dir = "./src/model_tuning/result_Adam"
    # result_dir = "./src/model_tuning/result_Adam_with_Dropout"
    # result_dir = "./src/model_tuning/result_Adam_with_Dropout_BatchNorm"
    result_dir = "result_Adam_with_Dropout_BatchNorm_DataAugmentation"
    metrics_path = f"{result_dir}/result.csv"
    model_path = f"{result_dir}/model.pth"
    batch_size = 100
    n_output = 10
    lr = 0.01
    num_epochs = 100

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
    # net = CNN(n_output).to(device)
    # net = CNN_with_Dropout(n_output).to(device)
    net = CNN_with_Dropout_BatchNorm(n_output).to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimize function
    # optimizer = optim.SGD(net.parameters(), lr=lr)
    # optimizer = optim.SGD(net.parameters, lr=lr, momentum=0.9)
    optimizer = optim.Adam(net.parameters())

    # for record metrics
    history = np.zeros((0, 5))

    # training and validation
    trainer = Trainer()
    history = trainer.fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history)

    # record metrics and model
    np.savetxt(metrics_path, history, delimiter=",")
    torch.save(net.state_dict(), model_path)


if __name__ == "__main__":
    main()
