import numpy as np
import torch
from tqdm import tqdm


class Trainer:
    def __init__(self):
        pass

    # training loop
    def fit(self, net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history):
        base_epochs = len(history)
        for epoch in range(base_epochs, num_epochs+base_epochs):
            train_loss, train_acc = self.train(net, optimizer, criterion, epoch, train_loader, device)
            val_loss, val_acc = self.test(net, criterion, epoch, test_loader, device)
            item = np.array([epoch+1, train_loss, train_acc, val_loss, val_acc])
            history = np.vstack((history, item))
        return history

    # training in one epoch
    def train(self, net, optimizer, criterion, epoch, train_loader, device):
        net.train()
        train_loss = 0
        train_acc = 0
        count = 0

        # batch training
        for inputs, labels in tqdm(train_loader):
            count += len(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()   # 勾配の初期化
            outputs = net(inputs)   # 予測計算
            loss = criterion(outputs, labels)  # 損失計算
            train_loss += loss.item()
            loss.backward()  # 勾配計算
            optimizer.step()  # パラメータ修正

            predicted = torch.max(outputs, 1)[1]  # 予測値算出
            train_acc += (predicted == labels).sum().item()  # 正解件数算出
            avg_train_loss = train_loss / count  # 損失の計算
            avg_train_acc = train_acc / count  # 精度の計算

        print(f"Epoch: {epoch+1}, train_loss: {avg_train_loss:.5f}, train_acc: {avg_train_acc:.5f}")
        return avg_train_loss, avg_train_acc

    # test in one epoch
    def test(self, net, criterion, epoch, test_loader, device):
        net.eval()
        val_loss = 0
        val_acc = 0
        count = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                count += len(labels)
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predicted = torch.max(outputs, 1)[1]
                val_acc += (predicted == labels).sum().item()
                avg_val_loss = val_loss / count
                avg_val_acc = val_acc / count

        print(f"Epoch: {epoch+1}, val_loss: {avg_val_loss:.5f}, val_acc: {avg_val_acc:.5f}")
        return avg_val_loss, avg_val_acc
