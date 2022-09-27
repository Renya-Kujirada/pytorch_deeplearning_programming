import numpy as np
import torch
from tqdm import tqdm


class Trainer:
    def __init__(self):
        pass

    # training loop
    def fit(self, net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history):
        base_epochs = len(history)  # 追加学習用
        for epoch in range(base_epochs, num_epochs+base_epochs):
            train_loss, train_acc = self.train(net, optimizer, criterion, epoch, train_loader, device)
            val_loss, val_acc = self.test(net, optimizer, criterion, epoch, test_loader, device)
            item = np.array([epoch+1, train_loss, train_acc, val_loss, val_acc])
            history = np.vstack((history, item))
        return history

    # training in one epoch
    def train(self, net, optimizer, criterion, epoch, train_loader, device):
        net.train()
        train_loss = 0
        train_acc = 0
        count = 0

        # バッチ学習
        for inputs, labels in tqdm(train_loader):
            count += len(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 勾配の初期化
            optimizer.zero_grad()

            # 予測計算
            outputs = net(inputs)

            # 損失計算
            loss = criterion(outputs, labels)
            train_loss += loss.item()

            # 勾配計算
            loss.backward()

            # パラメータ修正
            optimizer.step()

            # 予測値算出*
            predicted = torch.max(outputs, 1)[1]

            # 正解件数算出
            train_acc += (predicted == labels).sum().item()

            # 損失と精度の計算
            avg_train_loss = train_loss / count
            avg_train_acc = train_acc / count

        print(f"Epoch: {epoch+1}, loss: {avg_train_loss:.5f}, acc: {avg_train_acc:.5f}")
        return avg_train_loss, avg_train_acc

    def test(self, net, optimizer, criterion, epoch, test_loader, device):
        net.eval()
        val_loss = 0
        val_acc = 0
        count = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                count += len(labels)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 予測計算
                outputs = net(inputs)

                # 損失計算
                loss = criterion(outputs, labels)
                val_loss += loss.item()  # sum up batch loss

                # 予測値算出
                predicted = torch.max(outputs, 1)[1]

                # 正解件数算出
                val_acc += (predicted == labels).sum().item()

                # 損失と精度の計算
                avg_val_loss = val_loss / count
                avg_val_acc = val_acc / count

        print(f"Epoch: {epoch+1}, val_loss: {avg_val_loss:.5f}, val_acc: {avg_val_acc:.5f}")
        return avg_val_loss, avg_val_acc


"""
*: predicted = torch.max(outputs, 1)[1]
softmaxは入力の段階で最大の項目が出力後も最大となるため, torch.maxで抽出．
torch.maxは最大値そのもの(value)と最大値を撮ったインデックス(indices)の2つを同時に返す．
2つ目の引数はどの軸で集計するかを表し，1だと行ごとに最大値を集計する．
ラベル値を取得したい場合，2つ目のindicesを指定するため[1]としている．
"""
