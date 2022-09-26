import torch
from tqdm import tqdm


class Trainer:
    def __init__(self):
        pass

    # training loop
    def fit(self, net, optimizer, criterion, num_epoch, train_loader, test_loader, device, history):
        base_epochs = len(history)  # 追加学習用
        for epoch in range(base_epochs, num_epoch+base_epochs):
            self.train(net, optimizer, criterion, epoch, train_loader, device)
            self.test(net, optimizer, criterion, epoch, test_loader, device)

    # training in one epoch
    def train(self, net, optimizer, criterion, epoch, train_loader, device):
        net.train()

        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0
        count = 0

        # バッチ学習
        for inputs, label in tqdm(train_loader):
            count += len(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 勾配の初期化
            optimizer.zero_grad()

            # 予測計算
            outputs = net(inputs)

            # 損失計算
            loss = criterion(outpurs, labels)
            train_loss += loss.item()

            # 勾配計算
            loss.backward()

            # パラメータ修正
            optimiser.step()

            # 予測値算出*
            predicted = torch.max(outputs, 1)[1]

            # 正解件数算出
            train_acc += (predicted == labels).sum().item()

            # 損失と精度の計算
            avg_train_loss = train_loss / count
            avg_train_acc = train_acc / count

        print(f"Epoch: {epoch}, loss: {avg_train_loss}")

    def test(self, net, optimizer, criterion, epoch, test_loader, device):
        pass


"""
*: predicted = torch.max(outputs, 1)[1]
softmaxは入力の段階で最大の項目が出力後も最大となるため, torch.maxで抽出．
torch.maxは最大値そのもの(value)と最大値を撮ったインデックス(indices)の2つを同時に返す．
2つ目の引数はどの軸で集計するかを表し，1だと行ごとに最大値を集計する．
ラベル値を取得したい場合，2つ目のindicesを指定するため[1]としている．
"""
