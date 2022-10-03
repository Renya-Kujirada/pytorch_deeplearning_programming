import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=(1, 1))  # 入力チャネル数，出力チャネル数，カーネルサイズ
        self.conv2 = nn.Conv2d(32, 32, 3, padding=(1, 1))  # padding: 入力テンソルの外側を(1,1)の大きさのダミーデータで埋める．
