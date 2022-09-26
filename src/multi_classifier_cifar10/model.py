import torch.nn as nn


class CNN(nn.module):
    def __init__(self, n_output, n_hidden):
        super().__init__()
        # input channel:3, output channel:32, kernel size:3
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d((2, 2))
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(6272, n_hidden)  # 32 * 14 * 14
        self.l2 = nn.Linear(n_hidden, n_output)

        self.features = nn.Sequential(
            self.conv1,  # output: 32ch * 30 * 30
            self.relu,
            self.conv2,  # output: 32ch * 28 * 28
            self.relu,
            self.maxpool  # output: 32ch * 14 * 14
        )

        self.classifier = nn.Sequential(
            self.l1,
            self.relu,
            self.l2
        )

    def forward(self, x):
        x1 = self.features(x)
        x2 = self.flatten(x1)
        x3 = self.classifier(x2)
        return x3
