- [multi_classifier_cifar10](https://github.com/Renya-Kujirada/pytorch_deeplearning_programming/tree/master/src/multi_classifier_cifar10)
  - CNNによるCIFAR10の画像認識．(チャネル数と画素数は，3 * 32 * 32)
  - CNNの構成は以下．
    - (畳み込み関数+relu)×2
    - maxpooling
    - 2層の隠れ層の全結合モデル(layer関数+relu+layer関数)
    - softmax + 交差エントロピー関数

- train.pyが訓練スクリプト．
