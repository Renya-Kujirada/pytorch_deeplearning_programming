# src内の説明


## [model_tuning](https://github.com/Renya-Kujirada/pytorch_deeplearning_programming/tree/master/src/model_tuning)
- [multi_classifier_cifar10](https://github.com/Renya-Kujirada/pytorch_deeplearning_programming/tree/master/src/multi_classifier_cifar10)の精度向上のため，下記を検証
  - Dropout: 学習毎に，ランダムに中間テンソルの要素の出力値を0にする．
  - Batch Normalization
  - Data Augumentation

## [multi_classifier_cifar10](https://github.com/Renya-Kujirada/pytorch_deeplearning_programming/tree/master/src/multi_classifier_cifar10)
- CNNによるCIFAR10の画像認識．(チャネル数と画素数は，3 * 32 * 32)
- CNNの構成は以下．
  - (畳み込み関数+relu)×2
  - maxpooling
  - 隠れ層×2の全結合モデル(layer関数 + relu + layer関数)
  - softmax + 交差エントロピー関数
- train.pyが訓練スクリプト．
- 50epochの実験の結果，過学習気味であることがわかり，20epoch程度で止めておくべきであると考えられる．
- 精度としては66%程度．


## [multi_classifier_iris](https://github.com/Renya-Kujirada/pytorch_deeplearning_programming/tree/master/src/multi_classifier_iris)
- irisの多値分類．
- 後で追記

## [mlflow](https://github.com/Renya-Kujirada/pytorch_deeplearning_programming/tree/master/src/mlflow)
- mlflowで色々遊んでみた
- 後で追記
