# src内の説明

## [fine_tuning](https://github.com/Renya-Kujirada/pytorch_deeplearning_programming/tree/master/src/fine_tuning)
- 事前学習済みモデルを利用した効率の良い学習方法には，ファインチューニングと転移学習がある．一般的に，学習データが大量にある場合はファインチューニングが，学習データが少ない場合は転移学習が向いていると言われている．
  - ファインチューニング：事前学習済みモデルのパラメータを初期値として利用し，全レイヤー関数のパラメータを学習する．
  - 転移学習：事前学習済みモデルモデルのうち，入力に近い部分のレイヤー関数はすべて固定し，出力に近い部分のみ学習する．

## [model_tuning](https://github.com/Renya-Kujirada/pytorch_deeplearning_programming/tree/master/src/model_tuning)
- [multi_classifier_cifar10](https://github.com/Renya-Kujirada/pytorch_deeplearning_programming/tree/master/src/multi_classifier_cifar10)の精度向上のため，下記を検証
  - Dropout: 学習毎に，ランダムに中間テンソルの要素の出力値を0にすることで，過学習を回避する．過学習に対して頑健になるが，学習に要する時間が長くなる．
  - Batch Normalization: ミニバッチ学習時，畳込み層の出力に対して正規化処理を実施した後，次の畳込み層の入力とすることで，学習効率向上と過学習回避を目指す．
  - Data Augumentation: 学習前の入力データに対しランダムに加工（反転，切り抜き，リサイズ，アフィン変換，，）を施し学習データのバリエーションを増やすことで，頑健性を向上させる．

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
