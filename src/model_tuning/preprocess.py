import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class Preprocess():
    def __init__(self, mu, sigma, dataset_path):
        self.mu = mu
        self.sigma = sigma
        self.data_root = dataset_path

    def prepare_dataset(self):
        transform_train = transforms.Compose([
            # ランダムに左右反転
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),  # データのテンソル化
            transforms.Normalize(self.mu, self.sigma),  # データの正規化
            # ランダムに部分領域を削除
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),  # データのテンソル化
            transforms.Normalize(self.mu, self.sigma),  # データの正規化
        ])

        # 訓練データセット，3階テンソル
        train_set = datasets.CIFAR10(
            root=self.data_root,
            train=True,
            download=True,
            transform=transform_train
        )
        # 検証データセット，3階テンソル
        test_set = datasets.CIFAR10(
            root=self.data_root,
            train=False,
            download=True,
            transform=transform_test
        )
        return train_set, test_set

    def torch_seed(self, seed=123):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.use_determinstic_algorithms = True
