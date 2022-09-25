import torch
from preprocess import Preprocess


def main():
    seed = 123
    batch_size = 100
    mu = 0.5
    sigma = 0.5
    dataset_path = "./data"

    preprocess = Preprocess(mu, sigma, dataset_path)
    train_set, test_set = preprocess.prepare_dataset()


if __name__ == "__main__":
    main()
