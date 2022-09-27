import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib


class Summarize():
    def create_numpy_result_from(self, result_path):
        history = np.loadtxt(result_path, delimiter=",")
        return history

    def evaluate_history(self, xaxis, history_train, history_val, y_label, title, fig_path):
        num_epoch = len(history_train)
        unit = num_epoch / 10

        # 学習曲線の表示
        plt.figure(figsize=(9, 8))
        plt.plot(xaxis, history_train, "b", label="訓練")
        plt.plot(xaxis, history_val, "k", label="検証")
        plt.grid()
        plt.xticks(np.arange(0, num_epoch+1, unit))
        plt.xlabel("epoch")
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.show()
        plt.savefig(fig_path)


if __name__ == "__main__":
    fig_dir = "./src/multi_classifier_cifar10/result"
    result_path = f"{fig_dir}/result.csv"
    fig_loss_path = f"{fig_dir}/train_loss_curve.png"
    fig_acc_path = f"{fig_dir}/train_acc_curve.png"

    summarizer = Summarize()
    history = summarizer.create_numpy_result_from(result_path)
    summarizer.evaluate_history(history[:, 0], history[:, 1], history[:, 3], "loss", "学習曲線（損失）", fig_loss_path)
    summarizer.evaluate_history(history[:, 0], history[:, 2], history[:, 4], "accuracy", "学習曲線（精度）", fig_acc_path)
