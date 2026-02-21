import os
import datetime
import torch
import matplotlib.pyplot as plt
import csv
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from diff import Diffuser
from models.unet import Unet
# from models.unet2 import Unet
# from diff import Diffuser

class Utils:
    @staticmethod
    def recordResult(model=None, train_losses=None, val_losses=None, images=None, **kwargs) -> None:
        try:
            # current directory上にresultディレクトリがない場合は作成
            cd = os.getcwd()
            result_dir = os.path.join(cd, "result")
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            # resultディレクトリ上に現在の日時の名前のディレクトリを作成、そこに各種記録を保存
            now = datetime.datetime.now()
            dir_name: str = now.strftime(now.strftime("%Y_%m_%d_%H_%M"))
            dir_path: str = os.path.join(result_dir, dir_name)
            if os.path.exists(dir_path):
                raise FileExistsError(f'{dir_path} is already exists')
            os.makedirs(dir_path, exist_ok=True)

            # ハイパーパラメーター、学習時間などの情報をテキストファイルに書き込み、保存
            if kwargs:
                file_name: str = "record.txt"
                file_path = os.path.join(dir_path, file_name)

                with open(file_path, 'w', encoding='utf-8') as f:
                    for key, value in kwargs.items():
                        if key == "learning_time":
                            f.write(f'{key} : {value} (s)\n')
                            continue
                        f.write(f'{key} : {value}\n')

            # 学習済みモデルのパラメーターの保存
            if model: Utils.saveModelParameter(dir_path, model)

            # lossのグラフを保存, 値をcsvに出力
            if train_losses and val_losses: 
                Utils.saveTrainValLossGraph(dir_path, train_losses, val_losses)
                Utils.saveTrainValLossCsv(dir_path, train_losses, val_losses)

            # 生成画像を保存
            if images:
                image_dir = os.path.join(dir_path, "generated_pic_arc")
                os.makedirs(image_dir)
                Utils.saveImages(image_dir, images)

        except Exception as e:
            print(f'Error: {e}')

    @staticmethod
    def saveModelParameter(dir_path: str, model: nn.Module) -> None:
        output_path: str = os.path.join(dir_path, "trained_para.pth")
        torch.save(model.state_dict(), output_path) 
    
    @staticmethod
    def loadModel(path: str, model: nn.Module, device='cpu') -> nn.Module:
        """modelはインスタンスを入力しないとエラーになる"""
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device=device)
        model.eval()
        return model

    @staticmethod
    def saveLossToGraph(dir_path: str, losses) -> None:
        file_path: str = os.path.join(dir_path, "losses.png")
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(file_path)
        plt.close()

    @staticmethod
    def saveLossToCsv(dir_path: str, losses) -> None:
        file_path: str = os.path.join(dir_path, "losses.csv")
        with open(file_path, 'w', newline="") as f:
            title = ["epoch", "loss"]
            writer = csv.writer(f)
            writer.writerow(title)

            for i, loss in enumerate(losses):
                writer.writerow([i+ 1, loss])

            writer.writerow(["最小値", min(losses)])

    @staticmethod
    def saveTrainValLossGraph(dir_path: str, train_losses, val_losses, filename: str = "losses_train_val.png") -> None:
        """
        train_loss と val_loss を同じ図に描いて保存する。
        長さが異なる場合は短い方に合わせて描画する。
        """
        n = min(len(train_losses), len(val_losses))
        if n == 0:
            print("Warning: no data to plot (train/val losses are empty).")
            return
        if len(train_losses) != len(val_losses):
            print(f"Note: train({len(train_losses)}) and val({len(val_losses)}) lengths differ; plotting first {n} epochs.")

        x = list(range(1, n + 1))
        plt.figure()
        plt.plot(x, train_losses[:n], label="train_loss")
        plt.plot(x, val_losses[:n], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train & Val Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(dir_path, filename)
        plt.savefig(out_path)
        plt.close()

    @staticmethod
    def saveTrainValLossGraph(dir_path: str, train_losses, val_losses, filename: str = "losses_train_val.png") -> None:
        """
        train_loss と val_loss を同じ図に描いて保存する。
        val が 5epoch ごと等で疎な場合（np.nan を含む）にも対応。
        """

        if train_losses is None or len(train_losses) == 0:
            print("Warning: train_losses is empty.")
            return

        epochs = len(train_losses)
        x = np.arange(1, epochs + 1)

        train_arr = np.asarray(train_losses, dtype=float)

        # val は無い/短い/NaN含みを許容
        if val_losses is None:
            val_arr = np.full(epochs, np.nan, dtype=float)
        else:
            val_arr = np.asarray(val_losses, dtype=float)
            if len(val_arr) < epochs:
                # 足りない分は NaN で埋める
                val_arr = np.concatenate([val_arr, np.full(epochs - len(val_arr), np.nan)])
            elif len(val_arr) > epochs:
                val_arr = val_arr[:epochs]

        plt.figure()
        plt.plot(x, train_arr, label="train_loss")

        # NaN じゃないところだけプロット（疎に点が出る）
        ok = np.isfinite(val_arr)
        if np.any(ok):
            plt.plot(x[ok], val_arr[ok], label="val_loss", marker="o", linestyle="-")
        else:
            print("Note: val_losses has no finite values; only train_loss will be plotted.")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train & Val Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = os.path.join(dir_path, filename)
        plt.savefig(out_path)
        plt.close()

    @staticmethod
    def saveTrainValLossCsv(dir_path: str, train_losses, val_losses, filename: str = "losses_train_val.csv") -> None:
        """
        train/val の損失を 1 つの CSV にまとめて保存する。
        val が 5epoch ごと等で疎な場合（np.nan を含む）にも対応。
        val 未評価epochは空欄で出力。
        """
        import os, csv
        import numpy as np

        os.makedirs(dir_path, exist_ok=True)

        if train_losses is None or len(train_losses) == 0:
            print("Warning: train_losses is empty.")
            return

        epochs = len(train_losses)
        train_arr = np.asarray(train_losses, dtype=float)

        if val_losses is None:
            val_arr = np.full(epochs, np.nan, dtype=float)
        else:
            val_arr = np.asarray(val_losses, dtype=float)
            if len(val_arr) < epochs:
                val_arr = np.concatenate([val_arr, np.full(epochs - len(val_arr), np.nan)])
            elif len(val_arr) > epochs:
                val_arr = val_arr[:epochs]

        out_path = os.path.join(dir_path, filename)
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])
            for i in range(epochs):
                v = val_arr[i]
                writer.writerow([i + 1, float(train_arr[i]), "" if not np.isfinite(v) else float(v)])

            # min は NaN を無視して計算
            writer.writerow(["min_train", float(np.nanmin(train_arr)), ""])
            if np.any(np.isfinite(val_arr)):
                writer.writerow(["min_val", "", float(np.nanmin(val_arr))])
            else:
                writer.writerow(["min_val", "", ""])


    @staticmethod
    def saveImages(dir_path: str, images) -> None:
        # 画像をまとめて保存
        # Utils.concat_images(dir_path, images)

        # 画像を個々に保存する
        for i, image in enumerate(images):
            file_name = os.path.join(dir_path, f'pic{i+1}.png')
            image.save(file_name)


    @staticmethod
    def concat_images(dir_path: str, images, rows: int=2, cols: int=10) -> None:
        file_name: str = os.path.join(dir_path, f'catpic1_{rows * cols}.png')
        fig = plt.figure(figsize=(cols, rows), facecolor='gray')
        i = 0
        for r in range(rows):
            for c in range(cols):
                fig.add_subplot(rows, cols, i + 1)
                plt.imshow(images[i])
                plt.axis("off") # 縦軸、横軸を非表示にする
                i += 1

        plt.savefig(file_name)
        plt.close()

    @staticmethod
    def generate(model_path: str, num: int):
        # current directory上にgenerate_picディレクトリがない場合は作成
        cd = os.getcwd()
        result_dir = os.path.join(cd, "generate_pic")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        image_dir = os.path.join(result_dir, "generated_pic")
        os.makedirs(image_dir)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        diff = Diffuser(device=device)
        model = Utils.loadModel(model_path, Unet(), device=device)
        images = diff.sample(model, x_shape=(num, 3, 32, 32))
        Utils.saveImages(image_dir, images)


if __name__ == "__main__":
    # sample code: use of Utils.loadModel()
    model_path = "result/2025_02_02_04_23/trained_para.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diff = Diffuser(device=device)
    model = Utils.loadModel(model_path, Unet(), device=device)
    images = diff.sample(model, x_shape=(500, 3, 32, 32))
    print(len(images))
    Utils.recordResult(images=images)
    
    # Utils.recordResult()