import os
import datetime
import torch
import matplotlib.pyplot as plt
import csv
import inspect

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torch import nn
# from models.unet2 import Unet
# from diff import Diffuser

class Utils:
    @staticmethod
    def recordResult(model=None, losses=None, images=None, **kwargs) -> None:
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
            if losses: 
                Utils.saveLossToGraph(dir_path, losses)
                Utils.saveLossToCsv(dir_path, losses)

            # 生成画像を保存
            if images:
                image_dir = os.path.join(dir_path, "generated_pic")
                os.makedirs(image_dir)
                Utils.saveImages(image_dir, images)

        except Exception as e:
            print(f'エラーが発生しました：{e}')

    @staticmethod
    def saveModelParameter(dir_path: str, model: nn.Module) -> None:
        output_path: str = os.path.join(dir_path, "trained_para.pth")
        torch.save(model.state_dict(), output_path) 
    
    @staticmethod
    def loadModel(path: str, model: nn.Module, device='cpu') -> nn.Module:
        model.to(device=device)
        model.load_state_dict(torch.load(path))
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
    def saveImages(dir_path: str, images) -> None:
        # 画像をまとめて保存
        Utils.concat_images(dir_path, images)

        # 画像を個々に保存する
        for i, image in enumerate(images):
            fig = plt.figure(facecolor="gray")
            file_name = dir_path + f'/pic{i+1}.png'
            plt.imshow(image)
            plt.axis("off")
            plt.savefig(file_name)
            plt.close()


    @staticmethod
    def concat_images(dir_path: str, images, rows: int=2, cols: int=10) -> None:
        file_name: str = dir_path + f'/catpic1_{rows * cols}.png'
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

class Datasets(Dataset):
    # パスとtransformの取得
    def __init__(self, img_dir, transform=None):
        self.img_paths = self._get_img_paths(img_dir)
        self.transform = transform

    # データの取得
    def __getitem__(self, index): 
        path = self.img_paths[index]
        img= Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    # パスの取得
    def _get_img_paths(self, img_dir):
        img_dir = Path(img_dir)
        img_paths = [path for path in img_dir.iterdir() if path.suffix == ".jpg"]
        return img_paths

    # データの数を取得
    def __len__(self):
        return len(self.img_paths)

if __name__ == "__main__":
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # diff = Diffuser(device=device)
    # model = Utils.loadModel("result/2025_01_20_14_23/trained_para.pth", Unet(), device=device)

    # images = diff.sample(model, x_shape=(50, 3, 32, 32))
    # Utils.recordResult(images=images)
    
    Utils.recordResult()