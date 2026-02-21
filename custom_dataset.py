from torch.utils.data import DataLoader, Dataset
import torch
from pathlib import Path
import pandas as pd
import os
from PIL import Image
from typing import Dict, List, Sequence, Tuple, Optional, Union

class ClipDataset(Dataset):
    """
    ３つのディレクトリからデータセットを作るカスタムデータセットクラス（各ディレクトリにはそれぞれのエンティティのキャプション用のcsvファイルと画像が格納されている
    """

    def __init__(self, dataset_path, preprocess, image_col="image_name", text_col="text", strict_images=True):
        """
        Args:
            dataset_path: [(csv_path, image_dir, class_id), ...]
            preprocess: 前処理関数
            image_col: csv内の画像ファイル名列
            text_col: csv内のキャプションファイル名列
            strict_images: trueの時、画像が存在しない場合例外を投げる
        """

        self.preprocess = preprocess
        self.image_col = image_col
        self.text_col = text_col
        self.strict_images = strict_images

        self.dataset = []
        for (csv_path, image_dir, class_id) in dataset_path:
            df = pd.read_csv(csv_path)
            base = Path(image_dir)
            for _, row in df.iterrows():
                img_name = str(row[self.image_col])
                text = str(row[self.text_col])
                path = os.path.join(base, img_name)
                
                if not os.path.exists(path):
                    if self.strict_images:
                        raise FileNotFoundError(f"Missing image: {path}")
                    else:
                        continue
                self.dataset.append((str(path), text, class_id))
        
        if len(self.dataset) == 0:
            raise RuntimeError("No sample collected. Check paths and csv columns")


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, text, class_id = self.dataset[idx]
        image = self.preprocess(Image.open(path).convert("RGB"))
        return image, text, class_id

class LabelDataset(Dataset):
    '''
    image_sizeとdrawing_sizeの縦横比は同じ想定
    '''

    KEY_ORDER = ["x1", "y1", "x2", "y2", "cx", "cy", "cr", "ax", "ay", "ar", "theta1", "theta2"]
    KEY_INDEX = {k:i for i,k in enumerate(KEY_ORDER)}

    CLASS_KEYS = {
        1: ["x1","y1","x2","y2"],                 # line
        2: ["cx","cy","cr"],                      # circle
        3: ["ax", "ay", "ar", "theta1", "theta2"] # arc
    }

    def __init__(
        self,
        dataset_path: Sequence[Tuple[str, str, int]],
        preprocess=None,
        strict_images=True,
        image_prefix='p',
        image_ext='.jpg',
        image_size=(224, 224),
        drawing_size=(400, 400),
        normalize_to_01=True,   # 返すvalsを0-1にするか（画像座標で）
        return_as='tuple'
    ):
        self.image_prefix = image_prefix
        self.image_ext = image_ext
        self.preprocess = preprocess
        self.strict_images = strict_images
        self.drawW, self.drawH = drawing_size
        self.W, self.H = image_size
        self.normalize_to_01 = normalize_to_01
        self.return_as = return_as

        # 図面→画像の倍率
        self.sx = self.W / self.drawW
        self.sy = self.H / self.drawH

        self.dataset: List[Tuple[str, List[float], List[float], int]] = []

        for (csv_path, image_dir, class_id) in dataset_path:
            class_id = int(class_id)
            df = pd.read_csv(csv_path, header=None)
            base = Path(image_dir)

            # 画像名のゼロパディング（基本はCSVの行数から推定するのが安全）
            num_rows = len(df)
            # zero_pad = max(1, len(str(num_rows - 1)))
            zero_pad = 5

            for i, row in df.iterrows():
                img_name = f"{self.image_prefix}{str(i).zfill(zero_pad)}{self.image_ext}"
                path = str(base / img_name)

                if self.strict_images and not os.path.exists(path):
                    raise FileNotFoundError(f"Missing image: {path}")
                if not os.path.exists(path):
                    continue

                K = len(self.KEY_ORDER)
                vals = [0.0] * K
                mask = [0.0] * K

                # ---- helper: 図面座標(x,y)→画像座標(x_img,y_img) ----
                def to_img_xy(x_draw: float, y_draw: float):
                    x_img = x_draw * self.sx
                    y_img = (self.drawH - y_draw) * self.sy  # Y反転
                    return x_img, y_img

                # ---- helper: 図面半径→画像半径 ----
                # 円としての見た目を保ちたいなら等方スケールを採用（ここでは sx を採用）
                def to_img_r(r_draw: float):
                    return r_draw * self.sx

                # ---- helper: 0-1正規化（画像座標）----
                def norm_x(x_img: float):
                    return x_img / self.W if self.normalize_to_01 else x_img

                def norm_y(y_img: float):
                    return y_img / self.H if self.normalize_to_01 else y_img

                def norm_r(r_img: float):
                    # 半径は基準をどうするかが流儀。ここでは「画像幅」で0-1化（max=W）に統一
                    return r_img / self.W if self.normalize_to_01 else r_img

                if class_id == 1:
                    # CSV: 図面座標の x1,y1,x2,y2 が row[1:5] にある前提
                    x1, y1, x2, y2 = row[1:5].astype('float32').tolist()
                    x1i, y1i = to_img_xy(x1, y1)
                    x2i, y2i = to_img_xy(x2, y2)

                    vals[self.KEY_INDEX["x1"]] = norm_x(x1i)
                    vals[self.KEY_INDEX["y1"]] = norm_y(y1i)
                    vals[self.KEY_INDEX["x2"]] = norm_x(x2i)
                    vals[self.KEY_INDEX["y2"]] = norm_y(y2i)

                elif class_id == 2:
                    # CSV: 図面座標の cx,cy,r が row[5:8] にある前提
                    cx, cy, r = row[5:8].astype('float32').tolist()
                    cxi, cyi = to_img_xy(cx, cy)
                    ri = to_img_r(r)

                    vals[self.KEY_INDEX["cx"]] = norm_x(cxi)
                    vals[self.KEY_INDEX["cy"]] = norm_y(cyi)
                    vals[self.KEY_INDEX["cr"]] = norm_r(ri)

                elif class_id == 3:
                    # CSV: 図面座標の cx,cy,r,theta1,theta2 が row[8:13]
                    ax, ay, r, t1, t2 = row[8:13].astype('float32').tolist()
                    axi, ayi = to_img_xy(ax, ay)
                    ri = to_img_r(r)

                    vals[self.KEY_INDEX["ax"]] = norm_x(axi)
                    vals[self.KEY_INDEX["ay"]] = norm_y(ayi)
                    vals[self.KEY_INDEX["ar"]] = norm_r(ri)

                    # 角度は画像座標化ではなく「表現」の問題：そのまま0-1へ
                    vals[self.KEY_INDEX["theta1"]] = t1 / 360.0
                    vals[self.KEY_INDEX["theta2"]] = t2 / 360.0

                # mask（このクラスで使うキーだけ1）
                for k in self.CLASS_KEYS.get(class_id, []):
                    mask[self.KEY_INDEX[k]] = 1.0

                self.dataset.append((path, vals, mask, class_id))

        if len(self.dataset) == 0:
            raise RuntimeError("No sample collected. Check paths / csv / image names.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, vals, mask, class_id = self.dataset[idx]
        image = self.preprocess(Image.open(path).convert("RGB")) if self.preprocess else Image.open(path).convert("RGB")
        vals = torch.tensor(vals, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        return image, vals, mask, class_id


if __name__ == "__main__":
    from torchvision import transforms

    base = r"D:/2024_Satsuka/github/DiffusionModel/data"
    arc_dir = "arc_224x224"
    line_dir = "line_224x224"
    circle_dir = "circle_224x224_val"
    items = [
        # (fr"{base}\{arc_dir}\arc_224x224.csv", fr"{base}\{arc_dir}", 3),
        # (fr"{base}\{line_dir}\line_224x224.csv", fr"{base}\{line_dir}", 1),
        (fr"{base}\{circle_dir}\circle_224x224_val.csv", fr"{base}\{circle_dir}", 2),
    ]

    batch_size=126
    preprocess = transforms.ToTensor()
    dataset = LabelDataset(items, preprocess=preprocess)

    for i, (image, vals, mask, class_id) in enumerate(dataset):
        print(image)
        print(vals)
        print(mask)
        print(class_id)

        if i == 2: break