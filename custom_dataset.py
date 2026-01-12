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
    KEY_ORDER = ["x1", "y1", "x2", "y2", "cx", "cy", "cr", "ax", "ay", "ar", "theta1", "theta2"]
    KEY_INDEX = {k:i for i,k in enumerate(KEY_ORDER)}

    # クラスごとの使用キー（例）
    CLASS_KEYS = {
        1: ["x1","y1","x2","y2"],          # 直線
        2: ["cx","cy","cr"],                # 円
        3: ["ax", "ay", "ar", "theta1", "theta2"]  # 弧（中心+半径+開始角/終了角）
    }

    def __init__(
            self,
            dataset_path: Sequence[Tuple[str, str, str]],
            preprocess=None,
            strict_images=True,
            image_prefix='p',
            image_ext = '.jpg',
            image_size = (224, 224),
            drawing_size = (400, 280),
            return_as = 'tuple'
    ):

        self.image_prefix = image_prefix
        self.image_ext = image_ext
        self.preprocess = preprocess
        self.strict_images = strict_images,
        self.drawing_size_x, self.drawing_size_y = drawing_size
        self.W, self.H = image_size

        # 図面→画像の倍率
        self.sx = self.W / self.drawing_size_x
        self.sy = self.H / self.drawing_size_y

        self.return_as = return_as
        self.col_ranges_based = {
            1: (1, 5),
            2: (5, 8),
            3: (8, 13),
        }

        self.dataset: List[Tuple[str, Union[List, Tuple], int]] = []

        # --- 図面座標 → 画像座標（convert_cood相当） ---
        def to_img_xy(x_draw: float, y_draw: float):
            x_img = x_draw * self.sx
            y_img = (self.drawing_size_y - y_draw) * self.sy  # ★Y反転
            return x_img, y_img

        # --- 図面半径 → 画像半径 ---
        # 円を円として扱うなら等方スケールが必要だが、あなたの描画は sx,sy が効く座標系なので
        # ここでは sx を代表として採用（drawingとimgの縦横比が一致している前提で自然）
        def to_img_r(r_draw: float):
            return r_draw * self.sx


        for (csv_path, image_dir, class_id) in dataset_path:
            df = pd.read_csv(csv_path, header=None)
            base = Path(image_dir)
            
            # ゼロパッド自動計算
            num_images = len(list(base.glob(f'*{self.image_ext}')))
            zero_pad = len(str(num_images - 1)) if num_images > 0 else 5

            for i, row in df.iterrows():
                img_name = f"{self.image_prefix}{str(i).zfill(zero_pad)}{self.image_ext}"
                path = str(base / img_name)
                if self.strict_images and not os.path.exists(path):
                    raise FileNotFoundError(f"Missing image: {path}")

                # 値とマスクを初期化
                K = len(self.KEY_ORDER)
                vals = [0.0]*K
                mask = [0.0]*K

                # CSV → キーに対応する列インデックスの取り方はあなたの定義に合わせる
                # 例として：直線(1):(1..4), 円(2):(5..7), 弧(3):(8..12) のような既存ルールをマッピング
                if class_id == 1:  # line: x1,y1,x2,y2（図面）
                    x1, y1, x2, y2 = row[1:5].astype("float32").tolist()
                    x1i, y1i = to_img_xy(x1, y1)
                    x2i, y2i = to_img_xy(x2, y2)

                    vals[self.KEY_INDEX["x1"]] = x1i
                    vals[self.KEY_INDEX["y1"]] = y1i
                    vals[self.KEY_INDEX["x2"]] = x2i
                    vals[self.KEY_INDEX["y2"]] = y2i

                elif class_id == 2:  # circle: cx,cy,r（図面）
                    cx, cy, r = row[5:8].astype("float32").tolist()
                    cxi, cyi = to_img_xy(cx, cy)
                    ri = to_img_r(r)

                    vals[self.KEY_INDEX["cx"]] = cxi
                    vals[self.KEY_INDEX["cy"]] = cyi
                    vals[self.KEY_INDEX["cr"]] = ri

                elif class_id == 3:  # arc: cx,cy,r,theta1,theta2（図面）
                    ax, ay, r, t1, t2 = row[8:13].astype("float32").tolist()
                    axi, ayi = to_img_xy(ax, ay)
                    ri = to_img_r(r)

                    vals[self.KEY_INDEX["ax"]] = axi
                    vals[self.KEY_INDEX["ay"]] = ayi
                    vals[self.KEY_INDEX["ar"]] = ri

                    # 角度は画像座標化ではないので表現だけ整える
                    vals[self.KEY_INDEX["theta1"]] = t1 / 360.0
                    vals[self.KEY_INDEX["theta2"]] = t2 / 360.0


                # mask（このクラスで使用するキーのみ 1）
                for k in self.CLASS_KEYS.get(class_id, []):
                    mask[self.KEY_INDEX[k]] = 1.0


                self.dataset.append((path, vals, mask, int(class_id)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, vals, mask, class_id = self.dataset[idx]
        image = self.preprocess(Image.open(path).convert("RGB"))
        vals = torch.tensor(vals, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        return image, vals, mask, class_id

if __name__ == "__main__":
    from torchvision import transforms

    base = r"D:/2024_Satsuka/github/DiffusionModel/data"
    arc_dir = "arc_224x224"
    line_dir = "line_224x224"
    circle_dir = "circle_224x224"
    items = [
        # (fr"{base}\{arc_dir}\arc_224x224.csv", fr"{base}\{arc_dir}", 3),
        # (fr"{base}\{line_dir}\line_224x224.csv", fr"{base}\{line_dir}", 1),
        (fr"{base}\{circle_dir}\circle_224x224.csv", fr"{base}\{circle_dir}", 2),
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