from torch.utils.data import DataLoader, Dataset
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
            dataset_path: [(csv_path, image_dir, class_name), ...]
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
        for (csv_path, image_dir, class_name) in dataset_path:
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
                self.dataset.append((str(path), text, class_name))
        
        if len(self.dataset) == 0:
            raise RuntimeError("No sample collected. Check paths and csv columns")


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, text, class_name = self.dataset[idx]
        image = self.preprocess(Image.open(path).convert("RGB"))
        return image, text, class_name

class LabelDataset(Dataset):
    def __init__(
            self,
            dataset_path: Sequence[Tuple[str, str, str]],
            preprocess=None,
            strict_images=True,
            image_prefix='p',
            image_ext = '.jpg',
            return_as = 'tuple'
    ):

        self.image_prefix = image_prefix
        self.image_ext = image_ext
        self.preprocess = preprocess
        self.strict_images = strict_images
        self.return_as = return_as
        self.col_ranges_based = {
            1: (1, 5),
            2: (5, 8),
            3: (8, 13),
        }

        self.dataset: List[Tuple[str, Union[List, Tuple], int]] = []

        for (csv_path, image_dir, class_name) in dataset_path:
            df = pd.read_csv(csv_path, header=None)
            base = Path(image_dir)
            
            # ゼロパッド自動計算
            num_images = len(list(base.glob(f'*{self.image_ext}')))
            zero_pad = len(str(num_images - 1)) if num_images > 0 else 5

            if class_name in self.col_ranges_based.keys():
                start_idx, end_idx = self.col_ranges_based[class_name]
            else:
                start_idx, end_idx = None, None

            for i, row in df.iterrows():
                img_name = f"{self.image_prefix}{str(i).zfill(zero_pad)}{self.image_ext}"
                path = str(base / img_name)

                if self.strict_images and not os.path.exists(path):
                    raise FileNotFoundError(f"Missing image: {path}")

                # 指定範囲の列データを抽出
                if start_idx is not None:
                    values = row[start_idx:end_idx]
                    cols = tuple(values) if self.return_as == 'tuple' else list(values)
                else:
                    cols = []

                self.dataset.append((path, cols, class_name))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, data_info_tuple, class_name = self.dataset[idx]
        image = self.preprocess(Image.open(path).convert("RGB"))
        # return image, data_info_tuple, class_name
        return path, data_info_tuple, class_name

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

    for i, (image, data_info_tuple, class_name) in enumerate(dataset):
        print(image)
        print(data_info_tuple)
        print(class_name)

        if i==2: break