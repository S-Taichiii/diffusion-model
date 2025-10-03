from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pandas as pd
import os
from PIL import Image

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
        for (csv_path, image_dir, _) in dataset_path:
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
                self.dataset.append((str(path), text))
        
        if len(self.dataset) == 0:
            raise RuntimeError("No sample collected. Check paths and csv columns")


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, text = self.dataset[idx]
        image = self.preprocess(Image.open(path).convert("RGB"))
        return image, text
