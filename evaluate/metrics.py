import numpy as np
import cv2
from pathlib import Path
import glob
from scipy.ndimage import distance_transform_edt


def chamfer_distance(image1_path, image2_path):
    mask1 = make_binary_mask(image1_path)
    mask2 =make_binary_mask(image2_path)

    """双方向Chamfer距離（線どうしの距離の平均）"""
    da = distance_transform_edt(1 - mask1)
    db = distance_transform_edt(1 - mask2)

    # 双方向平均
    return (np.mean(da[mask2 > 0]) + np.mean(db[mask1 > 0])) / 2

def make_binary_mask(image_path):
    """
    Chamfer 距離用の 0/1 マスクを作成するヘルパー関数
    線:1, 背景:0
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    blur = cv2.GaussianBlur(img, (3, 3), 0)
    _, bin_img = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    # 0/255 → 0/1 にする
    mask = (bin_img > 0).astype(np.uint8)
    return mask

def get_sorted_image_list(dir_path, exts=["*.jpg", "*.png"], count = None):
    dir_path = Path(dir_path)
    files = []
    for ext in exts:
        pattern = str(dir_path / ext)
        files.extend(glob.glob(pattern))

    # ファイル名に含まれる数字部分でソート
    def num_key(p):
        name = Path(p).stem           # 例: "p00000", "pic1"
        digits = ''.join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else -1  # 数字が無い場合はとりあえず -1

    files = sorted(files, key=num_key)

    return files[:count] if count is not None else files
