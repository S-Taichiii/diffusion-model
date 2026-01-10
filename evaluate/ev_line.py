import cv2
import numpy as np
from pathlib import Path
from math import atan2, degrees 
from .metrics import chamfer_distance, get_sorted_image_list



# ==========================================
# 端点の導出
# ==========================================
def getEndPoint(image_path):
    """
    端点の導出
    """
    # 画像読み込み（グレースケール）
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # ノイズ軽減（必要なら）
    blur = cv2.GaussianBlur(img, (3, 3), 0)

    # 二値化（線を黒, 背景を白と想定）
    # もし線が白で背景が黒なら THRESH_BINARY の方に切り替え
    _, bin_img = cv2.threshold(blur, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    # エッジ検出
    edges = cv2.Canny(bin_img, 50, 150, apertureSize=3)

    # Hough変換で線分検出
    # パラメータは画像に合わせて適宜調整
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=10,
        minLineLength=5,
        maxLineGap=5
    )

    if lines is None:
        raise RuntimeError(f"No line detected in {Path(image_path).name}")

    # 最長の線分を採用
    max_len = -1
    best_line = None
    for l in lines:
        x1, y1, x2, y2 = l[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length > max_len:
            max_len = length
            best_line = (x1, y1, x2, y2)

    return best_line


def end_point_error(img1_path, img2_path):
    """
    端点誤差の導出
    """
    ep1 = getEndPoint(img1_path)
    ep2 = getEndPoint(img2_path)

    # 端点のペアは決まっていない場合の導出
    A1 = np.array([ep1[0], ep1[1]], dtype=float)
    A2 = np.array([ep1[2], ep1[3]], dtype=float)
    B1 = np.array([ep2[0], ep2[1]], dtype=float)
    B2 = np.array([ep2[2], ep2[3]], dtype=float)

    d1 = np.linalg.norm(A1 - B1) + np.linalg.norm(A2 - B2)
    d2 = np.linalg.norm(A1 - B2) + np.linalg.norm(A2 - B1)

    return min(d1, d2) / 2.0

def length_error(img1_path, img2_path):
    """
    長さの誤差を導出
    """
    ep1 = getEndPoint(img1_path)
    ep2 = getEndPoint(img2_path)

    # 端点のペアは決まっていない場合の導出
    A1 = np.array([ep1[0], ep1[1]], dtype=float)
    A2 = np.array([ep1[2], ep1[3]], dtype=float)
    B1 = np.array([ep2[0], ep2[1]], dtype=float)
    B2 = np.array([ep2[2], ep2[3]], dtype=float)

    len1 = np.linalg.norm(A1 - A2)
    len2 = np.linalg.norm(B1 - B2)

    return abs(len1 - len2)


def detect_line_angle(image_path, debug=False):
    """
    角度誤差の導出
    単一の直線が描かれた画像から角度[deg]を推定する関数。
    x軸正方向(右向き)を0度として、反時計回りが正。
    """
    x1, y1, x2, y2 = getEndPoint(image_path)

    # 角度を計算（rad -> deg）
    angle_rad = atan2(y2 - y1, x2 - x1)
    angle_deg = degrees(angle_rad)

    # 見やすいように -90〜90 に正規化しておく（任意）
    # 線は向きなしなので、180度周期で同じとみなせる
    if angle_deg < -90:
        angle_deg += 180
    elif angle_deg > 90:
        angle_deg -= 180

    if debug:
        print(f"{image_path}: line=({x1},{y1})-({x2},{y2}), angle={angle_deg:.2f} deg")

    return angle_deg


def angle_error_deg(img1_path, img2_path, debug=False):
    """
    2枚の画像の直線角度誤差[deg]を返す。
    線の向きは考えず、0〜90度の範囲に収める。
    """
    theta1 = detect_line_angle(img1_path, debug=debug)
    theta2 = detect_line_angle(img2_path, debug=debug)

    diff = abs(theta1 - theta2)
    # 線は向きなし → 180度対称
    diff = min(diff, 180 - diff)

    if debug:
        print(f"angle1={theta1:.2f}, angle2={theta2:.2f}, error={diff:.2f} deg")

    return diff

def evaluate_line(img1_dir, img2_dir):
    img1_dir = get_sorted_image_list(img1_dir)
    img2_dir = get_sorted_image_list(img2_dir)

    pic_count = min(len(img1_dir), len(img2_dir))

    img1_dir = img1_dir[:pic_count]
    img2_dir = img2_dir[:pic_count]

    angle_errors = []
    endpoint_errors = []
    length_errors = []
    chamfers = []
    skipped = 0
    for image1, image2 in zip(img1_dir, img2_dir):
        try:
            # 角度誤差
            angle_error = angle_error_deg(image1, image2, debug=False)
            angle_errors.append(angle_error)

            # 端点誤差
            ep_error = end_point_error(image1, image2)
            endpoint_errors.append(ep_error)

            # 長さ誤差
            l_error = length_error(image1, image2)
            length_errors.append(l_error)

            # chamfer距離
            chamfer_dis = chamfer_distance(image1, image2)
            chamfers.append(chamfer_dis)
        except Exception as e:
            # 検出失敗したらスキップ
            print(f"[SKIP] {Path(image1).name} vs {Path(image2).name} : {e}")
            skipped += 1
            continue

        # print(f"Angle error: {angle_error:.4f} degrees")
        # print(f'Endpoint Error: {ep_error:.4f}')

    valid_count = pic_count - skipped
    # --- 統計量の計算 ---
    angle_mean = np.mean(angle_errors)
    angle_std  = np.std(angle_errors)
    ep_mean = np.mean(endpoint_errors)
    ep_std  = np.std(endpoint_errors)
    length_mean = np.mean(length_errors)
    length_std  = np.std(length_errors)
    chamfer_mean = np.mean(chamfers)
    chamfer_std  = np.std(chamfers)

    print("\n========================")
    print(f"Total pairs: {pic_count}, skipped: {skipped}, valid: {valid_count}")
    print("------------------------")
    print(f"Angle Error Mean      : {angle_mean:.4f}")
    print(f"Angle Error Std       : {angle_std:.4f}")
    print("------------------------")
    print(f"Endpoint Error Mean   : {ep_mean:.4f}")
    print(f"Endpoint Error Std    : {ep_std:.4f}")
    print("------------------------")
    print(f"Lenght Error Mean     : {length_mean:.4f}")
    print(f"Lenght Error Std      : {length_std:.4f}")
    print("------------------------")
    print(f"Chamfer Error Mean    : {chamfer_mean:.4f}")
    print(f"Chamfer Error Std     : {chamfer_std:.4f}")
    print("========================")


# if __name__ == "__main__":
#     img1_dir = "D:/2024_Satsuka/github/DiffusionModel/data/line_224x224_val"   # 1枚目の画像パス
#     img2_dir = "D:/2024_Satsuka/github/DiffusionModel/generated_by_cond/2025_12_09_13_31/line"   # 2枚目の画像パス

#     pic_count = 100
#     img1_dir = get_sorted_image_list(img1_dir, count=pic_count)
#     img2_dir = get_sorted_image_list(img2_dir, count=pic_count)

#     angle_errors = []
#     endpoint_errors = []
#     length_errors = []
#     chamfers = []
#     skipped = 0
#     for image1, image2 in zip(img1_dir, img2_dir):
#         try:
#             # 角度誤差
#             angle_error = angle_error_deg(image1, image2, debug=False)
#             angle_errors.append(angle_error)

#             # 端点誤差
#             ep_error = end_point_error(image1, image2)
#             endpoint_errors.append(ep_error)

#             # 長さ誤差
#             l_error = length_error(image1, image2)
#             length_errors.append(l_error)

#             # chamfer距離
#             chamfer_dis = chamfer_distance(image1, image2)
#             chamfers.append(chamfer_dis)
#         except Exception as e:
#             # 検出失敗したらスキップ
#             print(f"[SKIP] {Path(image1).name} vs {Path(image2).name} : {e}")
#             skipped += 1
#             continue

#         print(f"Angle error: {angle_error:.4f} degrees")
#         print(f'Endpoint Error: {ep_error:.4f}')

#     valid_count = pic_count - skipped
#     # --- 統計量の計算 ---
#     angle_mean = np.mean(angle_errors)
#     angle_std  = np.std(angle_errors)
#     ep_mean = np.mean(endpoint_errors)
#     ep_std  = np.std(endpoint_errors)
#     length_mean = np.mean(length_errors)
#     length_std  = np.std(length_errors)
#     chamfer_mean = np.mean(chamfers)
#     chamfer_std  = np.std(chamfers)

#     print("\n========================")
#     print(f"Total pairs: {pic_count}, skipped: {skipped}, valid: {valid_count}")
#     print("------------------------")
#     print(f"Angle Error Mean      : {angle_mean:.4f}")
#     print(f"Angle Error Std       : {angle_std:.4f}")
#     print("------------------------")
#     print(f"Endpoint Error Mean   : {ep_mean:.4f}")
#     print(f"Endpoint Error Std    : {ep_std:.4f}")
#     print("------------------------")
#     print(f"Lenght Error Mean     : {length_mean:.4f}")
#     print(f"Lenght Error Std      : {length_std:.4f}")
#     print("------------------------")
#     print(f"Chamfer Error Mean    : {chamfer_mean:.4f}")
#     print(f"Chamfer Error Std     : {chamfer_std:.4f}")
#     print("========================")