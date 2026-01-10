import cv2
import numpy as np
from pathlib import Path
from .metrics import get_sorted_image_list, chamfer_distance

def detect_circle(image_path, debug=False):
    """
    単一の円が描かれた画像から
    (cx, cy, r) を返す。
    見つからなければ RuntimeError を投げる。
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # ノイズ軽減
    blur = cv2.GaussianBlur(img, (5, 5), 1.0)

    # 二値化（線を黒, 背景を白と想定）
    # もし線が白で背景が黒なら THRESH_BINARY の方に切り替え
    _, bin_img = cv2.threshold(blur, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    # エッジ検出
    edges = cv2.Canny(bin_img, 50, 150, apertureSize=3)

    # HoughCircles で円検出
    # パラメータはデータに合わせて調整が必要
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=10,
        param1=50,   # Canny 上限
        param2=5,    # 円判定のしきい値（小さいほど検出しやすいが誤検出も増える）
        minRadius=1,
        maxRadius=224   # 0 なら制限なし
    )

    if circles is None:
        raise RuntimeError(f"No circle detected in {Path(image_path).name}")

    # 最もスコアの高い円を一つ採用
    circles = circles[0, :].astype("float")
    # circles[i] = (cx, cy, r)
    cx, cy, r = circles[0]

    if debug:
        print(f"{image_path}: cx={cx}, cy={cy}, r={r}")

    return circles[0]


def center_error(img1_path, img2_path, debug=False):
    """
    2枚の円画像の中心誤差 [ピクセル] を返す。
    入力：画像パス2枚
    出力：中心座標間のユークリッド距離
    """
    cx1, cy1, _ = detect_circle(img1_path, debug=debug)
    cx2, cy2, _ = detect_circle(img2_path, debug=debug)

    p1 = np.array([cx1, cy1], dtype=float)
    p2 = np.array([cx2, cy2], dtype=float)

    dist = np.linalg.norm(p1 - p2)

    if debug:
        print(f"Center error: {dist:.4f} px")

    return dist


def radius_error(img1_path, img2_path, debug=False):
    """
    2枚の円画像の半径誤差 [ピクセル] を返す。
    入力：画像パス2枚
    出力：半径の絶対差
    """
    _, _, r1 = detect_circle(img1_path, debug=debug)
    _, _, r2 = detect_circle(img2_path, debug=debug)

    err = abs(r1 - r2)

    if debug:
        print(f"Radius error: {err:.4f} px")

    return err


def evaluate_circle(img1_dir, img2_dir):
    img1_dir = get_sorted_image_list(img1_dir)
    img2_dir = get_sorted_image_list(img2_dir)

    pic_count = min(len(img1_dir), len(img2_dir))

    img1_dir = img1_dir[:pic_count]
    img2_dir = img2_dir[:pic_count]

    center_errors = []
    radius_errors = []
    chamfers = []
    skipped = 0
    for image1, image2 in zip(img1_dir, img2_dir):
        try:
            # 中心誤差
            c_error = center_error(image1, image2)
            center_errors.append(c_error)

            # 半径誤差
            r_error = radius_error(image1, image2)
            radius_errors.append(r_error)

            # chamfer距離
            chamfer_dis = chamfer_distance(image1, image2)
            chamfers.append(chamfer_dis)
        except Exception as e:
            # 検出失敗したらスキップ
            print(f"[SKIP] {Path(image1).name} vs {Path(image2).name} : {e}")
            skipped += 1
            continue

        # print(f"Center error: {c_error:.4f} degrees")
        # print(f'Radius Error: {r_error:.4f}')

    valid_count = pic_count - skipped

    # --- 統計量の計算 ---
    c_errors_mean = np.mean(center_errors)
    c_errors_std  = np.std(center_errors)
    r_errors_mean = np.mean(radius_errors)
    r_errors_std  = np.std(radius_errors)
    chamfer_mean = np.mean(chamfers)
    chamfer_std  = np.std(chamfers)

    print("\n========================")
    print(f"Total pairs: {pic_count}, skipped: {skipped}, valid: {valid_count}")
    print("------------------------")
    print(f"Center Error Mean      : {c_errors_mean:.4f}")
    print(f"Center Error Std       : {c_errors_std:.4f}")
    print("------------------------")
    print(f"Radius Error Mean   : {r_errors_mean:.4f}")
    print(f"Radius Error Std    : {r_errors_std:.4f}")
    print("------------------------")
    print(f"Chamfer Error Mean    : {chamfer_mean:.4f}")
    print(f"Chamfer Error Std     : {chamfer_std:.4f}")
    print("========================")


# =========================================================
# Arc angle estimation (NEW)
# =========================================================
def _load_binary_for_points(image_path: str) -> np.ndarray:
    """黒線/白背景想定で線画を1、それ以外を0にする二値画像(0/255)を作る。"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    blur = cv2.GaussianBlur(img, (5, 5), 1.0)
    _, bin_img = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return bin_img


def _extract_line_points(image_path: str) -> np.ndarray:
    """線画ピクセルを (N,2) の (x,y) 点群で返す。"""
    bin_img = _load_binary_for_points(image_path)
    ys, xs = np.where(bin_img > 0)
    if len(xs) < 20:
        raise RuntimeError(f"Too few pixels detected in {Path(image_path).name}")
    return np.stack([xs, ys], axis=1).astype(np.float64)


def _circular_diff_deg(a: float, b: float) -> float:
    """角度差を 0..180 に畳む（deg）。"""
    d = abs((a - b) % 360.0)
    return float(min(d, 360.0 - d))


def detect_arc_angles(image_path: str, debug: bool = False):
    """
    円弧画像から開始角・終了角（deg, 0..360）を推定して返す。

    NOTE:
      - 中心(cx,cy)と半径rは detect_circle() をそのまま流用
      - 点群角度の「最大ギャップ」を跨がない区間を弧の区間として採用
    """
    cx, cy, _ = detect_circle(image_path, debug=debug)
    pts = _extract_line_points(image_path)

    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    ang = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
    ang.sort()

    # 最後→最初は +360 して差分を取る
    gaps = np.diff(np.r_[ang, ang[0] + 360.0])
    k = int(np.argmax(gaps))

    # 最大ギャップの次から始めると、点群を覆う最短区間になる
    start = float(ang[(k + 1) % len(ang)])
    span = float(360.0 - gaps[k])
    end = float((start + span) % 360.0)

    if debug:
        print(f"{image_path}: start={start:.3f}, end={end:.3f}, span={span:.3f}")

    return start, end

def detect_arc_geometry(image_path: str, debug: bool = False):
    """
    円弧画像から
      - start angle (deg)
      - end angle   (deg)
      - span        (deg)
      - mid angle   (deg)
    を推定して返す。
    """
    cx, cy, _ = detect_circle(image_path, debug=debug)
    pts = _extract_line_points(image_path)

    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    ang = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
    ang.sort()

    gaps = np.diff(np.r_[ang, ang[0] + 360.0])
    k = int(np.argmax(gaps))

    start = float(ang[(k + 1) % len(ang)])
    span = float(360.0 - gaps[k])
    end = float((start + span) % 360.0)

    # 中心角（弧の真ん中）
    mid = float((start + span * 0.5) % 360.0)

    if debug:
        print(
            f"{image_path}: "
            f"start={start:.3f}, end={end:.3f}, "
            f"span={span:.3f}, mid={mid:.3f}"
        )

    return start, end, span, mid

def span_error(img1_path: str, img2_path: str, debug=False) -> float:
    """
    円弧スパン誤差（deg）
    """
    _, _, span1, _ = detect_arc_geometry(img1_path, debug=debug)
    _, _, span2, _ = detect_arc_geometry(img2_path, debug=debug)
    return float(abs(span1 - span2))

def mid_angle_error(img1_path: str, img2_path: str) -> float:
    """
    円弧の中心角（位相）誤差（deg, 0..180）
    """
    _, _, _, mid1 = detect_arc_geometry(img1_path)
    _, _, _, mid2 = detect_arc_geometry(img2_path)
    return _circular_diff_deg(mid1, mid2)


def evaluate_arc(img1_dir, img2_dir):
    """2つの円弧画像ディレクトリを比較して、中心/半径/開始角/終了角/Chamfer を集計する。"""
    img1_dir = get_sorted_image_list(img1_dir)
    img2_dir = get_sorted_image_list(img2_dir)

    pic_count = min(len(img1_dir), len(img2_dir))
    img1_dir = img1_dir[:pic_count]
    img2_dir = img2_dir[:pic_count]

    center_errors = []
    radius_errors = []
    span_errors = []
    mid_angle_errors = []
    chamfers = []
    skipped = 0

    for image1, image2 in zip(img1_dir, img2_dir):
        try:
            c_error = center_error(image1, image2)
            r_error = radius_error(image1, image2)
            sp_error = span_error(image1, image2)
            mid_error = mid_angle_error(image1, image2)
            chamfer_dis = chamfer_distance(image1, image2)

            center_errors.append(c_error)
            radius_errors.append(r_error)
            span_errors.append(sp_error)
            mid_angle_errors.append(mid_error)
            chamfers.append(chamfer_dis)
        except Exception as e:
            print(f"[SKIP] {Path(image1).name} vs {Path(image2).name} : {e}")
            skipped += 1
            continue

        # print(f"Center error: {c_error:.4f} px")
        # print(f"Radius error: {r_error:.4f} px")
        # print(f"Start angle error: {s_error:.4f} deg")
        # print(f"End angle error: {e_error:.4f} deg")

    valid_count = pic_count - skipped
    if valid_count <= 0:
        raise RuntimeError("All pairs were skipped.")

    # --- 統計量の計算 ---
    c_mean, c_std = float(np.mean(center_errors)), float(np.std(center_errors))
    r_mean, r_std = float(np.mean(radius_errors)), float(np.std(radius_errors))
    s_mean, s_std = float(np.mean(span_errors)), float(np.std(span_errors))
    m_mean, m_std = float(np.mean(mid_angle_errors)), float(np.std(mid_angle_errors))
    ch_mean, ch_std = float(np.mean(chamfers)), float(np.std(chamfers))

    print("\n========================")
    print(f"Total pairs: {pic_count}, skipped: {skipped}, valid: {valid_count}")
    print("------------------------")
    print(f"Center Error Mean      : {c_mean:.4f}")
    print(f"Center Error Std       : {c_std:.4f}")
    print("------------------------")
    print(f"Radius Error Mean      : {r_mean:.4f}")
    print(f"Radius Error Std       : {r_std:.4f}")
    print("------------------------")
    print(f"Span Error Mean : {s_mean:.4f}")
    print(f"Span Error Std  : {s_std:.4f}")
    print("------------------------")
    print(f"Middle Angle Error Mean   : {m_mean:.4f}")
    print(f"Middle Angle Error Std    : {m_std:.4f}")
    print("------------------------")
    print(f"Chamfer Error Mean     : {ch_mean:.4f}")
    print(f"Chamfer Error Std      : {ch_std:.4f}")
    print("========================")

import numpy as np
import cv2
from pathlib import Path

def _point_on_circle(cx: float, cy: float, r: float, ang_deg: float) -> tuple[int, int]:
    """画像座標系(右=+x, 下=+y)で、角度ang_deg上の円周点を返す。"""
    th = np.deg2rad(ang_deg)
    x = cx + r * np.cos(th)
    y = cy + r * np.sin(th)  # dyをそのまま使っているので +sin でOK（y下向き）
    return int(round(x)), int(round(y))

def _sample_arc_points(cx: float, cy: float, r: float, start_deg: float, span_deg: float, n: int = 200) -> np.ndarray:
    """startからspan分の円弧上の点列を (N,1,2) int32 で返す（cv2.polylines用）。"""
    # spanが0や極端に小さい時の保険
    n = max(2, int(n))
    angs = (start_deg + np.linspace(0.0, span_deg, n)) % 360.0
    th = np.deg2rad(angs)
    xs = cx + r * np.cos(th)
    ys = cy + r * np.sin(th)
    pts = np.stack([xs, ys], axis=1)
    return np.round(pts).astype(np.int32).reshape(-1, 1, 2)

def save_arc_debug_overlay(
    image_path: str,
    out_path: str | None = None,
    *,
    debug: bool = False,
    arc_points: int = 240,
    line_thickness: int = 2,
):
    """
    円弧推定結果を可視化して保存する。
      - 円中心
      - 開始点/終了点
      - 中心→開始点、中心→終了点 の線分
      - 推定円弧（polyline）

    out_pathがNoneなら、元画像と同じフォルダに *_arcvis.png で保存。
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # 円（中心・半径）と円弧幾何を推定
    cx, cy, r = detect_circle(image_path, debug=debug)  # 既存の関数
    start, end, span, mid = detect_arc_geometry(image_path, debug=debug)

    # 描画用コピー
    vis = img.copy()

    # 主要点
    cpt = (int(round(cx)), int(round(cy)))
    spt = _point_on_circle(cx, cy, r, start)
    ept = _point_on_circle(cx, cy, r, end)
    mpt = _point_on_circle(cx, cy, r, mid)

    # 円弧（点列で描画：角度定義が確実に一致）
    arc_poly = _sample_arc_points(cx, cy, r, start, span, n=arc_points)
    cv2.polylines(vis, [arc_poly], isClosed=False, color=(0, 200, 0), thickness=line_thickness)

    # 中心→開始/終了 の線分
    cv2.line(vis, cpt, spt, color=(255, 0, 0), thickness=line_thickness)   # start: 青
    cv2.line(vis, cpt, ept, color=(0, 0, 255), thickness=line_thickness)   # end: 赤

    # 参考：中心→mid（中心角）も描くと分かりやすい（不要なら消してOK）
    cv2.line(vis, cpt, mpt, color=(0, 200, 200), thickness=1)

    # 点のマーカー
    cv2.circle(vis, cpt, 3, (0, 0, 0), -1)
    cv2.circle(vis, spt, 4, (255, 0, 0), -1)
    cv2.circle(vis, ept, 4, (0, 0, 255), -1)
    cv2.circle(vis, mpt, 3, (0, 200, 200), -1)

    # 文字情報（必要なら）
    txt = f"start={start:.1f} end={end:.1f} span={span:.1f} mid={mid:.1f}"
    cv2.putText(vis, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # 保存先
    if out_path is None:
        p = Path(image_path)
        out_path = str(p.with_name(p.stem + "_arcvis.png"))

    cv2.imwrite(out_path, vis)

    if debug:
        print(f"[saved] {out_path}")

    return out_path

if __name__ == "__main__":
    save_arc_debug_overlay("D:/2024_Satsuka/github/DiffusionModel/generated_by_cond/2026_01_10_17_42/arc/pic23.png", debug=True)
