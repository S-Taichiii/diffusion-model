# eval_iou_noise.py
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

# ---- optional: distance transform backend ----
_DT_BACKEND = None
try:
    from scipy.ndimage import distance_transform_edt  # type: ignore

    _DT_BACKEND = "scipy"
except Exception:
    distance_transform_edt = None  # type: ignore
    try:
        import cv2  # type: ignore

        _DT_BACKEND = "opencv"
    except Exception:
        cv2 = None  # type: ignore
        _DT_BACKEND = None


# =========================
# ファイル名パーサ
# =========================
P_GT = re.compile(r"^p(\d+)\.jpg$", re.IGNORECASE)      # p00000.jpg
P_GEN = re.compile(r"^pic(\d+)\.png$", re.IGNORECASE)  # pic1.png


def _extract_gt_index(name: str) -> int | None:
    m = P_GT.match(name)
    if not m:
        return None
    return int(m.group(1))  # p00000 -> 0


def _extract_gen_index(name: str) -> int | None:
    m = P_GEN.match(name)
    if not m:
        return None
    return int(m.group(1))  # pic1 -> 1


def list_gt_files(gt_dir: Path) -> List[Tuple[int, Path]]:
    files: List[Tuple[int, Path]] = []
    for p in gt_dir.iterdir():
        if p.is_file():
            idx = _extract_gt_index(p.name)
            if idx is not None:
                files.append((idx, p))
    files.sort(key=lambda x: x[0])
    return files


def list_gen_files(gen_dir: Path) -> List[Tuple[int, Path]]:
    files: List[Tuple[int, Path]] = []
    for p in gen_dir.iterdir():
        if p.is_file():
            idx = _extract_gen_index(p.name)
            if idx is not None:
                files.append((idx, p))
    files.sort(key=lambda x: x[0])
    return files


# =========================
# 画像 -> 二値マスク
# =========================
def load_binary_mask(
    image_path: Path,
    threshold: int = 128,
    invert: bool = True,
) -> np.ndarray:
    """
    画像をグレースケール化して二値化し、boolマスクを返す。
    invert=True: 黒(小さい値)を前景(True)として扱う（線画/CAD向け）。
    """
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.uint8)

    if invert:
        mask = arr < threshold   # 黒を前景(True)
    else:
        mask = arr >= threshold  # 白を前景(True)

    return mask.astype(bool)


def mask_to_pil(mask: np.ndarray) -> Image.Image:
    """boolマスクを0/255のL画像に変換"""
    arr = (mask.astype(np.uint8) * 255)
    return Image.fromarray(arr, mode="L")


def save_side_by_side(
    gt_mask: np.ndarray,
    gen_mask: np.ndarray,
    out_path: Path,
) -> None:
    """二値マスクを左右に並べた比較画像を保存（左=GT, 右=GEN）"""
    gt_img = mask_to_pil(gt_mask)
    gen_img = mask_to_pil(gen_mask)

    w, h = gt_img.size
    if gen_img.size != (w, h):
        gen_img = gen_img.resize((w, h), resample=Image.NEAREST)

    canvas = Image.new("L", (w * 2, h), color=0)
    canvas.paste(gt_img, (0, 0))
    canvas.paste(gen_img, (w, 0))
    canvas.save(out_path)


def save_diff_visual(
    gt_mask: np.ndarray,
    gen_mask: np.ndarray,
    out_path: Path,
) -> None:
    """
    差分可視化（RGB）
    - 背景: 白
    - TP(一致=GT∩GEN): 黒
    - FN(欠け=GT\\GEN): 青
    - FP(ノイズ=GEN\\GT): 赤
    """
    tp = np.logical_and(gt_mask, gen_mask)
    fn = np.logical_and(gt_mask, np.logical_not(gen_mask))
    fp = np.logical_and(gen_mask, np.logical_not(gt_mask))

    h, w = gt_mask.shape

    # 背景を白で初期化
    rgb = np.full((h, w, 3), 255, dtype=np.uint8)

    # TP: 黒
    rgb[tp, :] = 0

    # FN: 青 (R=0,G=0,B=255)
    rgb[fn, 0] = 0
    rgb[fn, 1] = 0
    rgb[fn, 2] = 255

    # FP: 赤 (R=255,G=0,B=0)
    rgb[fp, 0] = 255
    rgb[fp, 1] = 0
    rgb[fp, 2] = 0

    Image.fromarray(rgb, mode="RGB").save(out_path)


# =========================
# 距離変換 & ガウス重みRecall
# =========================
def _distance_map_to_gt(gt_mask: np.ndarray) -> np.ndarray:
    """
    GTからの距離マップ（GT上は距離0）。float64で返す。
    backend:
      - scipy.ndimage.distance_transform_edt(~gt)
      - opencv distanceTransform (cv2.DIST_L2)
    """
    if _DT_BACKEND == "scipy":
        dist = distance_transform_edt(~gt_mask)  # type: ignore
        return dist.astype(np.float64)

    if _DT_BACKEND == "opencv":
        # cv2.distanceTransform: 0が障害物(=距離0)なので、GTを0にする
        src = np.where(gt_mask, 0, 1).astype(np.uint8)
        dist = cv2.distanceTransform(src, distanceType=cv2.DIST_L2, maskSize=3)  # type: ignore
        return dist.astype(np.float64)

    raise RuntimeError(
        "距離変換が使えません。scipy か opencv-python のどちらかをインストールしてください。\n"
        "例: pip install scipy  または pip install opencv-python"
    )


def gaussian_weighted_recall(
    gt: np.ndarray,
    pred: np.ndarray,
    sigma: float = 2.0,
) -> float:
    """
    ガウス型距離重み付きRecall
      w(d)=exp(-d^2/(2*sigma^2))
      Recall_gauss = sum_x pred(x)*w(dist_to_GT(x)) / sum_x gt(x)

    直感: 「正解線の近傍に、どれだけ生成線が存在したか」
    """
    gt_area = int(gt.sum())
    if gt_area == 0:
        return 1.0

    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    dist_map = _distance_map_to_gt(gt)
    weight = np.exp(-(dist_map ** 2) / (2.0 * (sigma ** 2)))

    weighted_hit = (pred.astype(np.float64) * weight).sum(dtype=np.float64)
    return float(weighted_hit / gt_area)


def far_noise_ratio(
    gt: np.ndarray,
    pred: np.ndarray,
    sigma: float = 2.0,
) -> float:
    """
    近傍外ノイズ率（Hard threshold）
      far_noise_ratio = | {x | pred(x)=1 and dist_to_GT(x) > sigma} | / |pred|

    - 距離スケールは gaussian_weighted_recall と同じ sigma を使う
    - pred が空なら 0.0（ノイズを出していない）とする
    """
    pred_area = int(pred.sum())
    if pred_area == 0:
        return 0.0
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    dist_map = _distance_map_to_gt(gt)  # GT上=0, 離れるほど増える
    far = dist_map > sigma
    far_noise = np.logical_and(pred, far).sum(dtype=np.int64)
    return float(far_noise / pred_area)



# =========================
# 指標計算
# =========================
def compute_metrics(gt: np.ndarray, pred: np.ndarray, sigma: float = 2.0) -> Dict[str, float]:
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: gt{gt.shape} vs pred{pred.shape}")

    inter = np.logical_and(gt, pred).sum(dtype=np.int64)
    union = np.logical_or(gt, pred).sum(dtype=np.int64)

    gt_area = gt.sum(dtype=np.int64)
    pred_area = pred.sum(dtype=np.int64)

    # 通常IOU
    iou = float(inter / union) if union > 0 else 1.0  # 両方空なら一致扱い

    # 正解母数IOU (Coverage / Recall-like)
    gt_iou = float(inter / gt_area) if gt_area > 0 else 1.0

    # 近傍外ノイズ率（gauss_recall と同じ sigma を閾値として使用）
    far_noise = far_noise_ratio(gt, pred, sigma=sigma)

    # ガウス型距離重み付きRecall
    gauss_recall = gaussian_weighted_recall(gt, pred, sigma=sigma)

    return {
        "iou": iou,
        "gt_iou": gt_iou,
        "far_noise_ratio": far_noise,   # ★ ここが新しいノイズ率
        "gauss_recall": gauss_recall,
        "inter": float(inter),
        "union": float(union),
        "gt_area": float(gt_area),
        "pred_area": float(pred_area),
        # fp は（他で使う可能性があるので）残すが、far_noise率の分子では使わない
        "fp": float(np.logical_and(pred, np.logical_not(gt)).sum(dtype=np.int64)),
    }


def mean_std(x: np.ndarray) -> Tuple[float, float]:
    if x.size == 0:
        return float("nan"), float("nan")
    return float(x.mean()), float(x.std(ddof=0))


def quantiles(x: np.ndarray, ps: List[float]) -> Dict[str, float]:
    """
    ps: 0-100 のパーセンタイル値（例: [50, 90, 95]）
    """
    if x.size == 0:
        return {f"p{int(p)}": float("nan") for p in ps}
    vals = np.percentile(x, ps)
    return {f"p{int(p)}": float(v) for p, v in zip(ps, vals)}


def overdraw_rate(x: np.ndarray, threshold: float = 1.0) -> float:
    """
    gauss_recall が threshold を超える割合
    """
    if x.size == 0:
        return float("nan")
    return float((x > threshold).mean())


# =========================
# メイン
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", type=str, required=True, help="正解画像ディレクトリ (p00000.jpg...)")
    ap.add_argument("--gen_dir", type=str, required=True, help="生成画像ディレクトリ (pic1.png...)")
    ap.add_argument("--out_dir", type=str, required=True, help="出力先ディレクトリ（この下にrun_*を作る）")
    ap.add_argument("--threshold", type=int, default=128, help="二値化閾値 (0-255)")
    ap.add_argument("--invert", action="store_true", help="黒を前景(True)として扱う（線画向け）。指定すると有効。")
    ap.add_argument("--sigma", type=float, default=2.0, help="ガウス重みRecallのsigma(px)。大きいほどズレに甘い")
    ap.add_argument("--max_pairs", type=int, default=-1, help="評価する最大ペア数（-1で全て）")
    ap.add_argument("--save_diff", action="store_true", help="差分可視化(FP赤/FN青/TP白)も保存する")
    args = ap.parse_args()

    gt_dir = Path(args.gt_dir)
    gen_dir = Path(args.gen_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if not gt_dir.exists():
        raise FileNotFoundError(f"gt_dir not found: {gt_dir}")
    if not gen_dir.exists():
        raise FileNotFoundError(f"gen_dir not found: {gen_dir}")

    # 実行結果をまとめる run ディレクトリ
    run_name = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 二値化画像保存先
    bin_dir = run_dir / "binarized"
    bin_gt_dir = bin_dir / "gt"
    bin_gen_dir = bin_dir / "gen"
    bin_pair_dir = bin_dir / "pair"
    for d in (bin_gt_dir, bin_gen_dir, bin_pair_dir):
        d.mkdir(parents=True, exist_ok=True)

    diff_dir = run_dir / "diff"
    if args.save_diff:
        diff_dir.mkdir(parents=True, exist_ok=True)

    gt_list = list_gt_files(gt_dir)     # (0, p00000.jpg) ...
    gen_list = list_gen_files(gen_dir)  # (1, pic1.png) ...

    gen_map = {idx: p for idx, p in gen_list}

    # ペアリング規則: gen_idx = gt_idx + 1
    pairs: List[Tuple[int, Path, Path]] = []
    missing = 0
    for gt_idx, gt_path in gt_list:
        gen_idx = gt_idx + 1
        gen_path = gen_map.get(gen_idx, None)
        if gen_path is None:
            missing += 1
            continue
        pairs.append((gt_idx, gt_path, gen_path))

    if args.max_pairs is not None and args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    if len(pairs) == 0:
        raise RuntimeError(
            "有効な比較ペアが見つかりません。\n"
            "正解: p00000.jpg, p00001.jpg...\n"
            "生成: pic1.png, pic2.png...\n"
            "対応: p00000 <-> pic1, p00001 <-> pic2 ...\n"
        )

    rows = []
    for gt_idx, gt_path, gen_path in pairs:
        gt_mask = load_binary_mask(gt_path, threshold=args.threshold, invert=args.invert)
        gen_mask = load_binary_mask(gen_path, threshold=args.threshold, invert=args.invert)

        # 二値化画像を書き出し（PNG）
        gt_bin_path = bin_gt_dir / f"{gt_path.stem}_bin.png"
        gen_bin_path = bin_gen_dir / f"{gen_path.stem}_bin.png"
        mask_to_pil(gt_mask).save(gt_bin_path)
        mask_to_pil(gen_mask).save(gen_bin_path)

        # 左右比較（GT | GEN）
        pair_path = bin_pair_dir / f"pair_gt{gt_idx:05d}_vs_{gen_path.stem}.png"
        save_side_by_side(gt_mask, gen_mask, pair_path)

        # 差分可視化
        diff_path = None
        if args.save_diff:
            diff_path = diff_dir / f"diff_gt{gt_idx:05d}_vs_{gen_path.stem}.png"
            save_diff_visual(gt_mask, gen_mask, diff_path)

        # 指標
        m = compute_metrics(gt_mask, gen_mask, sigma=args.sigma)

        rows.append({
            "gt_index": gt_idx,
            "gt_file": gt_path.name,
            "gen_file": gen_path.name,
            "gt_bin": str(gt_bin_path.relative_to(run_dir)),
            "gen_bin": str(gen_bin_path.relative_to(run_dir)),
            "pair_bin": str(pair_path.relative_to(run_dir)),
            "diff_bin": str(diff_path.relative_to(run_dir)) if diff_path is not None else "",
            **m
        })

    df = pd.DataFrame(rows)

    # summary: 平均・標準偏差（従来）
    iou_mean, iou_std = mean_std(df["iou"].to_numpy(dtype=np.float64))
    gt_iou_mean, gt_iou_std = mean_std(df["gt_iou"].to_numpy(dtype=np.float64))

    # ★ 近傍外ノイズ率（far_noise_ratio）
    fnr = df["far_noise_ratio"].to_numpy(dtype=np.float64)
    fnr_mean, fnr_std = mean_std(fnr)
    fnr_q = quantiles(fnr, [50, 90, 95])  # median, p90, p95

    gr = df["gauss_recall"].to_numpy(dtype=np.float64)
    gr_mean, gr_std = mean_std(gr)

    # ---- gauss_recall の主指標（robust） ----
    q = quantiles(gr, [50, 90, 95])  # median, p90, p95
    od = overdraw_rate(gr, threshold=1.0)

    summary = pd.DataFrame([{
        "n_pairs": int(len(df)),
        "missing_pairs_skipped": int(missing),
        "threshold": int(args.threshold),
        "invert": bool(args.invert),
        "sigma": float(args.sigma),
        "distance_backend": _DT_BACKEND or "none",

        "iou_mean": iou_mean,
        "iou_std": iou_std,
        "gt_iou_mean": gt_iou_mean,
        "gt_iou_std": gt_iou_std,

        # ★ 近傍外ノイズ率（平均・標準偏差 + robust）
        "far_noise_ratio_mean": fnr_mean,
        "far_noise_ratio_std": fnr_std,
        "far_noise_ratio_median": fnr_q["p50"],
        "far_noise_ratio_p90": fnr_q["p90"],
        "far_noise_ratio_p95": fnr_q["p95"],

        # gauss_recall: 参考（平均・標準偏差）
        "gauss_recall_mean": gr_mean,
        "gauss_recall_std": gr_std,

        # gauss_recall: 主（robust）
        "gauss_recall_median": q["p50"],
        "gauss_recall_p90": q["p90"],
        "gauss_recall_p95": q["p95"],
        "gauss_overdraw_rate_gt1": od,

        "run_dir": str(run_dir),
    }])


    # 出力
    detail_path = run_dir / "metrics_detail.csv"
    summary_path = run_dir / "metrics_summary.csv"
    df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # 実行条件メモ
    config_path = run_dir / "config.txt"
    config_path.write_text(
        "\n".join([
            f"gt_dir={gt_dir}",
            f"gen_dir={gen_dir}",
            f"threshold={args.threshold}",
            f"invert={bool(args.invert)}",
            f"sigma={args.sigma}",
            f"distance_backend={_DT_BACKEND}",
            f"max_pairs={args.max_pairs}",
            f"save_diff={bool(args.save_diff)}",
            f"missing_pairs_skipped={missing}",
        ]) + "\n",
        encoding="utf-8"
    )

    print(f"[OK] run_dir: {run_dir}")
    print(f"[OK] detail:  {detail_path}")
    print(f"[OK] summary: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

    # python eval_iou_noise.py --gt_dir "./data/arc_224x224_test" --gen_dir "D:\2024_Satsuka\github\DiffusionModel\generated_by_cond\lambda_01\arc" --out_dir "D:\2024_Satsuka\github\DiffusionModel\eval_result\lambda_01\arc" --invert --save_diff