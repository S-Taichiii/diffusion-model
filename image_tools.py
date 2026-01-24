# image_tools.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import random
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# =========================
# 共通：画像一覧取得＋自然ソート
# =========================
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_num_re = re.compile(r"(\d+)")


def _natural_key(p: Path):
    """
    ファイル名中の数字を考慮した自然ソート用キー
    例: p00002 < p00010, pic2 < pic10, t1 < t12
    """
    parts = _num_re.split(p.stem)
    key = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part.lower())
    return key


def list_images(dir_path: str | Path) -> List[Path]:
    d = Path(dir_path)
    if not d.exists():
        raise FileNotFoundError(f"Directory not found: {d}")
    if not d.is_dir():
        raise NotADirectoryError(f"Not a directory: {d}")

    files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in _IMG_EXTS]
    if not files:
        raise FileNotFoundError(f"No image files found in: {d}")

    return sorted(files, key=_natural_key)


# =========================
# ① ディレクトリ内画像のタイル表示（背景：灰色）＋保存
# =========================
def show_image_tiles(
    dir_path: str | Path,
    tile_shape: Tuple[int, int],
    is_random: bool = False,
    seed: Optional[int] = None,
    out_dir: Optional[str | Path] = None,
    filename: str = "tile.png",
    show: bool = True,
):
    """
    ディレクトリ内の画像をタイル表示・保存する（背景：灰色）

    Args:
        dir_path : 入力画像ディレクトリ
        tile_shape : (rows, cols)
        is_random : Trueならランダム抽出 / Falseなら番号順
        seed : ランダムシード（再現性用）
        out_dir : 出力先ディレクトリ（Noneなら保存しない）
        filename : 保存ファイル名
        show : Trueなら画面表示 / Falseなら保存のみ
    """
    rows, cols = tile_shape
    if rows <= 0 or cols <= 0:
        raise ValueError("tile_shape must be positive (rows, cols).")

    paths = list_images(dir_path)
    need = rows * cols

    # ---- 画像選択 ----
    if is_random:
        rng = random.Random(seed)
        chosen = paths[:]
        rng.shuffle(chosen)
        chosen = chosen[: min(need, len(chosen))]
    else:
        chosen = paths[: min(need, len(paths))]

    # ---- 描画設定（灰色背景）----
    gray = (0.7, 0.7, 0.7)  # matplotlibは0〜1 RGB
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 3, rows * 3),
        facecolor=gray,
    )
    axes = np.array(axes).reshape(rows, cols)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            ax.set_facecolor(gray)
            ax.axis("off")

            if idx < len(chosen):
                img = Image.open(chosen[idx]).convert("RGB")
                ax.imshow(img)
                # ax.set_title(chosen[idx].name, fontsize=9)
            idx += 1

    plt.tight_layout()

    # ---- 保存処理 ----
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        fig.savefig(out_path, facecolor=fig.get_facecolor(), dpi=200)
        print(f"[Saved] {out_path}")

    # ---- 表示 or クローズ ----
    if show:
        plt.show()
    else:
        plt.close(fig)


# =========================
# ② ディレクトリ内画像を動画にする（文字入れオプションあり）
# =========================
def _label_from_filename(p: Path) -> str:
    """
    ファイル名からラベル生成：
      t1.png     -> t=1
      p00003.jpg -> p=3
      pic12.png  -> pic=12
    数字が取れない場合は stem をそのまま返す。
    """
    m = re.match(r"^([A-Za-z]+)(\d+)$", p.stem)
    if m:
        prefix, num = m.group(1), int(m.group(2))
        return f"{prefix}={num}"

    m2 = re.search(r"([A-Za-z]+).*?(\d+)", p.stem)
    if m2:
        prefix, num = m2.group(1), int(m2.group(2))
        return f"{prefix}={int(num)}"

    return p.stem


def images_to_video(
    dir_path: str | Path,
    with_text: bool = False,
    out_path: Optional[str | Path] = None,
    fps: int = 30,
    fourcc: str = "mp4v",
    resize_to_first: bool = True,
    bottom_pad_px: int = 60,
    bg_color_bgr: Tuple[int, int, int] = (255, 255, 255),
):
    """
    ディレクトリ内画像を1本の動画にする。

    Args:
        dir_path : 入力画像ディレクトリ
        with_text : Trueなら画像下部に "t=1" のような文字を入れる
        out_path : 出力動画パス（Noneなら dir_path/video.mp4）
        fps : 動画FPS
        fourcc : コーデック（mp4vが無難）
        resize_to_first : Trueなら全フレームを1枚目サイズにリサイズ
        bottom_pad_px : 文字入れ時に下に足す余白ピクセル
        bg_color_bgr : 文字エリア背景色（BGR）
    """
    paths = list_images(dir_path)
    d = Path(dir_path)
    out_path = Path(out_path) if out_path else (d / "video.mp4")

    # 1枚目でサイズ決定
    first = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read image: {paths[0]}")
    h, w = first.shape[:2]

    out_h = h + (bottom_pad_px if with_text else 0)
    out_w = w

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*fourcc),
        fps,
        (out_w, out_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {out_path}")

    try:
        for p in paths:
            frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Failed to read image: {p}")

            if resize_to_first and (frame.shape[0] != h or frame.shape[1] != w):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

            if with_text:
                canvas = np.full((out_h, out_w, 3), bg_color_bgr, dtype=np.uint8)
                canvas[0:h, 0:w] = frame

                label = _label_from_filename(p)

                # 文字描画（中央寄せ気味）
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                x = max(10, (out_w - tw) // 2)
                y = h + (bottom_pad_px + th) // 2

                cv2.putText(
                    canvas,
                    label,
                    (x, y),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                    cv2.LINE_AA,
                )
                writer.write(canvas)
            else:
                writer.write(frame)
    finally:
        writer.release()

    print(f"[Saved] {out_path}")
    return out_path


# =========================
# CLI（コマンド実行したい人用）
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Image tiling & video creation tools")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # tile
    p_tile = sub.add_parser("tile", help="Show & save image tiles")
    p_tile.add_argument("dir", type=str, help="Image directory")
    p_tile.add_argument("--rows", type=int, default=3)
    p_tile.add_argument("--cols", type=int, default=3)
    p_tile.add_argument("--random", action="store_true")
    p_tile.add_argument("--seed", type=int, default=None)
    p_tile.add_argument("--out_dir", type=str, default=None)
    p_tile.add_argument("--filename", type=str, default="tile.png")
    p_tile.add_argument("--no_show", action="store_true")

    # video
    p_vid = sub.add_parser("video", help="Create video from images")
    p_vid.add_argument("dir", type=str, help="Image directory")
    p_vid.add_argument("--text", action="store_true")
    p_vid.add_argument("--out", type=str, default=None)
    p_vid.add_argument("--fps", type=int, default=30)

    args = parser.parse_args()

    if args.cmd == "tile":
        show_image_tiles(
            dir_path=args.dir,
            tile_shape=(args.rows, args.cols),
            is_random=args.random,
            seed=args.seed,
            out_dir=args.out_dir,
            filename=args.filename,
            show=(not args.no_show),
        )
    elif args.cmd == "video":
        images_to_video(
            dir_path=args.dir,
            with_text=args.text,
            out_path=args.out,
            fps=args.fps,
        )
