# generate_steps_from_csv.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import torch
from tqdm import tqdm

from diff import Diffuser
from entityCsvSampler import EntityCsvSampler
from models.unet_cond_geom import UnetCondWithGeomHead
from models.vae import VAE
from utils import Utils

import numpy as np
from PIL import Image

# =========================================================
# 回帰ヘッド付きモデル対策：noise_predだけ返すラッパ
# =========================================================
class NoiseOnlyWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x, t, y, cond_vals=None, cond_mask=None):
        out = self.model(x, t, y, cond_vals=cond_vals, cond_mask=cond_mask)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out



def save_latent_channels_by_dir(
    z: torch.Tensor,   # (1,C,H,W)
    step: int,
    latent_root: str,
):
    """
    latent/
      ├ ch00/t1000.png
      ├ ch01/t1000.png
      ...
    """
    z = z[0].detach().cpu()  # (C,H,W)

    C = z.shape[0]
    for c in range(C):
        ch_dir = os.path.join(latent_root, f"ch{c:02d}")
        os.makedirs(ch_dir, exist_ok=True)

        ch = z[c]

        # 可視化用 min-max 正規化
        vmin = ch.min()
        vmax = ch.max()
        if vmax > vmin:
            ch_norm = (ch - vmin) / (vmax - vmin)
        else:
            ch_norm = torch.zeros_like(ch)

        img = (ch_norm.numpy() * 255).astype(np.uint8)
        pil = Image.fromarray(img, mode="L")
        pil.save(os.path.join(ch_dir, f"t{step}.png"))


# =========================================================
# CSVの n 行目の条件だけで、逆拡散ステップ画像を保存
# =========================================================
@torch.no_grad()
def save_reverse_steps_for_csv_row(
    *,
    csv_path: str,
    row_index: int,                    # 「n番目の行」（0始まり）
    class_id: int,                     # 1=line, 2=circle, 3=arc
    model: torch.nn.Module,
    vae: torch.nn.Module,
    device: str = "cuda",
    num_timesteps: int = 1000,
    z_shape: tuple[int, int, int, int] = (1, 4, 28, 28),  # (B,C,H,W)  B=1
    guidance_scale: float = 3.0,
    null_label: int = 0,
    save_steps: Optional[Sequence[int]] = None,  # 例: [1000,900,...,1]
    save_every: Optional[int] = None,            # 例: 50 -> 1000,950,...,50,1
    run_name: Optional[str] = None,              # step_images配下のディレクトリ名
    out_root: str = "./step_images",
    base_wh: tuple[float, float] = (400, 400),   # EntityCsvSamplerに準拠（図面サイズ）
    progress: bool = True,
) -> str:
    """
    - EntityCsvSampler と同じ方法で CSV を読み、cond_vals/cond_mask を構築（正規化・Y反転含む）
    - row_index の 1行だけ取り出して逆拡散
    - step_images/<run_name>/t{step}.png で保存
    """
    device_t = torch.device(device)
    B = z_shape[0]
    if B != 1:
        raise ValueError("このスクリプトは 'n番目の行だけ' 用なので z_shape[0] は 1 を推奨します。")

    # 出力先ディレクトリ
    if run_name is None:
        entity = ['line', 'circle', 'arc']
        run_name = f"class_{entity[int(class_id) - 1]}_row{int(row_index):05d}"
    out_dir = os.path.join(out_root, run_name)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    pixel_dir = os.path.join(out_dir, "pixel")
    latent_dir = os.path.join(out_dir, "latent")
    Path(pixel_dir).mkdir(parents=True, exist_ok=True)
    Path(latent_dir).mkdir(parents=True, exist_ok=True)


    # Diffuser（あなたのDiffuserに合わせて device 渡し）
    diffuser = Diffuser(num_timesteps=num_timesteps, device=device_t)

    # sampler：CSVの読み方・cond/mask作りはこれに完全準拠させる
    sampler = EntityCsvSampler(
        diffuser=diffuser,
        model=model,   # ここは使わない（load_condだけ使う）が、構造上渡す
        vae=vae,
        class_id=class_id,
        base_wh=base_wh,
        device=device_t,
    )

    # CSVの row_index 行「だけ」を取得（count=1, start=row_index）
    # -> vals, mask の shape は (1,K)
    vals, mask = sampler.load_cond(csv_path, count=1, start=row_index)

    # y は class_id を 1つだけ
    y = torch.tensor([int(class_id)], device=device_t, dtype=torch.long)

    # モデルは noise_pred のみ返す形に統一
    model_noise = model.to(device_t)
    model_noise.eval()
    vae.eval()

    # 初期ノイズ z_T
    x = torch.randn(z_shape, device=device_t)

    # 保存ステップ集合
    if save_steps is not None:
        save_set = set(int(s) for s in save_steps)
    elif save_every is not None:
        step = max(int(save_every), 1)
        save_set = set(range(num_timesteps, 0, -step))
        save_set.add(1)
    else:
        # デフォルト：全ステップ保存（重い）
        save_set = set(range(1, num_timesteps + 1))

    step_iter = range(num_timesteps, 0, -1)
    if progress:
        step_iter = tqdm(step_iter, desc=f"Reverse diffusion (row={row_index})")

    for i in step_iter:
        # ==================================
        # 保存（denoise 前 = x_t）
        # ==================================
        if i in save_set:
            # ---------- pixel ----------
            img = vae.decode(x)
            img = img.clamp(0, 1)
            pil = diffuser.reverse_to_img(img[0])
            pil.save(os.path.join(pixel_dir, f"t{i}.png"))

            # ---------- latent ----------
            save_latent_channels_by_dir(
                z=x,
                step=i,
                latent_root=latent_dir,
            )

        # ==================================
        # 逆拡散 1 step
        # ==================================
        t = torch.full((B,), i, device=device_t, dtype=torch.long)
        x = diffuser.denoise_cond(
            model_noise,
            x,
            t,
            y=y,
            guidance_scale=guidance_scale,
            null_label=null_label,
            cond_vals=vals,
            cond_mask=mask,
        )

    return out_dir


# =========================================================
# 実行例（ここだけあなたの環境に合わせて埋める）
# =========================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===================== ここを設定 =====================
    csv_path = "./data/line_224x224_test/line_224x224_test.csv"
    row_index = 52          # ← n番目（0始まり）
    class_id = 1           # 1=line, 2=circle, 3=arc

    num_timesteps = 1000
    z_shape = (1, 4, 28, 28)  # あなたのVAE潜在に合わせる

    guidance_scale = 3.0
    save_every = 1           # 1なら全ステップ、50なら間引き、など
    # save_steps = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 1]  # こういう指定も可
    save_steps = None

    # run_name を省略すると class{class_id}_row{row_index} が自動で付く
    run_name = None

    # 図面サイズ（EntityCsvSamplerの base_wh と同じ意味）
    base_wh = (400, 400)


    # 条件付きlatent diffusionの学習済みパラメーターのパス
    unet_cond_ckpt = "D:/2024_Satsuka/github/DiffusionModel/result/2026_01_24_15_06/model_para/trained_para.pth"
    # VAEの学習済みパラメーターのパス
    vae_ckpt = "./vae/2025_09_30_19_34/vae_best.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    unet = Utils.loadModel(unet_cond_ckpt, UnetCondWithGeomHead(), device=device)
    vae = Utils.loadModel(vae_ckpt, VAE(), device=device)
    # ======================================================

    out_dir = save_reverse_steps_for_csv_row(
        csv_path=csv_path,
        row_index=row_index,
        class_id=class_id,
        model=unet,
        vae=vae,
        device=device,
        num_timesteps=num_timesteps,
        z_shape=z_shape,
        guidance_scale=guidance_scale,
        save_steps=save_steps,
        save_every=save_every if save_steps is None else None,
        run_name=run_name,
        out_root="./step_images/lambda01",
        base_wh=base_wh,
        progress=True,
    )
    print(f"[DONE] saved -> {out_dir}")
