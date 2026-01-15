# generate_steps.py
import os
from pathlib import Path
from typing import Optional, Sequence, Dict, Any

import torch
from tqdm import tqdm

from diff import Diffuser


# =========================================================
# 回帰ヘッド付きモデル対策：noise_predだけ返すラッパ
# =========================================================
class NoiseOnlyWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t, y, cond_vals=None, cond_mask=None):
        out = self.model(x, t, y, cond_vals=cond_vals, cond_mask=cond_mask)
        # out が (noise_pred, geom_pred) の場合
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


# =========================================================
# 逆拡散ステップ画像保存
# =========================================================
@torch.no_grad()
def generate_reverse_steps(
    model: torch.nn.Module,
    vae: torch.nn.Module,
    *,
    device: str,
    out_dir: str,
    num_timesteps: int = 1000,
    z_shape: tuple[int, int, int, int] = (1, 4, 28, 28),  # (B,C,H,W)
    y: Optional[torch.Tensor] = None,                     # (B,)
    cond_vals: Optional[torch.Tensor] = None,             # (B,K)
    cond_mask: Optional[torch.Tensor] = None,             # (B,K)
    guidance_scale: float = 3.0,
    null_label: int = 0,
    save_steps: Optional[Sequence[int]] = None,           # 例: [1000, 900, ..., 1]
    save_every: Optional[int] = 50,                       # 例: 50なら 1000,950,...,50,1
    decode_clamp_01: bool = True,
    progress: bool = True,
) -> None:
    """
    逆拡散の途中経過を画像として保存する。

    - save_steps を指定した場合：そのステップだけ保存
    - save_steps=None の場合：save_every 間隔で保存（最後に t=1 も保存）
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Diffuser
    diffuser = Diffuser(num_timesteps=num_timesteps, device=device)

    # モデルは noise_pred が返る形に統一
    model_noise = NoiseOnlyWrapper(model).to(device)
    model_noise.eval()
    vae.eval()

    # 初期ノイズ z_T
    x = torch.randn(z_shape, device=device)

    # y が None のときは uncond
    if y is None:
        y = torch.full((z_shape[0],), null_label, device=device, dtype=torch.long)
    else:
        y = y.to(device).long()

    if cond_vals is not None:
        cond_vals = cond_vals.to(device)
    if cond_mask is not None:
        cond_mask = cond_mask.to(device)

    # 保存ステップの決定
    if save_steps is not None:
        save_set = set(int(s) for s in save_steps)
    else:
        # 例: save_every=50 -> 1000,950,...,50 と最後に1を保証
        save_set = set(range(num_timesteps, 0, -max(int(save_every), 1)))
        save_set.add(1)

    step_iter = range(num_timesteps, 0, -1)
    if progress:
        step_iter = tqdm(step_iter, desc="Reverse diffusion")

    for i in step_iter:
        t = torch.full((z_shape[0],), i, device=device, dtype=torch.long)

        # 1ステップ逆拡散（条件付き+CFG）
        x = diffuser.denoise_cond(
            model_noise,
            x,
            t,
            y=y,
            guidance_scale=guidance_scale,
            null_label=null_label,
            cond_vals=cond_vals,
            cond_mask=cond_mask,
        )

        # 指定ステップならデコードして保存
        if i in save_set:
            img = vae.decode(x)  # (B,3,H,W) を想定（あなたのVAE実装に準拠）

            if decode_clamp_01:
                img = img.clamp(0, 1)

            # 保存（バッチ対応）
            for b in range(img.size(0)):
                pil = diffuser.reverse_to_img(img[b])  # [0,1] -> PIL
                pil.save(os.path.join(out_dir, f"step_{i:04d}_b{b:02d}.png"))

    print(f"[DONE] saved steps -> {out_dir}")


# =========================================================
# 使い方例（ここを書き換えて使う）
# =========================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------------------------------
    # ここはあなたの環境に合わせてロード
    # model: UnetCond または UnetCondWithGeomHead
    # vae  : 学習済みVAE
    # -----------------------------------------------------
    # 例（あなたの train_latent_cond.py のロード方式に合わせて書き換え）
    #
    # from models.unet_cond import UnetCond
    # from models.unet_cond_geom import UnetCondWithGeomHead
    # from models.vae import VAE
    #
    # model = UnetCondWithGeomHead(...).to(device)
    # model.load_state_dict(torch.load("path/to/unet.pth", map_location="cpu"))
    #
    # vae = VAE(...).to(device)
    # vae.load_state_dict(torch.load("path/to/vae.pth", map_location="cpu"))
    #
    # -----------------------------------------------------

    raise SystemExit("モデルとVAEのロード部分をあなたのパスに合わせて設定してください。")
