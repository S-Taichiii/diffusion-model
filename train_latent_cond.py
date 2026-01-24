import torch
import torch.nn.functional as F
import time, inspect, os
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from diff import Diffuser
from models.vae import VAE

# [ADD] 回帰ヘッド付きUNet & マスク付き回帰損失
from models.unet_cond_geom import UnetCondWithGeomHead
from losses.geom_losses import masked_geom_mse

# データセット（train_vae.pyで使っていたやつ）
from custom_dataset import LabelDataset

from utils import Utils

def main():
    # ---------------- Config ----------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} を使用しています")


    # 学習設定
    batch_size = 32
    epochs = 200
    lr = 1e-4
    num_timesteps = 1000
    z_ch = 4 # 潜在サイズ
    cfg_drop_prob = 0.1 # classifier-free dropout during traing

    # [ADD] 回帰ヘッド損失の重み（最初は小さめ推奨）
    geom_lambda = 0.15

    # [ADD] cond_vals の次元（あなたのデータに合わせて要調整）
    # 例：直線/円/円弧で共通ベクトルを12次元にしているなら 12
    geom_dim = 12

    # ToTensor() は [0,1] 範囲のfloat化。VAEがその前提になっているのでOK
    preprocess = transforms.ToTensor()

    # データセット
    # train
    base = r"D:/2024_Satsuka/github/DiffusionModel/data"
    arc_dir = "arc_224x224"
    line_dir = "line_224x224"
    circle_dir = "circle_224x224"
    train_items = [
        (fr"{base}\{arc_dir}\arc_224x224.csv", fr"{base}\{arc_dir}", 3),
        (fr"{base}\{line_dir}\line_224x224.csv", fr"{base}\{line_dir}", 1),
        (fr"{base}\{circle_dir}\circle_224x224.csv", fr"{base}\{circle_dir}", 2),
    ]
    train_dataset = LabelDataset(train_items, preprocess)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=False, shuffle=True)

    # val
    base = r"D:/2024_Satsuka/github/DiffusionModel/data"
    arc_dir = "arc_224x224_val"
    line_dir = "line_224x224_val"
    circle_dir = "circle_224x224_val"
    train_items = [
        (fr"{base}\{arc_dir}\arc_224x224_val.csv", fr"{base}\{arc_dir}", 3),
        (fr"{base}\{line_dir}\line_224x224_val.csv", fr"{base}\{line_dir}", 1),
        (fr"{base}\{circle_dir}\circle_224x224_val.csv", fr"{base}\{circle_dir}", 2),
    ]
    val_dataset = LabelDataset(train_items, preprocess)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=False, shuffle=False)

    os.makedirs("./model_para", exist_ok=True)

    # ---------------- Models ----------------
    # 学習済みVAEの読み込み（train_vae.pyで保存したpthを指定）
    vae_ckpt_dir = "./vae/2025_09_30_19_34"            # 例: train_vae.pyの保存先
    vae_ckpt = os.path.join(vae_ckpt_dir, "vae_best.pth")

    vae = VAE(in_channels=3, z_channels=z_ch)
    vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"))
    # dropout などを推論モードに。
    vae.eval().to(device)
    for p in vae.parameters():
        p.requires_grad = False  # 凍結。VAEに勾配を流さない（計算グラフも作られずにメモリ節約）

    # 潜在用のU-Net（入力チャネル=4）
    # [REPLACE] UnetCond -> UnetCondWithGeomHead
    # 重要：cfg_drop は「モデル内」ではなく学習ループで制御する（回帰損失との整合のため）
    model = UnetCondWithGeomHead(
        in_ch=z_ch,
        num_classes=3,
        cfg_drop_prob=0.0,    # [ADD] 内部dropは使わない
        geom_dim=geom_dim,    # [ADD]
        geom_hidden=256,      # [ADD] 好みで
    )
    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    diffuser = Diffuser(num_timesteps=num_timesteps, device=device)

    # ---------------- Train Loop ----------------
    start = time.time()
    train_losses = []
    val_losses = []
    val_interval = 5

    for epoch in range(1, epochs+1):
        model.train()
        loss_sum, cnt = 0.0, 0
        max_loss = float('inf')

        # non_blocking=True + Dataloaderのpin_memory=True: CPU -> GPU転送を非同期化して高速化
        for images, vals, mask, class_names in tqdm(train_loader):
            images = images.to(device, non_blocking=False)
            vals = vals.to(device, non_blocking=False)
            mask = mask.to(device, non_blocking=False)
            class_names = class_names.to(device).long()

            # [ADD] safety: dimsチェック（最初に1回だけでもOK）
            # if vals.shape[1] != geom_dim:
            #     raise ValueError(f"vals dim mismatch: vals={vals.shape[1]}, geom_dim={geom_dim}")

            # 画像→潜在（スケール済みzが返る）
            # VAEは固定なので計算グラフ不要.メモリ節約＆高速化
            # ---- VAE encode: micro-batch (NEW) ----
            with torch.no_grad():
                micro = 8  # 8/16/32あたりで調整
                z_list = []
                for img_mb in images.split(micro, dim=0):
                    z_mb, _ = vae.encode(img_mb)
                    z_list.append(z_mb)
                z = torch.cat(z_list, dim=0)

            # 時刻をサンプリングして潜在にノイズを付与
            t = torch.randint(1, num_timesteps+1, (len(z),), device=device)
            z_noisy, noise = diffuser.add_noise(z, t)

            # [ADD] CFG dropout を学習ループ側で作成（class + numeric）
            drop = (torch.rand(len(z), device=device) < cfg_drop_prob)   # (B,)
            y_used = torch.where(drop, torch.zeros_like(class_names), class_names)  # uncond は 0

            keep = (~drop).float().unsqueeze(1)  # (B,1)
            vals_used = vals * keep
            mask_used = mask * keep

            # [REPLACE] UNet forward: noise_pred だけでなく geom_pred も受け取る
            noise_pred, geom_pred = model(z_noisy, t, y_used, cond_vals=vals_used, cond_mask=mask_used)

            # [ADD] ノイズ損失（元のまま）
            loss_noise = F.mse_loss(noise_pred, noise)

            # [ADD] 回帰ヘッド損失（マスク付き）
            # 教師は「本来の条件vals」（※スケールは0-1正規化推奨）
            geom_mask_eff = mask * keep  # uncond(drop=True)はmask=0になるので損失に効かない
            loss_geom = masked_geom_mse(geom_pred, vals, geom_mask_eff)

            # [ADD] 合成損失
            loss = loss_noise + geom_lambda * loss_geom

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            cnt += 1

            if loss.item() < max_loss:
                Utils.saveModelParameter("./model_para",model=model)
                max_loss = loss.item()

        train_loss_avg = loss_sum / max(cnt, 1)
        train_losses.append(train_loss_avg)

        # ---------------- Val (every 5 epochs) ----------------
        if epoch % val_interval == 0:
            model.eval()
            v_sum, v_cnt = 0.0, 0

            with torch.no_grad():
                for images, vals, mask, class_names in tqdm(val_loader):
                    images = images.to(device, non_blocking=False)
                    vals = vals.to(device, non_blocking=False)
                    mask = mask.to(device, non_blocking=False)
                    class_names = class_names.to(device).long()

                    # VAE encode
                    micro = 8
                    z_list = []
                    for img_mb in images.split(micro, dim=0):
                        z_mb, _ = vae.encode(img_mb)
                        z_list.append(z_mb)
                    z = torch.cat(z_list, dim=0)

                    # valでは cfg_drop を入れる/入れないは設計次第だが、
                    # 「条件付き性能」を見たいなら drop無し（=keep=1）がおすすめ。
                    t = torch.randint(1, num_timesteps + 1, (len(z),), device=device)
                    z_noisy, noise = diffuser.add_noise(z, t)

                    y_used = class_names
                    vals_used = vals
                    mask_used = mask

                    noise_pred, geom_pred = model(z_noisy, t, y_used, cond_vals=vals_used, cond_mask=mask_used)

                    loss_noise = F.mse_loss(noise_pred, noise)
                    loss_geom = masked_geom_mse(geom_pred, vals, mask)

                    v_loss = loss_noise + geom_lambda * loss_geom
                    v_sum += v_loss.item()
                    v_cnt += 1

            val_loss_avg = v_sum / max(v_cnt, 1)
            val_losses.append(val_loss_avg)
            print(f"[Epoch {epoch:03d}] train={train_loss_avg:.6f}  val={val_loss_avg:.6f}")
        else:
            # valしないepochは NaN を入れて “epochと同じ長さ” を保つ
            val_losses.append(np.nan)
            print(f"[Epoch {epoch:03d}] train={train_loss_avg:.6f}  val=skip")


    # ---------------- time ----------------
    learning_time = time.time() - start

    # ---------------- sampling ----------------
    # [ADD] diffuser.sample_latent_cond が「noise_predのみ」を期待する場合のためのラッパ
    class _NoiseOnlyWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x, t, y, cond_vals=None, cond_mask=None):
            noise_pred, _ = self.m(x, t, y, cond_vals=cond_vals, cond_mask=cond_mask)
            return noise_pred

    images = None
    try:
        images = diffuser.sample_latent_cond(
            _NoiseOnlyWrapper(model),     # [REPLACE] ラップしたモデルを渡す
            class_counts={1:100},
            vae=vae,
            to_pil=True
        )
    except Exception as e:
        print(f"Sampling failed, continue without images: {e}")


    # ---------------- Save & Log ----------------
    Utils.recordResult(
        model=model,
        train_losses=train_losses,
        val_losses=val_losses,
        images=images,
        batch_size=batch_size,
        num_timesteps=num_timesteps,
        epochs=epochs,
        learning_rate=lr,
        device=device,
        learning_time=learning_time,
        dataset_name=f"{arc_dir}\n{line_dir}\n{circle_dir}",
        network_file=inspect.getfile(UnetCondWithGeomHead),  # [REPLACE]
    )

if __name__ == "__main__":
    main()