import os, math, csv
import torch
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from models.vae import VAE
from utils import Datasets

@torch.no_grad()
def recon_check(vae, dataloader, device="cuda", out_dir="./vae_recon", max_batches=5):
    os.makedirs(out_dir, exist_ok=True)
    vae.eval()

    def psnr_from_mse(m):
        # 画像が [0,1] のとき、MAX_I=1 → PSNR = 10*log10(1/m)
        return 10.0 * math.log10(1.0 / max(m, 1e-12))

    all_metrics = []
    n_imgs_saved = 0

    for b_idx, x in enumerate(dataloader):
        if b_idx >= max_batches: break
        x = x.to(device)                    # [B,3,H,W], [0,1]
        z_enc, _ = vae.encode(x)            # 外部z（scale_factor適用済）
        x_rec   = vae.decode(z_enc)         # [0,1]

        # 数値評価（バッチ内各画像ごと）
        # reduction='none' で [B,3,H,W] → 画像次元で平均して [B]
        mse = torch.mean((x_rec - x) ** 2, dim=(1,2,3)).cpu().tolist()
        mae = torch.mean(torch.abs(x_rec - x), dim=(1,2,3)).cpu().tolist()
        psnr = [psnr_from_mse(m) for m in mse]

        # 画像の値域確認（飽和チェック）
        rng_min = float(x_rec.min().item())
        rng_max = float(x_rec.max().item())
        print(f"[batch {b_idx}] recon range: min={rng_min:.4f}, max={rng_max:.4f}")

        # 保存（元→再構成の順で並べたグリッド）
        # 先に0-1クリップしてから保存
        x_vis    = torch.clamp(x, 0, 1)
        xrec_vis = torch.clamp(x_rec, 0, 1)
        pair = torch.cat([x_vis, xrec_vis], dim=0)           # 2B 枚
        grid = make_grid(pair, nrow=x.size(0), padding=2)    # 1段目:元, 2段目:再構成
        save_image(grid, os.path.join(out_dir, f"recon_grid_b{b_idx:03d}.png"))

        # 個別保存（任意）
        for i in range(min(x.size(0), 8)):
            save_image(x_vis[i],    os.path.join(out_dir, f"orig_b{b_idx:03d}_{i:02d}.png"))
            save_image(xrec_vis[i], os.path.join(out_dir, f"recon_b{b_idx:03d}_{i:02d}.png"))
            n_imgs_saved += 1

        # メトリクスを集約
        for m, a, p in zip(mse, mae, psnr):
            all_metrics.append({"mse": m, "mae": a, "psnr": p})

    # CSVに保存
    csv_path = os.path.join(out_dir, "recon_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mse", "mae", "psnr"])
        w.writeheader()
        w.writerows(all_metrics)

    # サマリを表示
    import statistics as stats
    mses  = [d["mse"] for d in all_metrics]
    maes  = [d["mae"] for d in all_metrics]
    psnrs = [d["psnr"] for d in all_metrics]
    print(f"[Summary] N={len(all_metrics)} images")
    print(f"  MSE  mean={stats.mean(mses):.6f}  median={stats.median(mses):.6f}")
    print(f"  MAE  mean={stats.mean(maes):.6f}  median={stats.median(maes):.6f}")
    print(f"  PSNR mean={stats.mean(psnrs):.3f} dB  median={stats.median(psnrs):.3f} dB")
    print(f"Saved {n_imgs_saved} images to: {out_dir}")

# 例: 使い方（あなたの前処理/データと整合）
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_name = "data/line_224x224"
preprocess = transforms.ToTensor()              # VAEの出力が [0,1] なので入力も [0,1]
dataset = Datasets(dataset_name, preprocess)
dloader = DataLoader(dataset, batch_size=32, shuffle=False)

vae = VAE().to(device)
vae.load_state_dict(torch.load("./vae/2025_09_30_19_34/vae_best.pth", map_location=device))  # もし学習済みがあるなら

recon_check(vae, dloader, device=device, out_dir="./vae_recon_line", max_batches=3)
