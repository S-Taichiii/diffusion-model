import torch
import torch.nn.functional as F
import time, inspect, os
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm

from diff import Diffuser
from models.unet import Unet
from models.vae import VAE

# データセット（train_vae.pyで使っていたやつ）
from custom_dataset import ClipDataset

from utils import Utils

# ---------------- Config ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device} を使用しています")

# 画像サイズと潜在サイズ
z_ch = 4

# 学習設定
batch_size = 128
epochs = 100
lr = 1e-4
num_timesteps = 1000

# データセット（train_vae.pyと同じitemsを再利用）
base = r"D:/2024_Satsuka/github/DiffusionModel/data"
arc_dir = "arc_224x224"
line_dir = "line_224x224"
circle_dir = "circle_224x224"
items = [
    (fr"{base}\{arc_dir}\arc_224x224_caption.csv", fr"{base}\{arc_dir}", 0),
    (fr"{base}\{line_dir}\line_224x224_caption.csv", fr"{base}\{line_dir}", 1),
    (fr"{base}\{circle_dir}\circle_224x224_caption.csv", fr"{base}\{circle_dir}", 2),
]

# ToTensor() は [0,1] 範囲のfloat化。VAEがその前提になっているのでOK
preprocess = transforms.ToTensor()

dataset = ClipDataset(items, preprocess)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
model = Unet(in_ch=z_ch)  # 既存Unetをin_ch=4でインスタンス化
model.to(device)

optimizer = Adam(model.parameters(), lr=lr)
diffuser = Diffuser(num_timesteps=num_timesteps, device=device)

# ---------------- Train Loop ----------------
start = time.time()
losses = []

for epoch in range(1, epochs+1):
    model.train()
    loss_sum, cnt = 0.0, 0

    # non_blocking=True + Dataloaderのpin_memory=True: CPU -> GPU転送を非同期化して高速化
    for images, _ in tqdm(dataloader):
        images = images.to(device, non_blocking=True)

        # 画像→潜在（スケール済みzが返る）
        # VAEは固定なので計算グラフ不要.メモリ節約＆高速化
        with torch.no_grad():
            z, _kl = vae.encode(images)   # z: (B, 4, H/8, W/8)

        # 時刻をサンプリングして潜在にノイズを付与
        t = torch.randint(1, num_timesteps+1, (len(z),), device=device)
        z_noisy, noise = diffuser.add_noise(z, t)

        # U-Netは潜在上のノイズを予測
        noise_pred = model(z_noisy, t)

        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        cnt += 1

    loss_avg = loss_sum / max(cnt, 1)
    losses.append(loss_avg)
    print(f"[Epoch {epoch:03d}] loss={loss_avg:.6f}")

# ---------------- time ----------------
learning_time = time.time() - start

# ---------------- sampling ----------------
images = None
try: 
    images = diffuser.sample_latent(model, vae=vae, to_pil=True)
except Exception as e:
    print(f"Sampling failed, continue without images: {e}")

# ---------------- Save & Log ----------------
Utils.recordResult(
    model=model,
    losses=losses,
    images=images,
    batch_size=batch_size,
    num_timesteps=num_timesteps,
    epochs=epochs,
    learning_rate=lr,
    device=device,
    learning_time=learning_time,
    dataset_name=f"{arc_dir}\n{line_dir}\n{circle_dir}",
    network_file=inspect.getfile(Unet),
)
