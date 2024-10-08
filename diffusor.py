import math
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import time
from unet import Unet

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch import nn
from tqdm import tqdm
from pathlib import Path
from PIL import Image

class LineDatasets(Dataset):
    # パスとtransformの取得
    def __init__(self, img_dir, transform=None):
        self.img_paths = self._get_img_paths(img_dir)
        self.transform = transform

    # データの取得
    def __getitem__(self, index): 
        path = self.img_paths[index]
        img= Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    # パスの取得
    def _get_img_paths(self, img_dir):
        img_dir = Path(img_dir)
        img_paths = [path for path in img_dir.iterdir() if path.suffix == ".jpg"]
        return img_paths

    # データの数を取得
    def __len__(self):
        return len(self.img_paths)

class Diffuser:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0, t):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= self.num_timesteps).all()
        t_idx = t - 1

        alpha_bar = self.alpha_bars[t_idx]
        N = alpha_bar.size(0)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)

        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise

        return x_t, noise

    def denoise(self, model, x, t):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1  # alphas[0] is for t=1
        alpha = self.alphas[t_idx]
        alpha_bar = self.alpha_bars[t_idx]
        alpha_bar_prev = self.alpha_bars[t_idx-1]

        N = alpha.size(0)
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)

        model.eval()
        with torch.no_grad():
            eps = model(x, t)
        model.train()

        noise = torch.randn_like(x, device=self.device)
        noise[t == 1] = 0  # no noise at t=1

        mu = (x - ((1-alpha) / torch.sqrt(1-alpha_bar)) * eps) / torch.sqrt(alpha)
        std = torch.sqrt((1-alpha) * (1-alpha_bar_prev) / (1-alpha_bar))
        return mu + noise * std

    def reverse_to_img(self, x):
        x = x * 255
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        x = x.cpu()
        to_pil = transforms.ToPILImage()
        return to_pil(x)

    def sample(self, model, x_shape=(20, 3, 60, 60)):
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)

        for i in tqdm(range(self.num_timesteps, 0, -1)):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            x = self.denoise(model, x, t)
            
        images = [self.reverse_to_img(x[i]) for i in range(batch_size)]
        return images

def show_images(images, rows=2, cols=10):
    fig = plt.figure(figsize=(cols, rows), facecolor='gray')
    i = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, i + 1)
            plt.imshow(images[i])
            plt.axis("off") # 縦軸、横軸を非表示にする
            i += 1

    plt.show()

def show_image(images):
    fig = plt.figure(facecolor='gray')
    plt.imshow(images[0])
    plt.axis("off")
    plt.show()


batch_size = 128
num_timesteps = 1000
epochs = 15
lr = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}を使用しています")

preprocess = transforms.ToTensor()
dataset = LineDatasets("line_data_60_60", preprocess)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# diffuser = Diffuser(num_timesteps, device=device)
# # サイズを確認
# for x in dataloader:
#     plt.imshow(diffuser.reverse_to_img(x[2]))
#     plt.show()
#     print(x.shape)
#     break

torch.cuda.empty_cache()

diffuser = Diffuser(num_timesteps, device=device)
model = Unet()
model.to(device)
optimizer = Adam(model.parameters(), lr=lr)

start_time = time.time()
losses = []
for epoch in range(epochs):
    loss_sum = 0.0
    cnt = 0

    # generate samples every epoch ===================
    images = diffuser.sample(model)
    show_images(images)
    # ================================================

    for images in tqdm(dataloader):
        optimizer.zero_grad()
        x = images.to(device)
        t = torch.randint(1, num_timesteps+1, (len(x),), device=device)

        x_noisy, noise = diffuser.add_noise(x, t)
        noise_pred = model(x_noisy, t)
        loss = F.mse_loss(noise, noise_pred)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        cnt += 1

    loss_avg = loss_sum / cnt
    losses.append(loss_avg)
    print(f'Epoch {epoch} | Loss: {loss_avg}')

# lerning time 
print(f"learning time is {time.time() - start_time} (s)")

# lossのグラフ
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# 画像を生成
images = diffuser.sample(model, x_shape=(20, 3, 60, 60))
show_images(images)