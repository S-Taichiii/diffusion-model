import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
import time
import inspect
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from diff import Diffuser
from models.unet import Unet
from models.vae import VAE
from utils import Utils, Datasets

# ---------- Config ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}を使用しています")

# Diffusion
network_file = inspect.getfile(Unet)
batch_size = 128
num_timesteps = 1000
unet_epochs = 200
unet_lr = 1e-3

# Data
dataset_name = "data/circle_56x56" 
preprocess = transforms.ToTensor()
dataset = Datasets(dataset_name, preprocess)
# vae_dataloader = DataLoader(dataset, batch_size=vae_batch_size, shuffle=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# diffuser = Diffuser(num_timesteps, device=device)
# # サイズを確認
# for x in dataloader:
#     plt.imshow(diffuser.reverse_to_img(x[2]))
#     plt.show()
#     print(x.shape)
#     break

torch.cuda.empty_cache()

# ------------------ Diffusion ------------------
diffuser = Diffuser(num_timesteps, device=device)
unet = Unet(remove_deep_conv=True)
unet.to(device)
optimizer_unet = Adam(unet.parameters(), lr=unet_lr)

start_time = time.time()
unet_losses = []
for epoch in range(unet_epochs):
    loss_sum = 0.0
    cnt = 0

    # generate samples every epoch ===================
    # images = diffuser.sample(unet)
    # show_images(images)
    # ================================================

    for images in tqdm(dataloader):
        optimizer_unet.zero_grad()
        x = images.to(device)
        t = torch.randint(1, num_timesteps+1, (len(x),), device=device)

        x_noisy, noise = diffuser.add_noise(x, t)
        noise_pred = unet(x_noisy, t)
        loss = F.mse_loss(noise, noise_pred)

        loss.backward()
        optimizer_unet.step()

        loss_sum += loss.item()
        cnt += 1

    loss_avg = loss_sum / cnt
    unet_losses.append(loss_avg)
    print(f'Epoch {epoch} | Loss: {loss_avg}')

# lerning time 
diffusion_learning_time = time.time() - start_time

# 画像を生成
images = diffuser.sample(unet, x_shape=(200, 3, 56, 56))

Utils.recordResult(
    model=unet,
    losses=unet_losses,
    images=images,
    batch_size=batch_size,
    num_timesteps=num_timesteps,
    epochs=unet_epochs,
    learning_rate=unet_lr,
    device=device,
    learning_time=diffusion_learning_time,
    dataset_name=dataset_name,
    network_file=network_file
)