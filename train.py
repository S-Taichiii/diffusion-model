import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
import time
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from diff import Datasets, Diffuser
from unet import Unet
from utils import Utils

batch_size = 128
num_timesteps = 1000
epochs = 200
lr = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}を使用しています")

preprocess = transforms.ToTensor()
dataset = Datasets("data/circle_32x32", preprocess)
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
    # images = diffuser.sample(model)
    # show_images(images)
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
learning_time = time.time() - start_time

# 画像を生成
images = diffuser.sample(model, x_shape=(50, 3, 32, 32))

Utils.recordResult(
    model=model,
    losses=losses,
    images=images,
    batch_size=batch_size,
    num_timesteps=num_timesteps,
    epochs=epochs,
    learning_rate=lr,
    device=device,
    learning_time=learning_time
)