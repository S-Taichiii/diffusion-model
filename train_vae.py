import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
import time, os, math, datetime
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from models.vae import VAE
from utils import Utils
from custom_dataset import ClipDataset
from early_stopping import EarlyStopping

# ------------------ 学習用関数 ------------------
def train_one_epoch():
    vae.train()
    total_loss = 0.0
    cnt = 0
    for images, _ in tqdm(train_dataloader):
        optimizer_vae.zero_grad()
        x = images.to(device)
        _, _, loss, _ = vae(x)

        loss.backward()
        optimizer_vae.step()

        total_loss += loss.item()
        cnt += 1
    
    loss_avg = total_loss / cnt
    return loss_avg

@torch.no_grad()
def validate():
    vae.eval()
    total_loss = 0.0
    cnt = 0
    for images, _ in tqdm(val_dataloader):
        optimizer_vae.zero_grad()
        x = images.to(device)
        _, _, loss, _ = vae(x)
        total_loss += loss.item()
        cnt += 1
    loss_avg = total_loss / cnt
    return loss_avg


# ------------------ config ------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}を使用しています")

# parameter
vae_epochs = 100
vae_lr = 1e-3
vae_batch_size = 64

# traning data
base = r"D:/2024_Satsuka/github/DiffusionModel/data"
arc_dir = "arc_224x224_10000"
line_dir = "line_224x224_10000"
circle_dir = "circle_224x224_10000"
items = [
    (fr"{base}\{arc_dir}\arc_224x224_10000_caption.csv", fr"{base}\{arc_dir}", 0),
    (fr"{base}\{line_dir}\line_224x224_10000_caption.csv", fr"{base}\{line_dir}", 1),
    (fr"{base}\{circle_dir}\circle_224x224_10000_caption.csv", fr"{base}\{circle_dir}", 2),
]

preprocess = transforms.ToTensor()
dataset = ClipDataset(items, preprocess)
train_dataloader = DataLoader(dataset, batch_size=vae_batch_size, shuffle=True)

# validation data
val_arc_dir = "arc_224x224_val"
val_line_dir = "line_224x224_val"
val_circle_dir = "circle_224x224_val"
val_items = [
    (fr"{base}\{val_arc_dir}\arc_224x224_val_caption.csv", fr"{base}\{val_arc_dir}", 0),
    (fr"{base}\{val_line_dir}\line_224x224_val_caption.csv", fr"{base}\{val_line_dir}", 1),
    (fr"{base}\{val_circle_dir}\circle_224x224_val_caption.csv", fr"{base}\{val_circle_dir}", 2),
]

val_dataset = ClipDataset(val_items, preprocess)
val_dataloader = DataLoader(val_dataset, batch_size=vae_batch_size, shuffle=False)

# ------------------ model ------------------
vae = VAE()
vae.to(device)
optimizer_vae = Adam(vae.parameters(), lr=vae_lr)

# ------------------ early stopping ------------------
patience = 8
min_delta = 5e-7

now = datetime.datetime.now()
dir_name: str = now.strftime(now.strftime("%Y_%m_%d_%H_%M"))
save_dir = "./vae"
dir_path: str = os.path.join(save_dir, dir_name)
os.makedirs(dir_path, exist_ok=True)
ckpt_path = os.path.join(dir_path, "vae_best.pth")

early_stopping = EarlyStopping(patience=patience, verbose=True, delta=min_delta, path = ckpt_path)

# ------------------ loop ------------------
start_time = time.time()
best_val = math.inf
loss_history = {"train": [], "val":[]}

for epoch in range(1, vae_epochs+1):
    train_loss = train_one_epoch()
    val_loss = validate()
    loss_history["train"].append(train_loss)
    loss_history["val"].append(val_loss)

    print(f"[Epoch {epoch:03d}] train={train_loss:.6f}  val={val_loss:.6f}  (best={min(best_val, val_loss):.6f})")

    early_stopping(val_loss, vae)
    if val_loss < best_val:
        best_val = val_loss
    if early_stopping.early_stop:
        print(">>> Early stopping triggered.")
        break

# lerning time 
elapsed = time.time() - start_time
print(f"Training finished in {elapsed/60:.1f} min")

# losses to graph
Utils.saveTrainValLossGraph(dir_path, loss_history["train"], loss_history["val"])
Utils.saveTrainValLossCsv(dir_path,   loss_history["train"], loss_history["val"])




