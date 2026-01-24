"""
分類器なし条件付き拡散モデルの画像生成プログラム
"""
import os, datetime
import torch

from models.unet_cond import UnetCond
from models.unet_cond_geom import UnetCondWithGeomHead
from models.vae import VAE
from utils import Utils
from diff import Diffuser

from entityCsvSampler import EntityCsvSampler

"""
generated_by_cond/に実行時の時間でディレクトリを作成
その中にline, circle, arcの生成画像を格納
"""
out_dir = "./generated_by_cond"
now = datetime.datetime.now()
dir_name: str = now.strftime(now.strftime("%Y_%m_%d_%H_%M"))
out_dir: str = os.path.join(out_dir, dir_name)
os.makedirs(out_dir, exist_ok=True)

line_dir = os.path.join(out_dir, "line")
os.makedirs(line_dir, exist_ok=True)
circle_dir = os.path.join(out_dir, "circle")
os.makedirs(circle_dir, exist_ok=True)
arc_dir = os.path.join(out_dir, "arc")
os.makedirs(arc_dir, exist_ok=True)

print("directoryを作成しました")

# 条件付きlatent diffusionの学習済みパラメーターのパス
unet_cond_ckpt = "./result/2026_01_24_15_06/model_para/trained_para.pth"
# VAEの学習済みパラメーターのパス
vae_ckpt = "./vae/2025_09_30_19_34/vae_best.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

unet = Utils.loadModel(unet_cond_ckpt, UnetCondWithGeomHead(), device=device)
vae = Utils.loadModel(vae_ckpt, VAE(), device=device)

num_timesteps = 1000
diffuser = Diffuser(num_timesteps=num_timesteps, device=device)

# 生成する画像の枚数（各エンティティこの数生成）
image_count = 100

sampler = EntityCsvSampler(
    diffuser=diffuser,
    model=unet,
    vae=vae,
    class_id=1,                 # 1=line, 2=circle, 3=arc
    # base_wh=(400, 280),      # 必要なら固定（未指定ならCSVから推定→スナップ）
)

print(f"Making {image_count} line images")
line_images = sampler.sample("./data/line_224x224_test/line_224x224_test.csv", count=image_count)

# # 円を生成したい時
print(f"Making {image_count} circle images")
sampler.set_class_id(2)
circle_images = sampler.sample("./data/circle_224x224_test/circle_224x224_test.csv", count=image_count)

# # 弧を生成したい時
print(f"Making {image_count} arc images")
sampler.set_class_id(3)
arc_images = sampler.sample("./data/arc_224x224_test/arc_224x224_test.csv", count=image_count)


Utils.saveImages(line_dir, line_images)
Utils.saveImages(circle_dir, circle_images)
Utils.saveImages(arc_dir, arc_images)
