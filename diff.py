import torch
import matplotlib
matplotlib.use('TkAgg')
from torchvision import transforms
from tqdm import tqdm


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

    def sample(self, model, x_shape=(20, 3, 80, 80)):
        """
        Diffusion 向けのサンプラ。

        args: 
        - model: 潜在空間上のU-Net（例：in_ch=4）
        - x_shape: 生成する画像の形状 (B, C, H, W)。
        
        return:
          - PIL.Image のリスト
        """
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)

        for i in tqdm(range(self.num_timesteps, 0, -1)):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            x = self.denoise(model, x, t)
            
        images = [self.reverse_to_img(x[i]) for i in range(batch_size)]
        return images

    def sample_latent(self, model, z_shape=(1000, 4, 28, 28), vae=None, to_pil=True, progress=True):
        """
        Latent Diffusion 向けのサンプラ。

        args: 
        - model: 潜在空間上のU-Net（例：in_ch=4）
        - z_shape: 生成する潜在の形状 (B, C, H, W)。例：(16, 4, 28, 28)
        - vae: VAE インスタンス（decode(z) が [0,1] 画像Tensorを返す想定）。Noneなら潜在を返す
        - to_pil: True の場合、画像Tensorを PIL.Image のリストに変換して返す（vae が必要）
        - progress: tqdm の進捗表示
        
        return:
          - vae が None のとき: z (Tensor, shape=z_shape)
          - vae がある & to_pil=False: imgs (Tensor, shape=(B,3,H*8,W*8) 程度)
          - vae がある & to_pil=True: PIL.Image のリスト
        """
        batch_size = z_shape[0]
        x = torch.randn(z_shape, device=self.device)

        step_iter = range(self.num_timesteps, 0, -1)
        if progress:
            step_iter = tqdm(step_iter)
        
        with torch.no_grad():
            for i in step_iter:
                t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
                x = self.denoise(model, x, t) # 潜在空間で逆拡散

        # 画像に戻さず潜在を返す場合
        if vae is None:
            return x

        # 画像へデコード（VAE.decode 内で scale_factor を戻し、sigmoidで[0,1]化する想定）
        images = vae.decode(x)

        if to_pil:
            return [self.reverse_to_img(images[i]) for i in range(batch_size)]
        else:
            return images