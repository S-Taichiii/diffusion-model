
import torch
from torch import nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    A small convolutional VAE that maps images (B, 3, H, W) to spatial latents (B, z_channels, H//4, W//4)
    and back. Designed to be lightweight for 32x32~256x256 images.
    """
    def __init__(self, in_channels=3, z_channels=4, base_channels=64, scale_factor=0.18215):
        super().__init__()
        self.z_channels = z_channels
        self.scale_factor = scale_factor

        # Encoder: downsample by 4x (two strided convs)
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=1, padding=1),
            nn.GroupNorm(8, base_channels), nn.GELU(),
            nn.Conv2d(base_channels, base_channels, 4, stride=2, padding=1), # /2
            nn.GroupNorm(8, base_channels), nn.GELU(),
            nn.Conv2d(base_channels, base_channels*2, 3, stride=1, padding=1),
            nn.GroupNorm(8, base_channels*2), nn.GELU(),
            nn.Conv2d(base_channels*2, base_channels*2, 4, stride=2, padding=1), # /4
            nn.GroupNorm(8, base_channels*2), nn.GELU(),
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=1, padding=1),
            nn.GroupNorm(8, base_channels*4), nn.GELU(),
            nn.Conv2d(base_channels*4, base_channels*4, 4, stride=2, padding=1), # /8
            nn.GroupNorm(8, base_channels*4), nn.GELU(),
        )
        self.to_mu = nn.Conv2d(base_channels*4, z_channels, 1)
        self.to_logvar = nn.Conv2d(base_channels*4, z_channels, 1)

        # Decoder: upsample by 4x (two conv-transpose)
        self.dec = nn.Sequential(
            nn.Conv2d(z_channels, base_channels*4, 3, padding=1),
            nn.GroupNorm(8, base_channels*4), nn.GELU(),
            nn.ConvTranspose2d(base_channels*4, base_channels*4, 4, stride=2, padding=1), # x2
            nn.GroupNorm(8, base_channels*4), nn.GELU(),
            nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1),
            nn.GroupNorm(8, base_channels*2), nn.GELU(),
            nn.ConvTranspose2d(base_channels*2, base_channels*2, 4, stride=2, padding=1), # x4
            nn.GroupNorm(8, base_channels*2), nn.GELU(),
            nn.Conv2d(base_channels*2, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels), nn.GELU(),
            nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1), # x8
            nn.GroupNorm(8, base_channels), nn.GELU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )

    def encode(self, x):
        h = self.enc(x)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h).clamp(-30.0, 20.0)  # numerical stability
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        # scale like Stable Diffusion
        z = z * self.scale_factor
        # KL divergence per element
        kl = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=(1,2,3)) / (x.size(2)*x.size(3))
        return z, kl.mean()

    def decode(self, z):
        # invert scaling
        z = z / self.scale_factor
        x_recon = self.dec(z)
        # output in [0,1] via sigmoid
        return torch.sigmoid(x_recon)

    def forward(self, x):
        z, kl = self.encode(x)
        x_recon = self.decode(z)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        loss = recon_loss + 1e-6 * kl
        return x_recon, z, loss, {'recon_mse': recon_loss.detach(), 'kl': kl.detach()}

if __name__== "__main__":
    model = VAE()
    # sample data
    batch_size = 128
    input_channel = 3
    height, width = 224, 224
    x = torch.randn(batch_size, input_channel, height, width)
    
    output = model(x)

    print("出力の型：", type(output[1]))
    print("主力の形状", output[1].shape)

    print("出力の型：", type(output[0]))
    print("主力の形状", output[0].shape)
