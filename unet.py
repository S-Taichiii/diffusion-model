import torch
from torch import nn
import torch.nn.functional as F

def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)
        
class AttenionBlock(nn.Module):
    def __init__(self, channels):
        super(AttenionBlock, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    
    def forward(self, x: torch.Tensor):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = x = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResBlock(in_channels, in_channels, residual=True),
            ResBlock(in_channels, out_channels)
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            ResBlock(in_channels, in_channels, residual=True),
            ResBlock(in_channels, out_channels, in_channels // 2)
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class Unet(nn.Module):
    def __init__(self, in_ch=3, time_dim=256, remove_deep_conv=False):
        super().__init__()
        self.time_dim = time_dim
        self.remove_deep_conv = remove_deep_conv
        
        self.inc = ResBlock(in_ch, 64)
        self.down1 = Down(64, 128)
        self.sa1 = AttenionBlock(128)
        self.down2 = Down(128, 256)
        self.sa2 = AttenionBlock(256)
        self.down3 = Down(256, 256)
        self.sa3 = AttenionBlock(256)

        if remove_deep_conv:
            self.bot1 = ResBlock(256, 256)
            self.bot3 = ResBlock(256, 256)
        else:
            self.bot1 = ResBlock(256, 512)
            self.bot2 = ResBlock(512, 512)
            self.bot3 = ResBlock(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = AttenionBlock(128)
        self.up2 = Up(256, 64)
        self.sa5 = AttenionBlock(64)
        self.up3 = Up(128, 64)
        self.sa6 = AttenionBlock(64)
        self.out = nn.Conv2d(64, in_ch, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=one_param(self).device).float() / channels))

        pos_enc_sin = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_cos = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_sin, pos_enc_cos], dim=-1)
        return pos_enc

    def unet_forward(self, x, t):
        x1 = self.inc(x)
        # print("x1.shape=", x1.shape)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        # print("x2.shape=", x2.shape)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        # print("x3.shape=", x3.shape)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        if not self.remove_deep_conv:
            x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        # print("x4.shape=", x4.shape)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.out(x)

        return output

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        return self.unet_forward(x, t)


if __name__== "__main__":
    model = Unet(remove_deep_conv=True)
    # sample data
    batch_size = 128
    input_channel = 3
    height, width = 32, 32
    x = torch.randn(batch_size, input_channel, height, width)
    t = torch.randn(batch_size)
    
    output: torch.Tensor = model(x, t)

    print("出力の型：", type(output))
    print("主力の形状", output.shape)
