# models/unet_cond_geom.py
import torch
from torch import nn
import torch.nn.functional as F

from models.unet_cond import UnetCond, ResBlock, Down, Up, AttenionBlock  # ← 既存を利用

class GeomHead(nn.Module):
    """
    UNetの特徴マップ (B,C,H,W) から (B,D) の幾何ベクトルを回帰するヘッド
    """
    def __init__(self, in_ch: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # Global Average Pooling: (B,C,H,W) -> (B,C)
        g = feat.mean(dim=(2, 3))
        return self.mlp(g)


class UnetCondWithGeomHead(UnetCond):
    """
    既存 UnetCond を変更せずに、回帰ヘッド出力 geom_pred を追加で返す版。
    - 返り値: (eps_pred, geom_pred)
    """
    def __init__(
        self,
        in_ch=4,
        time_dim=256,
        num_classes=3,
        cfg_drop_prob=0.0,      # ★ 推奨: ドロップは学習側で制御（後述）
        remove_deep_conv=False,
        geom_dim=12,            # ★ 例: cond_vals の次元(K)に合わせる
        geom_hidden=256,
    ):
        super().__init__(
            in_ch=in_ch,
            time_dim=time_dim,
            num_classes=num_classes,
            cfg_drop_prob=cfg_drop_prob,
            remove_deep_conv=remove_deep_conv,
        )
        # UNetの最終直前の特徴は 64ch（self.out が 64->in_ch なので）
        self.geom_head = GeomHead(in_ch=64, out_dim=geom_dim, hidden=geom_hidden)

    # ★★★ 追加：unet_forward を「特徴も返す」版に上書き（元ファイルは変更しない） ★★★
    def unet_forward_with_feat(self, x, t):
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        if not self.remove_deep_conv:
            x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)

        feat = x                       # ← ここが (B,64,H,W)
        eps_pred = self.out(x)         # ← 既存のノイズ予測

        return eps_pred, feat

    # ★★★ 追加：forward を「eps_pred, geom_pred」を返すように拡張 ★★★
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cond_vals: torch.Tensor = None,
        cond_mask: torch.Tensor = None,
        cond_drop_prob: float = 0.0,   # ★ ここではドロップしない前提（学習側で制御）
    ):
        # 既存の fused embedding と数値条件埋め込みを使う（ただしcfg_dropは0推奨）
        emb = self.fused_embedding(t, y)

        if (cond_vals is not None) and (cond_mask is not None):
            # 内部での cond ドロップは使わない（外部制御推奨）
            cond_feat = torch.cat([cond_vals, cond_mask], dim=1)
            cond_emb = self.cond_mlp(cond_feat)
            emb = emb + cond_emb

        eps_pred, feat = self.unet_forward_with_feat(x, emb)
        geom_pred = self.geom_head(feat)  # (B, geom_dim)

        return eps_pred, geom_pred
