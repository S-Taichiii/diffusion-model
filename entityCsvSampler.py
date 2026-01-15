# entity_csv_sampler.py
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from typing import Optional, Tuple, Dict
from diff import Diffuser

class EntityCsvSampler:
    """
    直線/円/弧の CSV を読み込み、(cond_vals, cond_mask) を構築して
    diffuser.sample_latent_cond に渡すためのヘルパークラス。

    想定CSV（ヘッダなし, 13列）:
      col0: ダミー
      col1-4:  x1, y1, x2, y2
      col5-7:  cx, cy, cr
      col8-12: ax, ay, ar, theta1, theta2

    class_id:
      1 = line   (x1,y1,x2,y2)
      2 = circle (cx,cy,cr)
      3 = arc    (ax,ay,ar,theta1,theta2)
    """

    KEY_ORDER = ["x1","y1","x2","y2","cx","cy","cr","ax","ay","ar","theta1","theta2"]
    IDX: Dict[str, int] = {k:i for i,k in enumerate(KEY_ORDER)}

    def __init__(
        self,
        diffuser: Diffuser,
        model,
        vae,
        class_id: int = 1,
        base_wh: Optional[Tuple[float, float]] = (400, 400), # 図面サイズ
        device: Optional[torch.device] = None,
    ):
        self.diffuser = diffuser
        self.model = model
        self.vae = vae
        self.class_id = int(class_id)
        self.base_wh = base_wh  # (W,H) を固定したい時に指定
        self.device = device or getattr(diffuser, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # ===== 公開API ============================================================
    def set_class_id(self, class_id: int) -> None:
        """生成対象のエンティティ（直線/円/弧）を切り替える。"""
        self.class_id = int(class_id)

    def sample(
        self,
        csv_path: str,
        count: Optional[int] = None,
        start: int = 0,
        guidance_scale: float = 3.0,
    ):
        """
        指定 CSV を読み込み、class_id に応じて (cond_vals, cond_mask) を構築 → 生成。
        """
        df = pd.read_csv(csv_path, header=None)
        cond_vals_np, cond_mask_np = self._build_vals_mask_for(df, self.class_id, self.base_wh)

        # 使用範囲
        end = len(cond_vals_np) if count is None else min(start + count, len(cond_vals_np))
        if start >= end:
            raise ValueError("選択範囲にデータがありません。（start/countを確認）")
        vals = torch.from_numpy(cond_vals_np[start:end]).float().to(self.device)
        mask = torch.from_numpy(cond_mask_np[start:end]).float().to(self.device)

        N = vals.shape[0]

        return self.diffuser.sample_latent_cond(
            model=self.model,
            class_counts=(self.class_id, N),  # 単一クラス前提
            vae=self.vae,
            guidance_scale=guidance_scale,
            cond=vals,
            cond_mask=mask,
            # key_order は既定と一致（["x1","y1","x2","y2", ...]）
        )

    def load_cond(
        self,
        csv_path: str,
        count: Optional[int] = None,
        start: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成はせず、(cond_vals, cond_mask) の Tensor を返す。
        """
        df = pd.read_csv(csv_path, header=None)
        vals_np, mask_np = self._build_vals_mask_for(df, self.class_id, self.base_wh)
        end = len(vals_np) if count is None else min(start + count, len(vals_np))
        if start >= end:
            raise ValueError("選択範囲にデータがありません。（start/countを確認）")
        vals = torch.from_numpy(vals_np[start:end]).float().to(self.device)
        mask = torch.from_numpy(mask_np[start:end]).float().to(self.device)
        return vals, mask

    # ===== 内部：cond/mask 構築 ==============================================
    def _build_vals_mask_for(
        self,
        df: pd.DataFrame,
        class_id: int,
        base_wh: Optional[Tuple[float, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        # ここでの W,H は「図面(drawing)の幅・高さ」を意味させる（LabelDatasetのdrawW, drawHに相当）
        drawW, drawH = base_wh if base_wh is not None else self._infer_base_wh(df, class_id)  # [FIX] 意味を明確化
        B, K = len(df), len(self.KEY_ORDER)
        vals = np.zeros((B, K), dtype=np.float32)
        mask = np.zeros((B, K), dtype=np.float32)

        # [ADD] LabelDatasetと同じ正規化（図面→画像→0-1 の結果と一致）
        #  ※縦横比が同じ想定なら、画像サイズ(224)は消えて x/drawW, 1-y/drawH になる
        def norm_x_from_draw(x_draw: np.ndarray) -> np.ndarray:
            x_draw = x_draw.astype(np.float32)
            return x_draw / np.float32(drawW)

        def norm_y_from_draw(y_draw: np.ndarray) -> np.ndarray:
            y_draw = y_draw.astype(np.float32)
            return 1.0 - (y_draw / np.float32(drawH))  # LabelDatasetのY反転と一致

        def norm_r_from_draw(r_draw: np.ndarray) -> np.ndarray:
            r_draw = r_draw.astype(np.float32)
            # LabelDataset: r_img = r_draw * sx, norm_r = r_img / W_img
            # sx = W_img/drawW なので r_draw*(W_img/drawW)/W_img = r_draw/drawW
            return r_draw / np.float32(drawW)

        if class_id == 1:
            # line: x1,y1,x2,y2
            vals[:, self.IDX["x1"]] = norm_x_from_draw(df[1].to_numpy(dtype=np.float32))  # [FIX]
            vals[:, self.IDX["y1"]] = norm_y_from_draw(df[2].to_numpy(dtype=np.float32))  # [FIX]
            vals[:, self.IDX["x2"]] = norm_x_from_draw(df[3].to_numpy(dtype=np.float32))  # [FIX]
            vals[:, self.IDX["y2"]] = norm_y_from_draw(df[4].to_numpy(dtype=np.float32))  # [FIX]
            for k in ["x1", "y1", "x2", "y2"]:
                mask[:, self.IDX[k]] = 1.0

        elif class_id == 2:
            # circle: cx,cy,cr
            vals[:, self.IDX["cx"]] = norm_x_from_draw(df[5].to_numpy(dtype=np.float32))  # [FIX]
            vals[:, self.IDX["cy"]] = norm_y_from_draw(df[6].to_numpy(dtype=np.float32))  # [FIX]
            vals[:, self.IDX["cr"]] = norm_r_from_draw(df[7].to_numpy(dtype=np.float32))  # [FIX]
            for k in ["cx", "cy", "cr"]:
                mask[:, self.IDX[k]] = 1.0

        elif class_id == 3:
            # arc: ax,ay,ar,theta1,theta2
            vals[:, self.IDX["ax"]] = norm_x_from_draw(df[8].to_numpy(dtype=np.float32))  # [FIX]
            vals[:, self.IDX["ay"]] = norm_y_from_draw(df[9].to_numpy(dtype=np.float32))  # [FIX]
            vals[:, self.IDX["ar"]] = norm_r_from_draw(df[10].to_numpy(dtype=np.float32))  # [FIX]

            th1 = df[11].to_numpy(dtype=np.float32)
            th2 = df[12].to_numpy(dtype=np.float32)
            vals[:, self.IDX["theta1"]] = self._norm_angle_vec(th1)  # 0..1化（度→/360）でOK
            vals[:, self.IDX["theta2"]] = self._norm_angle_vec(th2)

            for k in ["ax", "ay", "ar", "theta1", "theta2"]:
                mask[:, self.IDX[k]] = 1.0

        else:
            raise ValueError("class_id must be 1(line), 2(circle), or 3(arc).")

        return vals, mask


    # ===== 内部：前処理ユーティリティ ========================================
    @staticmethod
    def _snap(v: float, choices=(224,256,280,300,320,384,400,448), tol=1.5) -> float:
        for c in choices:
            if abs(v - c) <= tol:
                return float(c)
        return float(v)

    def _infer_base_wh(self, df: pd.DataFrame, class_id: int) -> Tuple[float, float]:
        # クラスごとに対象列を絞って最大値から推定
        if class_id == 1:      # line
            xs = df[[1,3]].to_numpy()
            ys = df[[2,4]].to_numpy()
        elif class_id == 2:    # circle（半径は除外し、cx,cy だけ見る）
            xs = df[[5]].to_numpy()
            ys = df[[6]].to_numpy()
        elif class_id == 3:    # arc
            xs = df[[8]].to_numpy()
            ys = df[[9]].to_numpy()
        else:
            raise ValueError("class_id must be 1(line), 2(circle), or 3(arc).")

        x_max = float(np.max(np.abs(xs)))
        y_max = float(np.max(np.abs(ys)))
        return self._snap(x_max), self._snap(y_max)

    @staticmethod
    def _norm_angle_vec(v: np.ndarray) -> np.ndarray:
        """角度列を 0..1 に正規化。度（>1）と仮定して /360、既に 0..1 ならそのまま。"""
        out = v.astype(np.float32).copy()
        # >1 を度と見なす
        deg_mask = np.abs(out) > 1.0
        out[deg_mask] = (out[deg_mask] % 360.0) / 360.0
        return out
