import torch
import matplotlib
matplotlib.use('TkAgg')
from torchvision import transforms
from tqdm import tqdm
from typing import Dict, List, Iterable, Tuple, Union
from typing import Optional, Callable


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
        - x_shape: 生成する画像の形状 (image_num, C, H, W)。
        
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
        - z_shape: 生成する潜在の形状 (image_num, C, H, W)。例：(16, 4, 28, 28)
        - vae: VAE インスタンス（decode(z) が [0,1] 画像Tensorを返す想定）。Noneなら潜在を返す
        - to_pil: True の場合、画像Tensorを PIL.Image のリストに変換して返す（vae が必要）
        - progress: tqdm の進捗表示
        
        return:
          - vae が None のとき: z (Tensor, shape=z_shape)
          - vae がある & to_pil=False: imgs (Tensor, shape=(image_num,3,H*8,W*8) 程度)
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

        # 画像へデコード
        images = vae.decode(x)

        if to_pil:
            return [self.reverse_to_img(images[i]) for i in range(batch_size)]
        else:
            return images

    def denoise_cond(
            self,
            model,
            x,
            t,
            y=None,
            guidance_scale=0.0,
            null_label=0,
            cond_vals=None,
            cond_mask=None,
        ):
        """One DDPM step with optional classifier-free guidance."""
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()
        t_idx = t - 1
        alpha = self.alphas[t_idx].view(-1,1,1,1)
        alpha_bar = self.alpha_bars[t_idx].view(-1,1,1,1)
        alpha_bar_prev = self.alpha_bars[torch.clamp(t_idx-1, min=0)].view(-1,1,1,1)

        with torch.no_grad():
            if guidance_scale and y is not None and guidance_scale > 0:
                y_null = torch.full_like(y, null_label)
                eps_uncond = model(x, t, y_null, cond_vals=cond_vals, cond_mask=cond_mask)
                eps_cond = model(x, t, y, cond_vals=cond_vals, cond_mask=cond_mask)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                # plain conditional/unconditional
                if y is None:
                    y = torch.full((x.size(0),), null_label, device=x.device, dtype=torch.long)
                    eps = model(x, t, y, cond_vals=cond_vals, cond_mask=cond_mask)

        noise = torch.randn_like(x, device=self.device)
        noise[t == 1] = 0
        mu = (x - ((1-alpha) / torch.sqrt(1-alpha_bar)) * eps) / torch.sqrt(alpha)
        std = torch.sqrt((1-alpha) * (1-alpha_bar_prev) / (1-alpha_bar))
        return mu + noise * std


    def sample_cond(self, model, x_shape, y, guidance_scale=0.0, null_label=0):
        batch_size = x_shape[0]
        assert y.shape[0] == batch_size
        x = torch.randn(x_shape, device=self.device)
        for i in range(self.num_timesteps, 0, -1):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x = self.denoise_cond(model, x, t, y=y, guidance_scale=guidance_scale, null_label=null_label)
        return x


    # def sample_latent_cond(
    #     self,
    #     model,
    #     class_counts: Union[Dict[int, int], Tuple[int, int], List[Tuple[int, int]]],
    #     z_shape: Tuple[int, int, int] = None,   # (C, H, W) ※image_numはclass_countsから自動
    #     vae=None,
    #     to_pil: bool = True,
    #     progress: bool = True,
    #     guidance_scale: float = 3.0,
    #     null_label: int = 0,
    #     dummy_input_hw: Tuple[int, int] = (224, 224),  # VAEから潜在形状を推定する際の入力解像度
    # ):
    #     """
    #     Latent Diffusion 向け 条件付きサンプリング（クラスと枚数で指定）。
    #     例: class_counts={1:8, 2:8, 3:8} → 各クラス8枚ずつ生成

    #     args:
    #     - model: 潜在空間上の条件付きU-Net（forward(x, t, y)）
    #     - class_counts:
    #         * dict: {class_id: num, ...}（class_idは 1=line, 2=circle, 3=arc を推奨 / 0はuncond用）
    #         * または (class_id, num) のタプル、あるいはそれらのリスト
    #     - z_shape: 生成潜在の (C, H, W)。未指定なら VAE から自動推定（vae が必要）
    #     - vae: VAE インスタンス（decode(z)->[0,1] Tensor）
    #     - to_pil: Trueなら PIL.Image のリストを返す
    #     - progress: tqdm 進捗表示
    #     - guidance_scale: CFGの強さ（0で無効）
    #     - null_label: 無条件用ラベルID（通常0）
    #     - dummy_input_hw: VAE潜在形状推定に使うダミー画像サイズ（既定224x224）

    #     return:
    #     - vae が None: z (Tensor, shape=(image_num,C,H,W))
    #     - vae あり & to_pil=False: images Tensor (image_num,3,H_img,W_img)
    #     - vae あり & to_pil=True: List[PIL.Image]
    #     """

    #     device = self.device

    #     # ---- class_counts の正規化 ----
    #     def _norm_counts(cc):
    #         if isinstance(cc, dict):
    #             items = list(cc.items())
    #         elif isinstance(cc, tuple) and len(cc) == 2:
    #             items = [cc]
    #         elif isinstance(cc, list):
    #             items = list(cc)
    #         else:
    #             raise ValueError("class_counts は {cls: num} 辞書、(cls,num) タプル、またはそのリストで指定してください。")
    #         # フィルタ: num<=0 を除外
    #         items = [(int(c), int(n)) for c, n in items if int(n) > 0]
    #         if not items:
    #             raise ValueError("生成枚数が0です。class_countsを見直してください。")
    #         return items

    #     items = _norm_counts(class_counts)

    #     # ---- バッチ用 y ベクトルを組み立て（例: [1,1,1,2,2,3,3,3,...]）----
    #     y_list: List[int] = []
    #     for cls, num in items:
    #         y_list += [cls] * num
    #     image_num = len(y_list)
    #     y = torch.tensor(y_list, device=device, dtype=torch.long)

    #     # ---- 潜在の (C,H,W) 決定 ----
    #     if z_shape is None:
    #         if vae is None:
    #             raise ValueError("z_shape を省略する場合は vae が必要です。")
    #         with torch.no_grad():
    #             H, W = dummy_input_hw
    #             dummy = torch.zeros(1, 3, H, W, device=device)
    #             z, _ = vae.encode(dummy)  # (1, C, H', W')
    #             C, Hlat, Wlat = z.shape[1:]
    #     else:
    #         C, Hlat, Wlat = z_shape

    #     # ---- 逆拡散ループ（CFG対応）----
    #     x = torch.randn((image_num, C, Hlat, Wlat), device=device)

    #     step_iter: Iterable[int] = range(self.num_timesteps, 0, -1)
    #     if progress:
    #         step_iter = tqdm(step_iter, desc="Sampling (cond)")

    #     with torch.no_grad():
    #         for i in step_iter:
    #             t = torch.full((image_num,), i, device=device, dtype=torch.long)

    #             # 無条件(0)と条件(y)の差分を guidance_scale で強調
    #             x = self.denoise_cond(
    #                 model=model,
    #                 x=x,
    #                 t=t,
    #                 y=y,
    #                 guidance_scale=guidance_scale,
    #                 null_label=null_label,
    #             )

    #     # ---- 画像に戻さず潜在を返す場合 ----
    #     if vae is None:
    #         return x

    #     # ---- VAEでデコード
    #     images = vae.decode(x)  # (image_num,3,H_img,W_img), 値域[0,1]

    #     if to_pil:
    #         # reverse_to_img は既存の画像復元ヘルパ（[0,1]Tensor -> PIL）を想定
    #         return [self.reverse_to_img(images[i]) for i in range(image_num)]
    #     else:
    #         return images


    def sample_latent_cond(
        self,
        model,
        class_counts: Union[Dict[int, int], Tuple[int, int], List[Tuple[int, int]]],
        z_shape: Tuple[int, int, int] = None,   # (C, H, W)
        vae=None,
        to_pil: bool = True,
        progress: bool = True,
        guidance_scale: float = 3.0,
        null_label: int = 0,
        dummy_input_hw: Tuple[int, int] = (224, 224),

        # ▼▼ 追加: 数値条件の入力 ▼▼
        cond: Optional[Union[
            Dict[int, Dict[str, float]],     # 例: {1: {"x1":0.1,"y1":0.2,...}, 2:{...}}
            List[Dict[str, float]],           # サンプルごと指定（総枚数と同じ長さ）
            torch.Tensor                      # 直接(B,K)
        ]] = None,
        cond_mask: Optional[Union[
            Dict[int, Dict[str, float]],     # 0/1 で指定（辞書でもOK）
            List[Dict[str, float]],
            torch.Tensor
        ]] = None,
        key_order: Optional[List[str]] = None,      # 省略時は学習と同じ12キー
        class_keys: Optional[Dict[int, List[str]]] = None,  # マスク自動用（省略可）
    ):
        """
        Latent Diffusion 条件付きサンプリング（クラス＋数値条件対応）
        """
        device = self.device

        # ---- 既存: class_counts 正規化 ----
        def _norm_counts(cc):
            if isinstance(cc, dict):
                items = list(cc.items())
            elif isinstance(cc, tuple) and len(cc) == 2:
                items = [cc]
            elif isinstance(cc, list):
                items = list(cc)
            else:
                raise ValueError("class_counts は {cls:num}, (cls,num), そのリストのいずれか。")
            items = [(int(c), int(n)) for c, n in items if int(n) > 0]
            if not items:
                raise ValueError("生成枚数が0です。")
            return items

        items = _norm_counts(class_counts)

        # ---- y を展開 ----
        y_list: List[int] = []
        for cls, num in items:
            y_list += [cls] * num
        B = len(y_list)
        y = torch.tensor(y_list, device=device, dtype=torch.long)

        # ---- 既定キー・クラス使用キー ----
        if key_order is None:
            key_order = ["x1","y1","x2","y2","cx","cy","cr","ax","ay","ar","theta1","theta2"]
        K = len(key_order)
        key_index = {k:i for i,k in enumerate(key_order)}
        if class_keys is None:
            class_keys = {
                1: ["x1","y1","x2","y2"],
                2: ["cx","cy","cr"],
                3: ["ax","ay","ar","theta1","theta2"],
            }

        # ---- cond/cond_mask を (B,K) テンソルへ構築 ----
        def _as_tensor_like(x, fallback_zero=False):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            if fallback_zero:
                return torch.zeros((B, K), device=device, dtype=torch.float32)
            return None

        vals = _as_tensor_like(cond)
        msk  = _as_tensor_like(cond_mask)

        if vals is None:
            # 辞書/リスト入力を (B,K) に展開
            vals = torch.zeros((B, K), device=device, dtype=torch.float32)
            msk  = torch.zeros((B, K), device=device, dtype=torch.float32) if msk is None else msk

            if isinstance(cond, dict):
                # クラスごと指定
                for i, cls in enumerate(y_list):
                    if cls in cond:
                        for k, v in cond[cls].items():
                            if k in key_index:
                                vals[i, key_index[k]] = float(v)
                                # mask指定がない場合、自動で 1
                                if isinstance(cond_mask, dict):
                                    if cls in cond_mask and k in cond_mask[cls]:
                                        msk[i, key_index[k]] = float(cond_mask[cls][k])
                                    else:
                                        msk[i, key_index[k]] = 1.0
                                else:
                                    msk[i, key_index[k]] = 1.0
                    # cond に無いキーは 0/未使用
                    if isinstance(cond_mask, dict) and (cls in cond_mask):
                        # 明示マスクの残りも反映
                        for k, mv in cond_mask[cls].items():
                            if k in key_index:
                                msk[i, key_index[k]] = float(mv)

            elif isinstance(cond, list):
                if len(cond) != B:
                    raise ValueError(f"cond(list) の長さ {len(cond)} が生成枚数 {B} と不一致。")
                for i, d in enumerate(cond):
                    for k, v in d.items():
                        if k in key_index:
                            vals[i, key_index[k]] = float(v)
                            if isinstance(cond_mask, list) and i < len(cond_mask) and k in cond_mask[i]:
                                msk[i, key_index[k]] = float(cond_mask[i][k])
                            else:
                                msk[i, key_index[k]] = 1.0
                # list で mask だけ与えられた場合も反映
                if isinstance(cond_mask, list) and len(cond_mask) == B:
                    for i, d in enumerate(cond_mask):
                        for k, mv in d.items():
                            if k in key_index:
                                msk[i, key_index[k]] = float(mv)

            else:
                # cond=None なら、class_keys から「使う要素=1」でマスクだけ作る（値は0）
                for i, cls in enumerate(y_list):
                    for k in class_keys.get(cls, []):
                        if k in key_index:
                            msk[i, key_index[k]] = 1.0
        else:
            # vals が (B,K) テンソルで渡された場合
            if vals.ndim != 2 or vals.shape[0] != B or vals.shape[1] != K:
                raise ValueError(f"cond Tensor 形状は (B={B}, K={K}) 必須: got {tuple(vals.shape)}")
            if msk is None:
                # mask 未指定なら「非ゼロ項目を 1」として自動作成
                msk = (vals != 0).float()
            else:
                if msk.ndim != 2 or msk.shape != vals.shape:
                    raise ValueError(f"cond_mask Tensor 形状は cond と同じ (B,K) 必須。")

        # ---- 潜在形状の決定 ----
        if z_shape is None:
            if vae is None:
                raise ValueError("z_shape 省略時は vae が必要です。")
            with torch.no_grad():
                H, W = dummy_input_hw
                dummy = torch.zeros(1, 3, H, W, device=device)
                z, _ = vae.encode(dummy)
                C, Hlat, Wlat = z.shape[1:]
        else:
            C, Hlat, Wlat = z_shape

        # ---- 逆拡散（数値条件も渡す）----
        x = torch.randn((B, C, Hlat, Wlat), device=device)
        step_iter: Iterable[int] = range(self.num_timesteps, 0, -1)
        if progress:
            step_iter = tqdm(step_iter, desc="Sampling (cond+numeric)")

        with torch.no_grad():
            for i in step_iter:
                t = torch.full((B,), i, device=device, dtype=torch.long)
                x = self.denoise_cond(
                    model=model,
                    x=x,
                    t=t,
                    y=y,
                    guidance_scale=guidance_scale,
                    null_label=null_label,
                    cond_vals=vals,
                    cond_mask=msk,
                )

        if vae is None:
            return x
        # --- decode (chunked) ---
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        decode_chunk = 4  # まずは1で確実に。OKなら2,4,8と増やす
        outs = []

        vae.eval()
        with torch.inference_mode():
            for s in range(0, B, decode_chunk):
                xb = x[s:s + decode_chunk]

                # decodeだけAMPで軽量化
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    img_b = vae.decode(xb)

                outs.append(img_b)

        images = torch.cat(outs, dim=0)

        return [self.reverse_to_img(images[i]) for i in range(B)] if to_pil else images
