# diffusion-model

## リポジトリ構造とアーキテクチャ要約
このリポジトリは、図形（line / circle / arc）画像を対象にした **VAE + 潜在拡散モデル** の学習・生成・評価パイプラインを実装しています。学習は2段階で、まず `train_vae.py` で画像再構成用VAEを学習し、その後 `train_latent_cond.py` でVAE潜在空間上の条件付きU-Net（幾何条件回帰ヘッド付き）を学習します。生成は `generate_cond.py`（CSV条件から一括生成）と `generate_steps.py`（1サンプルの逆拡散途中を保存）で実行でき、評価は `eval_iou_noise.py` で2値化・IoU・ノイズ率・ガウス重みRecallを算出します。補助的に `image_tools.py` で画像タイル表示や動画化、`vae_recon_check.py` でVAE再構成確認が可能です。

---

## プロジェクト概要
本プロジェクトは、CSVで与えた幾何パラメータ（線分端点、円中心・半径、円弧パラメータ）を条件として、図形画像を生成するための潜在拡散モデル実験コードです。`custom_dataset.py` の `LabelDataset` がCSVと画像を対応づけ、座標系変換・正規化・マスク生成を実施します。`diff.py` が拡散過程（加噪・逆拡散・CFG付き条件サンプリング）を担当し、`models/` 配下にU-Net群とVAEを持ちます。学習・生成・評価は基本的にスクリプト実行で行い、結果は `result/`、`generated_by_cond/`、`step_images/`、`eval_result/` などに保存される設計です。

## ディレクトリ構造
```text
.
├── custom_dataset.py
├── diff.py
├── early_stopping.py
├── entityCsvSampler.py
├── eval_iou_noise.py
├── generate_cond.py
├── generate_steps.py
├── image_tools.py
├── train_latent_cond.py
├── train_vae.py
├── utils.py
├── vae_recon_check.py
├── losses/
│   └── geom_losses.py
└── models/
    ├── __init__.py
    ├── unet.py
    ├── unet2.py
    ├── unet_cond.py
    ├── unet_cond_geom.py
    └── vae.py
```

## エントリーポイント（mainスクリプト）
用途別の実行起点は以下です。

- 学習（VAE）: `train_vae.py`
- 学習（条件付き潜在拡散）: `train_latent_cond.py`
- 生成（CSV条件からバッチ生成）: `generate_cond.py`
- 生成（逆拡散ステップ可視化）: `generate_steps.py`
- 評価（IoU/ノイズ率）: `eval_iou_noise.py`
- 画像可視化・動画化: `image_tools.py`（CLIサブコマンドあり）
- VAE再構成チェック: `vae_recon_check.py`

---

## 主要ファイル・ディレクトリ詳細

### 1) `train_vae.py`
- 役割:
  - `models/vae.py` のVAEを学習し、早期終了と損失ログ保存を行います。
- 入力 → 処理 → 出力:
  - 入力: 画像+caption CSV（`ClipDataset`）
  - 処理: `vae(x)` による再構成損失学習、`EarlyStopping` によるベスト保存
  - 出力: ベスト重み、lossグラフ、loss CSV
- 実行方法:
  - `python train_vae.py`
  - ※データパスがスクリプト内でWindows絶対パス指定されているため、環境に合わせて編集が必要です。
- 依存関係:
  - `torch`, `torchvision`, `pandas`, `Pillow`, `tqdm`, `matplotlib`
- 出力内容/形式:
  - `vae/<timestamp>/vae_best.pth`（PyTorch重み）
  - `vae/<timestamp>/losses_train_val.png`（PNG）
  - `vae/<timestamp>/losses_train_val.csv`（CSV）
- 出力先:
  - `./vae/<YYYY_MM_DD_HH_MM>/`

### 2) `train_latent_cond.py`
- 役割:
  - 凍結VAEの潜在に対して、条件付きU-Net（`UnetCondWithGeomHead`）を学習。
  - ノイズ予測損失 + 幾何回帰損失（マスク付きMSE）を扱う。
- 入力 → 処理 → 出力:
  - 入力: 画像と幾何条件CSV（`LabelDataset`）
  - 処理: VAE encode → 拡散ノイズ付与 → 条件付きノイズ予測/回帰 → 最適化
  - 出力: 学習済み重み、学習記録、（可能なら）サンプル生成画像
- 実行方法:
  - `python train_latent_cond.py`
- 主な引数/設定（スクリプト内部）:
  - `batch_size`, `epochs`, `lr`, `num_timesteps`, `cfg_drop_prob`, `geom_lambda`, `geom_dim`
- 依存関係:
  - `torch`, `torchvision`, `numpy`, `pandas`, `Pillow`, `tqdm`, `matplotlib`
- 出力内容/形式:
  - `./model_para/trained_para.pth`（随時更新される重み）
  - `result/<timestamp>/record.txt`（学習条件テキスト）
  - `result/<timestamp>/trained_para.pth`（最終保存）
  - `result/<timestamp>/losses_train_val.png`, `losses_train_val.csv`
  - `result/<timestamp>/generated_pic_arc/pic*.png`（サンプル生成成功時）
- 出力先:
  - `./model_para/`, `./result/<timestamp>/`

### 3) `generate_cond.py`
- 役割:
  - 学習済み `UnetCondWithGeomHead` と `VAE` を読み込み、CSV条件に従って line/circle/arc 画像を生成。
- 入力 → 処理 → 出力:
  - 入力: 各クラスのCSV（テスト用）
  - 処理: `EntityCsvSampler.sample()` で条件付き逆拡散
  - 出力: 生成画像PNG（クラス別ディレクトリ）
- 実行方法:
  - `python generate_cond.py`
- 依存関係:
  - `torch`, `pandas`, `numpy`, `Pillow`, `tqdm`
- 出力内容/形式:
  - `generated_by_cond/<run_name>/line/pic1.png ...`
  - `generated_by_cond/<run_name>/circle/pic1.png ...`
  - `generated_by_cond/<run_name>/arc/pic1.png ...`
- 出力先:
  - `./generated_by_cond/`

### 4) `generate_steps.py`
- 役割:
  - CSVの指定1行に対し、逆拡散の途中状態をステップごとに保存。
- 入力 → 処理 → 出力:
  - 入力: `csv_path`, `row_index`, `class_id`
  - 処理: 条件読み込み → `t=T..1` で逆拡散 → 各ステップでpixel/latent保存
  - 出力: ピクセル画像PNG + 潜在チャネル可視化PNG
- 実行方法:
  - `python generate_steps.py`
- 依存関係:
  - `torch`, `numpy`, `pandas`, `Pillow`, `tqdm`
- 出力内容/形式:
  - `step_images/<run_name>/pixel/t1000.png ... t1.png`
  - `step_images/<run_name>/latent/ch00/t1000.png ...`
- 出力先:
  - `./step_images/`（デフォルトは `./step_images/lambda01/...` を使用）

### 5) `eval_iou_noise.py`
- 役割:
  - GT画像 (`p00000.jpg`) と生成画像 (`pic1.png`) を対応付けて評価。
  - IoU, GT基準IoU, far noise ratio, gauss recall を算出。
- 入力 → 処理 → 出力:
  - 入力: `--gt_dir`, `--gen_dir`
  - 処理: 二値化・ペア比較・指標集計
  - 出力: 詳細CSV、サマリCSV、二値化画像、（任意）差分画像
- 実行方法:
  - `python eval_iou_noise.py --gt_dir <gt> --gen_dir <gen> --out_dir <out> --invert --save_diff`
- オプション:
  - `--threshold`, `--sigma`, `--max_pairs`, `--invert`, `--save_diff`
- 依存関係:
  - `numpy`, `pandas`, `Pillow`
  - 距離変換バックエンドとして `scipy` または `opencv-python`（どちらか必須）
- 出力内容/形式:
  - `metrics_detail.csv`, `metrics_summary.csv`, `config.txt`
  - `binarized/gt/*.png`, `binarized/gen/*.png`, `binarized/pair/*.png`
  - `diff/*.png`（`--save_diff`指定時）
- 出力先:
  - `<out_dir>/run_<timestamp>/`

### 6) `image_tools.py`
- 役割:
  - 画像ディレクトリのタイル可視化、動画化、2ディレクトリ比較動画の作成。
- 入力 → 処理 → 出力:
  - 入力: 画像ディレクトリ
  - 処理: 自然ソート・画像読込・描画/エンコード
  - 出力: タイルPNG、MP4動画
- 実行方法:
  - タイル: `python image_tools.py tile <dir> --rows 2 --cols 5 --out_dir <out>`
  - 動画: `python image_tools.py video <dir> --text --out <out.mp4> --fps 30`
  - 2列比較動画: `python image_tools.py video2 <dir1> <dir2> --text --out <out.mp4>`
- 依存関係:
  - `opencv-python`, `numpy`, `Pillow`, `matplotlib`
- 出力内容/形式:
  - タイル画像（PNG）
  - 動画（MP4）
- 出力先:
  - `--out_dir` / `--out` 指定先（未指定時は入力ディレクトリ配下既定名）

### 7) `vae_recon_check.py`
- 役割:
  - 学習済みVAEで再構成品質を可視化（元画像/再構成画像/グリッド）。
- 実行方法（関数呼び出し型）:
  - `save_vae_recon_examples(...)` を他スクリプトまたは対話環境から実行。
- 依存関係:
  - `torch`, `torchvision`, `Pillow`
- 出力内容/形式:
  - `recon_grid_bXXX.png`, `orig_bXXX_YY.png`, `recon_bXXX_YY.png`
- 出力先:
  - 指定した `out_dir`

### 8) コアモジュール
- `diff.py`
  - 拡散の前向き/逆過程、条件付きCFGサンプリング、潜在→画像変換を提供。
- `entityCsvSampler.py`
  - CSVのエンティティ条件を読み取り、`cond_vals` と `cond_mask` を構築して生成に供給。
- `custom_dataset.py`
  - `ClipDataset`（画像+テキスト）と `LabelDataset`（画像+幾何条件+マスク）を提供。
- `models/`
  - `vae.py`: 画像↔潜在の変換
  - `unet*.py`: ノイズ予測ネットワーク（条件付き版・幾何ヘッド付き版含む）
- `losses/geom_losses.py`
  - マスク付きMSE損失
- `early_stopping.py`
  - VAE学習での早期停止
- `utils.py`
  - モデル保存・結果記録・画像保存など共通処理

---

## 実行手順

### 1. 環境構築
```bash
conda env create -f requirements.yml
conda activate project-env
```

### 2. VAE学習
```bash
python train_vae.py
```

### 3. 条件付き潜在拡散の学習
```bash
python train_latent_cond.py
```

### 4. 生成（クラス別一括）
```bash
python generate_cond.py
```

### 5. 生成（逆拡散ステップ保存）
```bash
python generate_steps.py
```

### 6. 評価
```bash
python eval_iou_noise.py \
  --gt_dir ./data/arc_224x224_test \
  --gen_dir ./generated_by_cond/lambda_0/arc \
  --out_dir ./eval_result/lambda_0/arc \
  --invert \
  --save_diff
```

### 7. オプション指定例
```bash
python eval_iou_noise.py \
  --gt_dir ./data/line_224x224_test \
  --gen_dir ./generated_by_cond/lambda_0/line \
  --out_dir ./eval_result/lambda_0/line \
  --invert \
  --sigma 3.0 \
  --threshold 100 \
  --max_pairs 200
```

---

## 出力の説明

### 出力ファイル形式
- 学習済み重み: `.pth`
- ログ/指標: `.txt`, `.csv`
- 可視化画像: `.png`
- 動画: `.mp4`

### 結果ディレクトリ構造（代表例）
```text
result/
└── YYYY_MM_DD_HH_MM/
    ├── record.txt
    ├── trained_para.pth
    ├── losses_train_val.png
    ├── losses_train_val.csv
    └── generated_pic_arc/
        ├── pic1.png
        └── ...

generated_by_cond/
└── <run_name>/
    ├── line/pic*.png
    ├── circle/pic*.png
    └── arc/pic*.png

step_images/
└── <run_name>/
    ├── pixel/t*.png
    └── latent/
        ├── ch00/t*.png
        ├── ch01/t*.png
        └── ...

eval_result/
└── <exp>/
    └── run_YYYYMMDD_HHMMSS/
        ├── metrics_detail.csv
        ├── metrics_summary.csv
        ├── config.txt
        ├── binarized/
        │   ├── gt/*.png
        │   ├── gen/*.png
        │   └── pair/*.png
        └── diff/*.png (optional)
```

### ログ/メトリクスの読み方
- `losses_train_val.csv`
  - `train_loss`, `val_loss` をepochごとに記録。
  - `val_loss` は学習設定により一部epochが空欄（未評価）になる実装。
- `metrics_detail.csv`
  - 画像ペアごとの `iou`, `gt_iou`, `far_noise_ratio`, `gauss_recall` を記録。
- `metrics_summary.csv`
  - 各指標の平均/標準偏差/分位点（median, p90, p95）を集約。
  - `far_noise_ratio` は「GTからsigmaより離れた予測画素の割合」。

---

## 設定パラメータ（主要）

### `train_vae.py`
- `vae_epochs`, `vae_lr`, `vae_batch_size`
- `patience`, `min_delta`（早期停止）

### `train_latent_cond.py`
- `batch_size`, `epochs`, `lr`, `num_timesteps`
- `cfg_drop_prob`（classifier-free guidance学習用drop）
- `geom_lambda`（回帰損失重み）
- `geom_dim`（幾何条件ベクトル次元）

### `generate_steps.py`
- `row_index`, `class_id`
- `guidance_scale`
- `save_every` または `save_steps`

### `eval_iou_noise.py`
- `--threshold`（二値化閾値）
- `--invert`（黒前景扱い）
- `--sigma`（距離許容スケール）
- `--save_diff`（差分可視化出力）

---

## 注意事項（再現性）
- 学習/生成スクリプト内に **絶対パス** が直接書かれている箇所があるため、実行前に自環境パスへ修正してください。
- モデル重みファイル（`.pth`）の存在を前提とするスクリプト（`generate_cond.py`, `generate_steps.py`, `vae_recon_check.py`）は、先に対応する学習または重み配置が必要です。
- `eval_iou_noise.py` は距離変換に `scipy` を優先し、未導入時は `opencv-python` をフォールバックとして使用します。
