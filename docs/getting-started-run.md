# hiragana-asr 実行手順書

このドキュメントは、初めてこのリポジトリを使う人向けに、
「実行できる状態にする」ための最短手順をまとめたものです。

対象OSは主に Windows (PowerShell) です。macOS/Linux でもコマンドはほぼ同じです。

## 0. 前提

- Git が使える
- Python 3.11 以上
- uv が使える
- マイク入力を使う場合はマイクデバイスが認識されている

### uv 未導入の場合

PowerShell で以下を実行:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

インストール後、ターミナルを開き直して `uv --version` が表示されることを確認してください。

## 1. リポジトリに移動

```powershell
cd c:\Project\hiragana-asr
```

## 2. 依存関係をインストール

```powershell
uv sync
```

初回のみ UniDic をダウンロード:

```powershell
uv run python -m unidic download
```

## 3. チェックポイントを準備

このプロジェクトの推論スクリプトは `.pt` 形式のチェックポイントを読み込みます。

- すでに学習済みチェックポイントがある場合: そのパスを `--checkpoint` に指定
- これから学習する場合: 学習後に `models/checkpoints` 配下に保存されたファイルを指定

### 3-1. 学習して作る（推奨）

`best.pt` は学習時に自動で保存されます。

```powershell
uv run python scripts/00_prepare_dataset.py --splits small
uv run python scripts/00b_preprocess.py ^
  --input data\datasets\reazonspeech\small ^
  --output data\datasets\reazonspeech\small_proc
uv run python scripts/01_train.py ^
  --pretrained reazon-research/japanese-wav2vec2-base ^
  --data-split small ^
  --dataset-dir data\datasets\reazonspeech
```

保存先（デフォルト）:

- `models/checkpoints/best.pt`

注意:

- `00_prepare_dataset.py` で使う ReazonSpeech は gated dataset なので、Hugging Face のアカウント作成・ログイン・利用規約同意が必要です。

### 3-2. 既存チェックポイントを配置する

手元の `.pt` がある場合は、次のように配置してください。

```powershell
mkdir models\checkpoints -Force
copy <手元のptファイル> models\checkpoints\best.pt
```

`best.pt` 以外の名前でも利用できます（実行時に `--checkpoint` で指定）。

### 3-3. 準備できたか確認する

```powershell
Get-ChildItem models\checkpoints\*.pt
```

1つ以上表示されれば準備完了です。

例:

- `models/checkpoints/best.pt`
- `models/checkpoints/best_large.pt`
- `models/checkpoints/best-medium-ep5-inference.pt`

## 4. 音声ファイルで推論

```powershell
uv run python scripts/03_infer.py ^
  --audio data\test.wav ^
  --checkpoint models\checkpoints\best.pt
```

補足:

- サンプリング周波数は自動で 16kHz にリサンプルされます
- GPU/MPS が使える環境では自動で利用されます

## 5. マイクでリアルタイム推論

`realtime_asr.py` のデフォルトは `models/checkpoints/best-medium-ep5-inference.pt` を参照します。
ファイル名が異なる場合は `--checkpoint` で明示してください。

```powershell
uv run python scripts/realtime_asr.py ^
  --checkpoint models\checkpoints\best.pt ^
  --pretrained reazon-research/japanese-wav2vec2-large
```

音素も表示したい場合:

```powershell
uv run python scripts/realtime_asr.py ^
  --checkpoint models\checkpoints\best.pt ^
  --pretrained reazon-research/japanese-wav2vec2-large ^
  --show-phonemes
```

終了は `Ctrl + C`。

## 6. うまく動かないときの確認ポイント

### チェックポイント関連エラー

- 指定した `--checkpoint` のパスが実在するか確認
- 学習時の `--pretrained` と推論時の `--pretrained` を揃える

### マイク入力が取れない

- 他アプリがマイクを占有していないか確認
- 入力デバイスを指定する場合は `--device-id` を使う

例:

```powershell
uv run python scripts/realtime_asr.py --device-id 1 --checkpoint models\checkpoints\best.pt
```

### 初回起動が遅い

- 初回はモデル読み込みと VAD 初期化で時間がかかります
- 2回目以降は短くなることがあります

## 7. 学習から試す場合 (任意)

### 7-1. ReazonSpeech の取得

```powershell
uv run python scripts/00_prepare_dataset.py --splits small
```

### 7-2. 前処理 (推奨)

```powershell
uv run python scripts/00b_preprocess.py ^
  --input data\datasets\reazonspeech\small ^
  --output data\datasets\reazonspeech\small_proc
```

### 7-3. 学習

```powershell
uv run python scripts/01_train.py ^
  --pretrained reazon-research/japanese-wav2vec2-base ^
  --data-split small ^
  --dataset-dir data\datasets\reazonspeech
```

### 7-4. 評価

```powershell
uv run python scripts/02_evaluate.py ^
  --checkpoint models\checkpoints\best.pt ^
  --data-split small
```

## 8. 参考

- 基本概要: `README.md`
- M2/MPS向け補足: `docs/m2-inference.md`
- 実行スクリプト:
  - `scripts/03_infer.py`
  - `scripts/realtime_asr.py`
  - `scripts/00_prepare_dataset.py`
  - `scripts/00b_preprocess.py`
  - `scripts/01_train.py`
  - `scripts/02_evaluate.py`
