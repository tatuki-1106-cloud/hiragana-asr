# hiragana-asr と Whisper / Moonshine の比較

## 1. モデル概要

| 項目 | hiragana-asr (本プロジェクト) | Whisper (OpenAI) | Moonshine (UsefulSensors) |
|------|-------------------------------|-------------------|---------------------------|
| アーキテクチャ | wav2vec2-large + Dual CTC | Encoder-Decoder Transformer | Encoder-Decoder Transformer (RoPE) |
| デコーダ種別 | **非自己回帰 (CTC)** | 自己回帰 (autoregressive) | 自己回帰 (autoregressive) |
| 言語モデル | **なし** | あり (強力) | あり |
| パラメータ数 | 315M (large) | 39M–1,550M (tiny〜large-v3) | 27M (tiny) / 67M (base) |
| 出力形式 | **ひらがな + 音素** | 漢字かな混じり自然文 | 自然文 (英語のみ) |
| 対応言語 | 日本語専用 | 100言語以上 | **英語のみ** |
| ライセンス | Apache 2.0 | MIT | Apache 2.0 |

---

## 2. ハルシネーション耐性

### hiragana-asr の特徴

CTC (Connectionist Temporal Classification) は **構造的にハルシネーションが不可能**。

- CTC は入力フレームと出力トークンの単調アライメントを前提とする
- 語彙に存在しない単語を「推測」する言語モデル成分をもたない
- 誤りは「音響的に近い音への誤認識」に限定される。入力音声に存在しない内容を生成しない
- 無音区間に対してテキストを捏造しない

### Whisper の問題

強力な言語モデルが音響的な曖昧性を語彙的に補完する。これはWERを下げる一方でハルシネーションリスクを生む。

- 固有名詞・新語・専門用語を既知語に引き寄せて誤認識する
- 無音・低品質音声区間でテキストを捏造することがある ([Koenecke et al., 2024](https://arxiv.org/abs/2309.13453))
- データを増やしても根本的には改善されない（言語モデルの強さ自体が原因）

### Moonshine の問題

Whisper と同様に自己回帰デコーダをもつため、原理的にハルシネーションのリスクがある。英語に特化しているため日本語には適用不可。

| | hiragana-asr | Whisper | Moonshine |
|---|---|---|---|
| ハルシネーション | **構造的に不可能** | 発生しうる | 発生しうる |
| 未知語・固有名詞 | 音響的に忠実に転写 | 既知語に変換することがある | 英語のみ対象外 |
| 無音区間の誤生成 | **なし** | あり | あり |

---

## 3. レイテンシと速度

### 非自己回帰 vs 自己回帰

CTC はすべての出力トークンを**1回のフォワードパスで並列生成**する。自己回帰デコーダは1トークンずつ逐次生成するため、出力長に比例してレイテンシが増大する。

| モデル | デバイス | レイテンシ目安 (5秒音声) | RTF |
|--------|---------|------------------------|-----|
| hiragana-asr (large) | M2 Air MPS/FP16 | **~100ms** | ~0.02 |
| hiragana-asr (base) | M2 Air CPU | ~500ms | ~0.1 |
| Whisper medium (whisper.cpp) | M2 Max | ~3,300ms | ~0.66 |
| Whisper small (MLX) | M2 Air | ~1,500ms | ~0.3 |
| Moonshine tiny | Raspberry Pi 4 (CPU) | ~数百ms | <1.0 |

> RTF < 1.0 でリアルタイム処理可能。

### 固定長パディングの問題

Whisper はすべての入力を**30秒に固定**してパディングする。1秒の発話でも30秒分の計算が走る。

Moonshine は **RoPE (Rotary Position Embedding)** により可変長入力に対応し、短い発話では計算量が大幅に削減される。

hiragana-asr は CTC + wav2vec2 の構造上、**入力長に比例した計算量**で処理できる（固定長パディング不要）。

---

## 4. 精度

### JSUT-BASIC5000 での比較

| モデル | KER / CER | 備考 |
|--------|-----------|------|
| hiragana-asr large (1,000h) | **7.47%** (KER) | ひらがな出力、漢字変換前 |
| Whisper large-v3 | ~5–8% (CER) | 漢字かな混じり出力 |
| Kotoba-Whisper v2.2 | ≤ Whisper large-v3 (CER) | large-v3 以上 |
| Moonshine | 非対応 | 英語のみ |

> **注意**: hiragana-asr の KER はひらがな単位の誤り率。Whisper の CER は漢字も含む文字単位。単純比較は困難。

### 精度のトレードオフ

hiragana-asr は**精度を犠牲にしてハルシネーション耐性を取る**設計思想。

- 既知語の認識精度: Whisper > hiragana-asr
- 未知語・固有名詞の信頼性: hiragana-asr > Whisper（ハルシネーションがない）
- リアルタイム対話における実用精度: 用途依存

---

## 5. 出力形式と下流タスクとの統合

### hiragana-asr のパイプライン

```
音声 → hiragana-asr → ひらがな列 → LLM (漢字変換 + 意図理解)
```

- ひらがなは日本語の最小表記単位。LLM にとって直接処理しやすい
- ASR の誤りと LLM の変換誤りが独立しているため、エラーの原因分離が容易
- Whisper の「ハルシネーションが LLM の推論を汚染する」問題を回避できる

### Whisper / Moonshine のパイプライン

```
音声 → Whisper → 漢字かな混じり自然文 → 後段処理
```

- 書き起こし・要約・翻訳など最終出力形式がそのまま使えるタスクに適している
- ただしハルシネーションが混入するリスクがある

---

## 6. エッジデプロイメント

| 観点 | hiragana-asr | Whisper | Moonshine |
|------|-------------|---------|-----------|
| モデルサイズ (INT8) | ~80MB (large) / ~24MB (base) | ~40MB (tiny) 〜 ~380MB (large) | ~7MB (tiny) / ~18MB (base) |
| CPU推論 | 可 (base モデル) | 可 (tiny/small) | **可 (edge 特化)** |
| ANE/CoreML対応 | 将来対応予定 | WhisperKit経由で対応 | 未確認 |
| クロスプラットフォーム | ONNX変換で対応可能 | whisper.cpp (広く対応) | TFLite/ONNX 対応 |
| 対応言語 | 日本語専用 | 100言語以上 | 英語のみ |

---

## 7. ユースケース別推奨

| ユースケース | 推奨モデル | 理由 |
|-------------|-----------|------|
| リアルタイム音声対話 (日本語) | **hiragana-asr** | 低レイテンシ、ハルシネーションなし、LLM統合しやすい |
| 書き起こし・議事録 (日本語) | Kotoba-Whisper / Whisper large-v3 | 高精度、自然文出力 |
| 英語エッジASR | Moonshine | 軽量、英語特化、5〜15倍高速 |
| 多言語対応 | Whisper | 100言語以上サポート |
| ハルシネーション許容不可な用途 | **hiragana-asr** | 構造的保証 |
| モバイル/IoT英語デバイス | Moonshine tiny | 最小フットプリント |

---

## 8. まとめ

### hiragana-asr のメリット

1. **ハルシネーションが構造的に不可能** — CTC の設計上の保証
2. **超低レイテンシ** — 非自己回帰、~100ms (M2 Air)
3. **LLM統合に最適** — ひらがな出力は LLM に直接渡せる
4. **シンプルなアーキテクチャ** — fine-tune・改造が容易
5. **音素出力も同時取得** — InterCTC による音素列 (VSR統合・発音診断に活用可能)

### hiragana-asr のデメリット

1. **漢字変換が別途必要** — 最終出力はひらがなのみ
2. **既知語の精度は Whisper に劣る** — 言語モデルがないため
3. **日本語専用** — 多言語対応なし
4. **エコシステムが小さい** — Whisper のように広く使われているわけではない

### Whisper のメリット

- 既知語の認識精度が非常に高い
- 100言語以上に対応
- 豊富なエコシステム (whisper.cpp, WhisperKit, Kotoba-Whisper 等)
- 書き起こし・翻訳・要約まで1モデルで対応

### Whisper のデメリット

- **ハルシネーション**: 固有名詞・無音区間での誤生成
- **高レイテンシ**: 自己回帰 + 30秒固定パディング
- 大型モデルはエッジ推論に不向き

### Moonshine のメリット

- 超軽量 (tiny: 27M params, ~7MB INT8)
- **英語で Whisper Tiny 比 WER 48%削減**
- 可変長入力 (RoPE) で短い発話のレイテンシが低い
- Raspberry Pi など超制約デバイスで動作

### Moonshine のデメリット

- **英語専用** — 日本語には適用不可
- 自己回帰デコーダのためハルシネーションリスクは存在
- 日本語 ASR のコミュニティに普及していない

---

## 参考文献

- [Whisper (Radford et al., 2022)](https://arxiv.org/abs/2212.04356)
- [Moonshine (UsefulSensors, 2024)](https://arxiv.org/abs/2410.15608)
- [CR-CTC (Yao et al., ICLR 2025)](https://arxiv.org/abs/2410.05101)
- [InterCTC (Lee & Watanabe, ICASSP 2021)](https://arxiv.org/abs/2102.03216)
- [Whisper hallucination study (Koenecke et al., 2024)](https://arxiv.org/abs/2309.13453)
- [ReazonSpeech (Reazon Research)](https://research.reazon.jp/)
- [hiragana-asr HuggingFace モデル](https://huggingface.co/sakasegawa/japanese-wav2vec2-large-hiragana-ctc)
