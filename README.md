# hiragana-asr

Lightweight Japanese ASR that outputs **hiragana only** — no hallucination by design.

wav2vec2-large + Dual CTC (InterCTC for phonemes, CR-CTC for kana). 315M parameters, runs real-time on MacBook Air M2.

## Why hiragana?

- **No hallucination**: CTC is structurally incapable of generating content not in the input
- **Lightweight**: 315M params (half of Whisper large-v3), FP16 inference ~630MB
- **Easy to fine-tune**: Simple CTC + wav2vec2 fine-tuning, no complex decoder
- **LLM-friendly**: Pass hiragana to an LLM for kanji conversion and intent understanding

## Model

Available on HuggingFace: [sakasegawa/japanese-wav2vec2-large-hiragana-ctc](https://huggingface.co/sakasegawa/japanese-wav2vec2-large-hiragana-ctc) / [Spaces Demo](https://huggingface.co/spaces/sakasegawa/hiragana-asr)

| Model | Data | JSUT KER | JVS KER | ReazonSpeech KER |
|-------|------|:--------:|:-------:|:----------------:|
| wav2vec2-large + 1,000h (ep5) | ReazonSpeech medium | 7.47% | 15.68% | 21.65% |

## Quick Start

```bash
# Install dependencies
uv sync

# Download UniDic (first time only)
uv run python -m unidic download

# Inference on audio file
uv run python scripts/03_infer.py --audio your_audio.wav

# Real-time ASR from microphone
uv run python scripts/realtime_asr.py
```

## Architecture

```
Audio (16kHz) → CNN Feature Extractor (frozen) → Transformer Encoder (24 layers)
                                                    ├── Layer 12 → Phoneme CTC Head (InterCTC)
                                                    └── Layer 24 → Kana CTC Head (CR-CTC)
```

- **Base encoder**: [reazon-research/japanese-wav2vec2-large](https://huggingface.co/reazon-research/japanese-wav2vec2-large) (pretrained on 35,000h)
- **InterCTC** ([Lee & Watanabe, ICASSP 2021](https://arxiv.org/abs/2102.03216)): Phoneme auxiliary task at layer 12
- **CR-CTC** ([Yao et al., ICLR 2025](https://arxiv.org/abs/2410.05101)): Consistency regularization for smoother CTC output
- **Loss**: `CR-CTC(kana) + 0.3 × CTC(phoneme)`

## Training

```bash
# Prepare dataset
uv run python scripts/00_prepare_dataset.py --splits medium

# Train (large model, 1000h)
uv run python scripts/01_train.py \
    --pretrained reazon-research/japanese-wav2vec2-large \
    --data-split medium --dataset-dir data/datasets/reazonspeech \
    --epochs 5 --batch-size 8 --grad-accum 4 --lr 5e-5 --bf16

# Evaluate
uv run python scripts/02_evaluate.py --checkpoint models/checkpoints/best.pt --dataset jsut
uv run python scripts/02_evaluate.py --checkpoint models/checkpoints/best.pt --dataset jvs
```

## Evaluation Datasets

| Dataset | Condition | Utterances |
|---------|-----------|------------|
| [JSUT-BASIC5000](https://sites.google.com/site/shinaborulab/publication/jsut) | Studio, single speaker | 5,000 |
| [JVS parallel100](https://sites.google.com/site/shinaborulab/publication/jvs) | 100 speakers | ~10,000 |
| [ReazonSpeech](https://research.reazon.jp/) | TV broadcast (wild) | ~2,600 |

## Blog Post

Detailed write-up (in Japanese): [ひらがなASRを作った話](https://nyosegawa.github.io/posts/hiragana-asr/)

## Additional Docs

- [Transformer の基本を知っている人向け hiragana-asr 解説](docs/transformer-project-overview.md)

## License

Apache-2.0. See [LICENSE](LICENSE).

Training data: [ReazonSpeech](https://research.reazon.jp/) (CDLA-Sharing-1.0 — model weights are unrestricted).
