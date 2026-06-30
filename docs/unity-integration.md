# Unity integration (ONNX + Sentis)

This document describes how the Python real-time hiragana ASR is made runnable
on-device in Unity, the design decisions, and the parity guarantees. The runtime
code and a step-by-step setup guide live in [`unity/HiraganaAsr/`](../unity/HiraganaAsr/README.md).

## Overview

```
Python (offline, once)                     Unity (runtime, every utterance)
─────────────────────                      ────────────────────────────────
checkpoint.pt                              Microphone (44.1/48 kHz)
  │ scripts/export_onnx.py                   │ StreamingResampler (anti-aliased)
  ▼                                          ▼
hiragana_asr_*.onnx  ───────────────────▶  16 kHz mono float[-1,1]
kana_vocab.json      ───────────────────▶    │ EnergyVadSegmenter
*.parity.json (fixture)                       ▼
                                            Sentis Worker (ONNX)
                                              ▼  kana_logits (1, T, 83)
                                            CtcDecoder (greedy / SWD)
                                              ▼
                                            ひらがな
```

The PyTorch pipeline (`scripts/realtime_asr.py`) is split into an **offline export**
step and a **C# runtime**. Everything that was Python at inference time — wav2vec2
normalization, the model forward pass, and CTC decoding — is reproduced exactly in
the exported graph and in C#.

## Model export

`scripts/export_onnx.py` wraps `DualCTCModel` into a kana-only module that maps a
raw waveform to `kana_logits`, then calls `torch.onnx.export`.

Important design choices:

- **Normalization is baked into the graph** (`--bake-norm`, default). The model
  input is the raw microphone waveform in `[-1, 1]`; the graph applies
  `(x - mean) / sqrt(var + 1e-5)` (population variance) internally. This removes a
  whole class of C#/Python preprocessing-parity bugs. Use `--no-bake-norm` only if
  you want to normalize yourself (`Wav2Vec2Preprocessor.Normalize`).
- **Eager attention** (`--attn eager`, default). The default SDPA attention exports
  `IsNaN`/`Where` guard ops that several Sentis versions do not support. Eager
  attention produces a cleaner graph. Measured effect on the medium model:
  `IsNaN` 24 → 0, `Where` 49 → 2.
- **Dynamic time axis** by default so variable-length utterances work. A
  `--fixed-length <sec>` option exports a single fixed shape as a fallback if a
  Sentis version struggles with dynamic shapes.
- **Kana-only by default**; `--dual` also exports the InterCTC phoneme head.
- **FP16** (`--fp16`) halves the model from ~1.26 GB to ~631 MB.

### ONNX operators (medium, eager, opset 17)

Every op produced by the default export was cross-checked, one by one, against the
Inference Engine 2.2 [supported operators](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.2/manual/supported-operators.html)
table (and its **Unsupported operators** list). All 27 op types are supported; none
appear in the unsupported list. The full per-export count is written to
`models/onnx/<name>.ops.json`.

| ONNX op (count) | wav2vec2 source | Inference Engine 2.2 |
|---|---|---|
| `Conv` (8) | feature-extractor Conv1d ×7 + conv positional embedding | ✅ 1D/2D/3D |
| `InstanceNormalization` (1) | feature-extractor **group norm** (see note) | ✅ |
| `LayerNormalization` (50) | per-layer layer norm | ✅ |
| `Erf` (32) | GELU activation | ✅ |
| `MatMul` (194) | attention / projections | ✅ |
| `Softmax` (24) | attention weights | ✅ |
| `Where` (2), `Equal` (1), `GreaterOrEqual` (1) | attention mask / range guards | ✅ (`Where`→`Select`) |
| `Add` `Mul` `Sub` `Div` `Sqrt` `ReduceMean` | residuals, baked normalization, GELU | ✅ |
| `Gather` `Transpose` `Reshape` `Concat` `Slice` `Unsqueeze` `Expand` `Shape` `Range` `ConstantOfShape` `Cast` `Constant` | shape / indexing plumbing | ✅ |

> **Why no `GroupNormalization`.** This is the one op to worry about: Inference
> Engine lists `GroupNormalization` (and `MeanVarianceNormalization`,
> `LpNormalization`) as **unsupported**. HuggingFace wav2vec2's
> `feat_extract_norm="group"` layer is defined as `nn.GroupNorm(dim, dim)` —
> `num_groups == num_channels` — which is mathematically per-channel (instance)
> normalization, so PyTorch's ONNX exporter emits it as `InstanceNormalization`,
> which **is** supported. The risk is avoided at export time, not worked around.

Other ops that wav2vec2 variants can produce are *not* present here and would be a
problem: `GroupNormalization`, `MeanVarianceNormalization`, RNN/GRU/`LSTM` (GPUPixel),
quantization ops (`QuantizeLinear`/`DequantizeLinear`), and `IsNaN`/`IsInf` (NaNs/Infs
are unsupported on `GPUCompute`/`GPUPixel`). Eager attention is what removes the
`IsNaN` guards that SDPA would otherwise emit.

**Verifying a re-export (e.g. the large preset).** Op coverage can change if you
swap the model, attention impl, or opset, so after any re-export:

1. Open `models/onnx/<name>.ops.json` and diff its keys against the table above.
2. Confirm none of the keys appear in Inference Engine's *Unsupported operators*
   list. Pay special attention to `GroupNormalization` reappearing — if it does, the
   source model uses `num_groups < num_channels` and needs a different normalization
   handling.
3. Run `Samples/SentisSmokeTest.cs`: an unsupported op surfaces as an import/compile
   error in the Console, so a clean run is the ground-truth confirmation.

If a future Sentis/Inference Engine version *does* reject an op, try `--opset 15`
(decomposes some fused ops) or switch the backend. Note that on **com.unity.sentis 1.x**
the `InstanceNormalization` / `Erf` / `LayerNormalization` coverage differs; use
`com.unity.ai.inference` (or Sentis 2.1+) for the matrix above.

## Parity guarantees

The export script verifies the ONNX model against PyTorch on CPU at multiple
lengths and writes a fixture for Unity to re-validate.

| Length | Frames | max\|Δlogit\| | decode match |
|-------:|-------:|--------------:|:------------:|
| 0.3 s  | 14     | ~1e-4         | ✓ |
| 1.0 s  | 49     | ~7e-5         | ✓ |
| 3.0 s  | 149    | ~6e-5         | ✓ |
| 10.0 s | 499    | ~2e-4         | ✓ |

Acceptance is **both** `max|Δlogit| < 1e-3` **and** identical decoded kana.
`Samples/SentisSmokeTest.cs` repeats the decoded-kana check inside Unity against
`*.parity.json` (real audio: `data/test.wav` → `ございますんにこんばんわ`). On GPU/FP16,
raw logits drift more than on CPU FP32, so the decoded-string equality is the
meaningful gate.

## C# runtime notes

- **Resampling.** Unity's `Microphone` records at 44.1/48 kHz. `StreamingResampler`
  applies a stateful Butterworth low-pass (cutoff `0.45 × 16 kHz`) before linear
  interpolation to avoid aliasing into the speech band, carrying filter state and
  the fractional read position across chunk boundaries.
- **VAD.** `EnergyVadSegmenter` reproduces the Python segmentation structure
  (0.5 s pre-buffer prepended on onset, 0.8 s silence timeout to finalize, force
  split on a max length) but detects speech with RMS energy against an **adaptive
  noise floor** plus hysteresis, instead of Silero. This is dependency-free and
  reliable in quiet rooms.
- **Decoding.** `CtcDecoder` implements greedy collapse (identical to
  `KanaVocab.decode`) and Spike Window Decoding (identical to `swd_decode` in
  `scripts/realtime_asr.py`, window 1).
- **Threading.** The sample runs inference on the main thread for simplicity.
  `DownloadToArray()` blocks until the GPU finishes; for production, prefer the
  async readback API (`ReadbackRequest` / `ReadbackAndCloneAsync`) so the frame
  doesn't stall.

## Optional: Silero VAD in Sentis

The energy VAD changes utterance boundaries versus Silero, which can affect
accuracy in noise. To match the Python behaviour more closely:

1. Export Silero VAD to ONNX (the upstream `snakers4/silero-vad` repo ships
   `silero_vad.onnx`).
2. Import it as a second `ModelAsset`.
3. Run it on 512-sample (32 ms) frames at 16 kHz. It is a **stateful RNN**, so you
   must carry its hidden state tensors between frames (feed the previous output
   state back as input) rather than treating each frame independently.
4. Replace the RMS test in a custom segmenter with `speechProb >= threshold`,
   keeping the same pre-buffer / silence-timeout / max-length logic.

This is a drop-in replacement for `EnergyVadSegmenter`'s detection step; the
segmentation bookkeeping is unchanged.

## Practical guidance for large models

`wav2vec2-large` (the medium/large presets) is heavy for on-device inference:
24 transformer layers, hidden 1024, and self-attention that scales with utterance
length. Before shipping:

- Benchmark size, memory and latency on the **actual target hardware**.
- Prefer the **FP16** export; consider Sentis import-time quantization.
- Keep `Max Utterance` modest (a few seconds) for XR/mobile.
- A smaller base model (12 layers, hidden 768) is a more realistic on-device
  default if you can train/obtain one.
