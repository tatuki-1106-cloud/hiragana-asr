# HiraganaAsr for Unity (ONNX + Sentis)

On-device, real-time Japanese **hiragana** speech recognition for Unity, ported
from the Python project [`hiragana-asr`](../../README.md). It runs the wav2vec2 +
Dual-CTC model with [Unity Sentis](https://docs.unity3d.com/Packages/com.unity.sentis@2.1/manual/index.html)
(a.k.a. *Inference Engine*) and re-implements VAD, preprocessing and CTC decoding
in C#.

```
Microphone → resample 16 kHz → Energy VAD → Sentis (ONNX) → CTC decode → ひらがな
```

## Requirements

- Unity 2023.2+ / Unity 6.
- The Sentis package: **com.unity.sentis 2.1+** *or* the renamed
  **com.unity.ai.inference**. If you use `com.unity.ai.inference`, change
  `using Unity.Sentis;` to `using Unity.InferenceEngine;` in the `.cs` files.
- A microphone, and microphone permission on the target platform.

## 1. Export the model (Python side)

From the repo root:

```bash
uv pip install onnx onnxruntime          # one-time
uv run python scripts/export_onnx.py --model medium --fixture-audio data/test.wav
# FP16 (half the size, recommended for on-device):
uv run python scripts/export_onnx.py --model medium --fixture-audio data/test.wav \
    --fp16 --output models/onnx/hiragana_asr_medium_fp16.onnx
```

This writes to `models/onnx/`:

| File | Purpose |
|------|---------|
| `hiragana_asr_medium.onnx` | the model (FP32 ≈ 1.26 GB, FP16 ≈ 631 MB) |
| `kana_vocab.json` | ordered id→token table for C# decoding |
| `hiragana_asr_medium.ops.json` | ONNX operator list (Sentis triage) |
| `hiragana_asr_medium.parity.json` | end-to-end fixture (waveform + expected kana) |

The exported graph takes a **raw 16 kHz mono waveform in [-1, 1]** and bakes the
wav2vec2 normalization in, so the C# side only resamples and decodes.

## 2. Import into Unity

1. Copy this `HiraganaAsr/` folder into your project's `Assets/`.
2. Copy the exported `*.onnx` into `Assets/` — Unity imports it as a `ModelAsset`.
3. `kana_vocab.json` is already here and imports as a `TextAsset`.

## 3. Smoke test first (recommended)

Sentis operator support is the main unknown for a large wav2vec2 graph, so verify
before building UI:

1. Create an empty GameObject, add **`SentisSmokeTest`**.
2. Assign the `ModelAsset`, `kana_vocab.json`, and `Samples/hiragana_asr_medium.parity.json`.
3. Press Play. The Console should log `ALL PASSED` — Sentis output matches the
   PyTorch reference for both greedy and SWD decoding.

If it fails to import/run, see *Troubleshooting* below.

## 4. Real-time microphone ASR

1. Add **`RealtimeAsrController`** to a GameObject.
2. Assign the `ModelAsset` and `kana_vocab.json`.
3. (Optional) hook `onUtteranceFinalized (string)` to a UI Text to display
   finalized lines, and `onPreviewUpdated (string)` to a second Text for the
   provisional live preview (it rewrites itself as you speak; prefixed `…` when the
   preview window is clipped). Toggle the preview off with `Enable Preview`.
4. Press Play and speak. Finalized hiragana lines are logged and raised as events.

Key inspector settings:

- **Backend** — `GPUCompute` (fastest), `CPU` (most compatible).
- **Decode Mode** — `Swd` (default, matches realtime_asr.py) or `Greedy`.
- **Bake Norm** — leave `true` unless you exported with `--no-bake-norm`.
- **Silence Timeout / Max Utterance / Prebuffer / Speech Factor** — VAD tuning.

## Files

| Script | Role |
|--------|------|
| `Runtime/KanaVocab.cs` | load `kana_vocab.json` |
| `Runtime/CtcDecoder.cs` | CTC greedy + Spike Window Decoding |
| `Runtime/Wav2Vec2Preprocessor.cs` | normalization (only for `--no-bake-norm`) |
| `Runtime/StreamingResampler.cs` | anti-aliased mic-rate → 16 kHz |
| `Runtime/MicrophoneCapture.cs` | mic streaming as 16 kHz mono |
| `Runtime/EnergyVadSegmenter.cs` | utterance segmentation (energy VAD) |
| `Runtime/HiraganaAsrModel.cs` | Sentis worker + decode |
| `Samples/RealtimeAsrController.cs` | full mic→kana MonoBehaviour |
| `Samples/SentisSmokeTest.cs` | one-shot parity validation |

## Troubleshooting

- **A layer/operator is unsupported by the backend** — every op in the default
  export was verified against Inference Engine 2.2's supported-operators table (see
  the matrix in [`docs/unity-integration.md`](../../docs/unity-integration.md)), so a
  fresh medium export should import cleanly. If you re-export (different model, opset
  or attention), diff the model's `*.ops.json` against the
  [Inference Engine supported operators](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.2/manual/supported-operators.html)
  and its *Unsupported operators* list — watch for `GroupNormalization` reappearing.
  Mitigations: try a different `--opset` (e.g. `--opset 15`), switch `Backend` to
  `CPU`. The default export uses **eager attention** specifically to avoid `IsNaN`/
  `Where` ops that some Sentis versions reject. On **com.unity.sentis 1.x** the
  normalization/`Erf` coverage differs — prefer `com.unity.ai.inference` (Sentis 2.1+).
- **First inference is slow / hitches** — Sentis compiles shaders and reallocates
  on the first run and on each new input length. Warm up at startup, and consider
  a fixed-length export (`--fixed-length 6.0`) to keep one shape.
- **Out of memory / too slow on mobile or XR** — use the FP16 export, lower
  `Max Utterance`, or train/use a smaller base model.
- **Wrong/garbled kana** — verify the smoke test passes; mismatches usually mean
  precision (FP16/GPU) drift, an operator fallback, or resampling differences.

See [`docs/unity-integration.md`](../../docs/unity-integration.md) for design
details and the optional Silero-VAD-in-Sentis upgrade.
