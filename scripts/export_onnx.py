"""Export the hiragana ASR (DualCTC) model to ONNX for Unity Sentis.

The exported graph takes a RAW 16 kHz mono waveform and (by default) bakes the
wav2vec2 zero-mean / unit-variance normalization into the graph, so the Unity
client only has to feed microphone samples in [-1, 1]. The primary output is
``kana_logits`` (B, T', kana_vocab_size); ``--dual`` additionally exports the
InterCTC phoneme logits.

Outputs (written next to ``--output``):
    <name>.onnx                  ONNX model
    kana_vocab.json              ordered id->token table + blank index (for C#)
    <name>.ops.json              list of ONNX operators in the graph (Sentis triage)
    <name>.parity.json           end-to-end parity fixture for Unity validation
    <name>.fixture.wav           the fixture waveform as 16 kHz mono PCM (optional)

Usage:
    uv run python scripts/export_onnx.py                       # medium preset, dynamic length
    uv run python scripts/export_onnx.py --model large
    uv run python scripts/export_onnx.py --dual                # also export phoneme head
    uv run python scripts/export_onnx.py --fixed-length 6.0    # fixed-shape fallback for Sentis
    uv run python scripts/export_onnx.py --opset 15            # lower opset if Sentis rejects ops
"""

# ruff: noqa: E402

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.asr.kana_vocab import BLANK_IDX, BLANK_TOKEN, KanaVocab
from src.asr.model import load_checkpoint
from src.asr.phoneme_vocab import PhonemeVocab

SAMPLE_RATE = 16000

MODEL_PRESETS = {
    "medium": {
        "checkpoint": "models/checkpoints/best-medium-ep5-inference.pt",
        "pretrained": "reazon-research/japanese-wav2vec2-large",
    },
    "large": {
        "checkpoint": "models/checkpoints/best_large.pt",
        "pretrained": "reazon-research/japanese-wav2vec2-large",
    },
}

NORM_EPS = 1e-5


def normalize_waveform(waveform: torch.Tensor) -> torch.Tensor:
    """Wav2Vec2-style per-utterance normalization (matches do_normalize=True)."""
    mean = waveform.mean(dim=-1, keepdim=True)
    var = waveform.var(dim=-1, keepdim=True, unbiased=False)
    return (waveform - mean) / torch.sqrt(var + NORM_EPS)


class KanaOnnxWrapper(nn.Module):
    """Raw-waveform -> kana_logits, with optional baked-in normalization."""

    def __init__(self, model: nn.Module, bake_norm: bool):
        super().__init__()
        self.encoder = model.encoder
        self.kana_head = model.kana_head
        self.bake_norm = bake_norm

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if self.bake_norm:
            audio = normalize_waveform(audio)
        hidden = self.encoder(audio).last_hidden_state
        return self.kana_head(hidden)


class DualOnnxWrapper(nn.Module):
    """Raw-waveform -> (kana_logits, phoneme_logits)."""

    def __init__(self, model: nn.Module, bake_norm: bool):
        super().__init__()
        self.encoder = model.encoder
        self.kana_head = model.kana_head
        self.phoneme_head = model.phoneme_head
        self.inter_ctc_layer = model.inter_ctc_layer
        self.bake_norm = bake_norm

    def forward(self, audio: torch.Tensor):
        if self.bake_norm:
            audio = normalize_waveform(audio)
        outputs = self.encoder(audio, output_hidden_states=True)
        kana_logits = self.kana_head(outputs.last_hidden_state)
        phoneme_logits = self.phoneme_head(outputs.hidden_states[self.inter_ctc_layer])
        return kana_logits, phoneme_logits


def ctc_greedy_decode(logits: np.ndarray, vocab: KanaVocab) -> str:
    """Greedy CTC decode of (T, V) logits -> kana string."""
    ids = logits.argmax(axis=-1).tolist()
    return vocab.decode(ids)


def swd_decode_ids(logits: np.ndarray, window: int = 1) -> list[int]:
    """Spike Window Decoding -> per-frame token ids (matches realtime_asr.py)."""
    x = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(x)
    probs /= probs.sum(axis=-1, keepdims=True)
    blank_prob = probs[:, BLANK_IDX]
    is_spike = blank_prob < 0.5
    t = probs.shape[0]
    if not is_spike.any():
        return logits.argmax(axis=-1).tolist()
    active = np.zeros(t, dtype=bool)
    for idx in np.nonzero(is_spike)[0]:
        active[max(0, idx - window):min(t, idx + window + 1)] = True
    pred = np.zeros(t, dtype=np.int64)
    pred[active] = logits[active].argmax(axis=-1)
    return pred.tolist()


def make_fixture_waveform(seconds: float) -> np.ndarray:
    """Deterministic pseudo-speech waveform for reproducible parity checks."""
    rng = np.random.default_rng(1106)
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    wave = (
        0.35 * np.sin(2 * np.pi * 140 * t)
        + 0.20 * np.sin(2 * np.pi * 320 * t)
        + 0.12 * np.sin(2 * np.pi * 850 * t)
    )
    wave *= 0.5 * (1 - np.cos(2 * np.pi * np.minimum(t, seconds - t) / seconds + 1e-6))
    wave += 0.02 * rng.standard_normal(n)
    peak = np.abs(wave).max()
    if peak > 0:
        wave = 0.9 * wave / peak
    return wave.astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser(description="Export hiragana ASR to ONNX for Unity Sentis")
    p.add_argument("--model", choices=list(MODEL_PRESETS), default="medium")
    p.add_argument("--checkpoint", type=Path, default=None, help="Override checkpoint path")
    p.add_argument("--pretrained", default=None, help="Override pretrained model id")
    p.add_argument("--output", type=Path, default=None,
                   help="Output .onnx path (default: models/onnx/hiragana_asr_<model>.onnx)")
    p.add_argument("--opset", type=int, default=17,
                   help="ONNX opset (try 15 if Sentis rejects ops)")
    p.add_argument("--dual", action="store_true", help="Also export phoneme (InterCTC) output")
    p.add_argument("--fixed-length", type=float, default=0.0,
                   help="Export with a fixed input length in seconds (0 = dynamic, default)")
    p.add_argument("--fp16", action="store_true", help="Export FP16 weights (skips numeric verify)")
    p.add_argument("--no-bake-norm", action="store_true",
                   help="Do NOT bake normalization into the graph (input must be pre-normalized)")
    p.add_argument("--attn", choices=["eager", "sdpa"], default="eager",
                   help="Attention implementation for export (eager avoids IsNaN/Where ops "
                        "that Unity Sentis may not support; default: eager)")
    p.add_argument("--fixture-audio", type=Path, default=None,
                   help="Real audio file for the Unity parity fixture (resampled to 16 kHz mono)")
    p.add_argument("--verify-lengths", default="0.3,1.0,3.0,10.0",
                   help="Comma-separated seconds to verify ONNX vs PyTorch parity")
    p.add_argument("--tolerance", type=float, default=1e-3, help="Max abs logit diff to accept")
    p.add_argument("--fixture-seconds", type=float, default=0.5,
                   help="Length of the Unity parity fixture waveform")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)

    preset = MODEL_PRESETS[args.model]
    checkpoint = args.checkpoint or ROOT / preset["checkpoint"]
    pretrained = args.pretrained or preset["pretrained"]
    bake_norm = not args.no_bake_norm

    output = args.output or ROOT / "models" / "onnx" / f"hiragana_asr_{args.model}.onnx"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[init] model={args.model} checkpoint={checkpoint}")
    print(f"[init] opset={args.opset} dual={args.dual} bake_norm={bake_norm} "
          f"fixed_length={args.fixed_length or 'dynamic'} fp16={args.fp16}")

    model = load_checkpoint(str(checkpoint), pretrained)
    model.eval()
    try:
        model.encoder.set_attn_implementation(args.attn)
        print(f"[init] attention implementation set to '{args.attn}'")
    except Exception as exc:  # noqa: BLE001
        print(f"[init] WARNING: could not set attn implementation to '{args.attn}': {exc}")

    kana_vocab = KanaVocab()
    phoneme_vocab = PhonemeVocab()

    wrapper: nn.Module = (
        DualOnnxWrapper(model, bake_norm) if args.dual else KanaOnnxWrapper(model, bake_norm)
    )
    wrapper.eval()

    if args.fp16:
        wrapper = wrapper.half()

    # Dummy input for tracing.
    if args.fixed_length > 0:
        trace_len = int(args.fixed_length * SAMPLE_RATE)
        dynamic_axes = None
    else:
        trace_len = int(2.0 * SAMPLE_RATE)
        axes = {0: "batch", 1: "samples"}
        out_axes = {0: "batch", 1: "frames"}
        dynamic_axes = {"audio": axes, "kana_logits": out_axes}
        if args.dual:
            dynamic_axes["phoneme_logits"] = out_axes

    input_dtype = torch.float16 if args.fp16 else torch.float32
    dummy = torch.zeros(1, trace_len, dtype=input_dtype)
    # Feed a non-degenerate waveform so tracing visits real code paths.
    dummy[0] = torch.from_numpy(make_fixture_waveform(trace_len / SAMPLE_RATE)).to(input_dtype)

    output_names = ["kana_logits"] + (["phoneme_logits"] if args.dual else [])

    print("[export] running torch.onnx.export ...")
    export_kwargs = dict(
        input_names=["audio"],
        output_names=output_names,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    if dynamic_axes is not None:
        export_kwargs["dynamic_axes"] = dynamic_axes

    try:
        torch.onnx.export(wrapper, (dummy,), str(output), dynamo=False, **export_kwargs)
    except TypeError:
        # Older torch signatures may not accept dynamo=.
        torch.onnx.export(wrapper, (dummy,), str(output), **export_kwargs)

    print(f"[export] wrote {output} ({output.stat().st_size / 1e6:.1f} MB)")

    # --- Inspect ONNX op set (Sentis compatibility triage) ---
    import onnx

    onnx_model = onnx.load(str(output))
    try:
        onnx.checker.check_model(onnx_model)
        print("[onnx] checker: OK")
    except Exception as exc:  # noqa: BLE001
        print(f"[onnx] checker WARNING: {exc}")

    op_counts = Counter(node.op_type for node in onnx_model.graph.node)
    ops_path = output.with_suffix(".ops.json")
    ops_path.write_text(json.dumps(dict(sorted(op_counts.items())), indent=2), encoding="utf-8")
    print(f"[onnx] {len(op_counts)} distinct op types -> {ops_path.name}")
    print("[onnx] ops: " + ", ".join(f"{k}:{v}" for k, v in sorted(op_counts.items())))

    # --- Write ordered vocab table for C# ---
    id_to_token = [kana_vocab.itos[i] for i in range(kana_vocab.size)]
    vocab_path = output.parent / "kana_vocab.json"
    vocab_path.write_text(
        json.dumps(
            {"blank_index": BLANK_IDX, "blank_token": BLANK_TOKEN, "id_to_token": id_to_token},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[vocab] wrote {vocab_path.name} ({kana_vocab.size} tokens)")

    if args.dual:
        ph_tokens = [phoneme_vocab.itos[i] for i in range(phoneme_vocab.size)]
        ph_path = output.parent / "phoneme_vocab.json"
        ph_path.write_text(
            json.dumps(
                {"blank_index": BLANK_IDX, "id_to_token": ph_tokens},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[vocab] wrote {ph_path.name} ({phoneme_vocab.size} tokens)")

    # --- Numeric parity vs PyTorch (skip for fp16: ORT-CPU fp16 is unreliable) ---
    if args.fp16:
        print("[verify] fp16 export: skipping numeric verification (run a Sentis smoke test).")
    else:
        verify_parity(
            output, wrapper, kana_vocab, args, bake_norm,
            fixed_length=args.fixed_length,
        )

    # --- End-to-end parity fixture for Unity ---
    write_fixture(output, wrapper, kana_vocab, args, bake_norm)

    print("\n[done] Next steps:")
    print("  1. Import the .onnx into Unity (com.unity.sentis) and run the Sentis smoke test.")
    print("  2. Place kana_vocab.json in StreamingAssets.")
    print("  3. Validate against the .parity.json fixture in Unity.")


def run_onnx(session, audio: np.ndarray, bake_norm: bool) -> np.ndarray:
    """Run the ONNX kana model on a (T,) waveform, returning (T', V) logits."""
    inp = audio.astype(np.float32)[None, :]
    if not bake_norm:
        m = inp.mean(axis=-1, keepdims=True)
        v = inp.var(axis=-1, keepdims=True)
        inp = (inp - m) / np.sqrt(v + NORM_EPS)
    outputs = session.run(None, {"audio": inp})
    return outputs[0][0]  # kana_logits, drop batch


def verify_parity(output, wrapper, kana_vocab, args, bake_norm, fixed_length):
    import onnxruntime as ort

    print("[verify] creating onnxruntime session ...")
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])

    lengths = [float(x) for x in args.verify_lengths.split(",") if x.strip()]
    if fixed_length > 0:
        lengths = [fixed_length]

    all_ok = True
    for sec in lengths:
        audio = make_fixture_waveform(sec)
        torch_in = torch.from_numpy(audio)[None, :]
        if not bake_norm:
            torch_in = normalize_waveform(torch_in)
        with torch.inference_mode():
            torch_out = wrapper(torch_in)
            torch_kana = (torch_out[0] if isinstance(torch_out, tuple) else torch_out)[0].numpy()

        onnx_kana = run_onnx(session, audio, bake_norm)

        if torch_kana.shape != onnx_kana.shape:
            print(f"[verify] {sec:>4}s SHAPE MISMATCH "
                  f"torch={torch_kana.shape} onnx={onnx_kana.shape}")
            all_ok = False
            continue

        max_diff = float(np.abs(torch_kana - onnx_kana).max())
        torch_dec = ctc_greedy_decode(torch_kana, kana_vocab)
        onnx_dec = ctc_greedy_decode(onnx_kana, kana_vocab)
        dec_ok = torch_dec == onnx_dec
        ok = max_diff <= args.tolerance and dec_ok
        all_ok &= ok
        flag = "OK " if ok else "FAIL"
        print(f"[verify] {flag} {sec:>4}s frames={torch_kana.shape[0]} "
              f"max_abs_diff={max_diff:.2e} decode_match={dec_ok}")

    if all_ok:
        print("[verify] ALL PASSED (PyTorch vs onnxruntime parity).")
    else:
        print("[verify] WARNING: parity check failed for at least one length.")


def load_fixture_audio(args) -> np.ndarray:
    """Return the fixture waveform: real audio (resampled to 16 kHz mono) or synthetic."""
    if args.fixture_audio is None:
        return make_fixture_waveform(args.fixture_seconds)
    import librosa

    audio, _ = librosa.load(str(args.fixture_audio), sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def write_fixture(output, wrapper, kana_vocab, args, bake_norm):
    audio = load_fixture_audio(args)
    param_dtype = next(wrapper.parameters()).dtype
    torch_in = torch.from_numpy(audio)[None, :]
    norm_in = normalize_waveform(torch_in)[0].numpy()
    model_in = torch_in if bake_norm else normalize_waveform(torch_in)
    model_in = model_in.to(param_dtype)
    with torch.inference_mode():
        out = wrapper(model_in)
        kana_logits = (out[0] if isinstance(out, tuple) else out)[0].float().numpy()

    greedy = ctc_greedy_decode(kana_logits, kana_vocab)
    swd = kana_vocab.decode(swd_decode_ids(kana_logits, window=1))

    fixture = {
        "sample_rate": SAMPLE_RATE,
        "bake_norm": bake_norm,
        "input_name": "audio",
        "source": str(args.fixture_audio) if args.fixture_audio else "synthetic",
        "num_samples": int(audio.shape[0]),
        "raw_waveform": [round(float(x), 6) for x in audio.tolist()],
        "normalized_waveform_head": [round(float(x), 6) for x in norm_in[:16].tolist()],
        "logits_shape": list(kana_logits.shape),
        "logits_head": [round(float(x), 5) for x in kana_logits[0, :8].tolist()],
        "expected_greedy": greedy,
        "expected_swd": swd,
    }
    fixture_path = output.with_suffix(".parity.json")
    fixture_path.write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[fixture] wrote {fixture_path.name} "
          f"(greedy='{greedy[:24]}{'...' if len(greedy) > 24 else ''}')")

    try:
        import soundfile as sf

        wav_path = output.with_suffix(".fixture.wav")
        sf.write(str(wav_path), audio, SAMPLE_RATE)
        print(f"[fixture] wrote {wav_path.name}")
    except Exception as exc:  # noqa: BLE001
        print(f"[fixture] (skipped wav: {exc})")


if __name__ == "__main__":
    main()
