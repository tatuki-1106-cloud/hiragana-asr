"""Inference demo for kana ASR with Spike Window Decoding (SWD).

Supports file input. Outputs kana sequence (+ optional phoneme from InterCTC).

Usage:
    uv run python scripts/03_infer.py --audio data/test.wav --checkpoint models/checkpoints/best.pt
    uv run python scripts/03_infer.py --audio data/test.wav --checkpoint ... --swd
"""

# ruff: noqa: E402

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.asr.kana_vocab import KanaVocab
from src.asr.model import load_checkpoint
from src.asr.phoneme_vocab import PhonemeVocab

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def swd_decode(logits: torch.Tensor, window: int = 1) -> torch.Tensor:
    """Spike Window Decoding (SWD) for CTC.

    Instead of argmax over all frames, only considers frames around CTC spikes.
    This reduces computation and can improve accuracy.

    Reference: ICASSP 2025 — "Spike Window Decoding"

    Args:
        logits: (1, T, V) raw logits from the model.
        window: Number of frames on each side of the spike to consider.

    Returns:
        Predicted token indices (1D tensor).
    """
    # Step 1: Find spikes (frames where non-blank is most likely)
    probs = logits.squeeze(0).softmax(dim=-1)  # (T, V)
    blank_prob = probs[:, 0]  # (T,)

    # A spike is where blank probability drops below 0.5
    is_spike = blank_prob < 0.5  # (T,)

    if not is_spike.any():
        # No spikes: return standard greedy decode
        return logits.squeeze(0).argmax(dim=-1)

    # Step 2: Expand spike positions by window
    T = probs.shape[0]
    spike_indices = is_spike.nonzero(as_tuple=True)[0]

    active = torch.zeros(T, dtype=torch.bool, device=logits.device)
    for idx in spike_indices:
        start = max(0, idx.item() - window)
        end = min(T, idx.item() + window + 1)
        active[start:end] = True

    # Step 3: Decode only active frames, set inactive to blank (0)
    pred_ids = torch.zeros(T, dtype=torch.long, device=logits.device)
    pred_ids[active] = logits.squeeze(0)[active].argmax(dim=-1)

    return pred_ids


def parse_args():
    p = argparse.ArgumentParser(description="Kana ASR inference with SWD")
    p.add_argument("--audio", required=True, type=Path, help="Audio file path")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--pretrained", default="reazon-research/japanese-wav2vec2-base")
    p.add_argument("--inter-ctc-layer", type=int, default=None)
    p.add_argument("--swd", action="store_true", help="Enable Spike Window Decoding")
    p.add_argument("--swd-window", type=int, default=1, help="SWD window size")
    p.add_argument("--show-phonemes", action="store_true", help="Also show InterCTC phonemes")
    p.add_argument("--fp16", action="store_true", help="Use FP16 inference (recommended for MPS)")
    return p.parse_args()


def main():
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    log.info(f"Loading model from {args.checkpoint}")
    model = load_checkpoint(
        str(args.checkpoint), args.pretrained,
        inter_ctc_layer=args.inter_ctc_layer,
    )
    if args.fp16:
        model.half()
    model.to(device)
    model.eval()

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.pretrained)
    kana_vocab = KanaVocab()
    phoneme_vocab = PhonemeVocab()

    # Load audio
    log.info(f"Loading audio: {args.audio}")
    waveform, sr = torchaudio.load(str(args.audio))

    # Resample to 16kHz if needed
    if sr != 16_000:
        resampler = torchaudio.transforms.Resample(sr, 16_000)
        waveform = resampler(waveform)
        sr = 16_000

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    audio_array = waveform.squeeze(0).numpy()
    duration = len(audio_array) / sr

    log.info(f"Audio duration: {duration:.2f}s")

    # Inference
    inputs = feature_extractor(
        audio_array,
        sampling_rate=sr,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device)
    if args.fp16:
        input_values = input_values.half()

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_values, attention_mask=attention_mask)
        kana_logits = outputs["kana_logits"]  # (1, T, kana_V)

        if args.swd:
            kana_pred_ids = swd_decode(kana_logits, window=args.swd_window)
        else:
            kana_pred_ids = kana_logits.squeeze(0).argmax(dim=-1)

    t1 = time.perf_counter()

    kana = kana_vocab.decode(kana_pred_ids.tolist())
    inference_time = t1 - t0
    rtf = inference_time / duration

    print(f"\nKana: {kana}")
    print(f"Inference: {inference_time * 1000:.1f}ms (RTF: {rtf:.3f})")
    if args.swd:
        print(f"Decoding: SWD (window={args.swd_window})")

    if args.show_phonemes:
        phoneme_logits = outputs["phoneme_logits"]
        phoneme_pred_ids = phoneme_logits.squeeze(0).argmax(dim=-1)
        phonemes = phoneme_vocab.decode(phoneme_pred_ids.tolist())
        print(f"Phonemes (InterCTC): {phonemes}")


if __name__ == "__main__":
    main()
