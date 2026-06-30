using System;
using Unity.Sentis;
using UnityEngine;

namespace HiraganaAsr
{
    /// <summary>
    /// Runs the exported hiragana ASR ONNX model with Unity Sentis and decodes
    /// the CTC kana output to a string.
    ///
    /// The default export (<c>scripts/export_onnx.py</c>) bakes wav2vec2
    /// normalization into the graph, so feed raw microphone samples in [-1, 1].
    /// If you exported with <c>--no-bake-norm</c>, set <see cref="BakeNorm"/> to
    /// false so this wrapper normalizes the audio first.
    ///
    /// NOTE: Sentis was renamed to "Inference Engine" (com.unity.ai.inference).
    /// If you use that package, replace <c>using Unity.Sentis;</c> with
    /// <c>using Unity.InferenceEngine;</c>.
    /// </summary>
    public sealed class HiraganaAsrModel : IDisposable
    {
        private readonly Model _model;
        private readonly Worker _worker;
        private readonly KanaVocab _vocab;

        public bool BakeNorm { get; set; } = true;
        public CtcDecodeMode DecodeMode { get; set; } = CtcDecodeMode.Swd;
        public int SwdWindow { get; set; } = 1;

        private bool _vocabValidated;
        private bool _busy;             // an inference is scheduled/awaiting readback
        private bool _disposeRequested; // Dispose was called while busy; defer it
        private bool _disposed;

        /// <summary>True while an inference is scheduled or awaiting readback.</summary>
        public bool IsBusy => _busy;

        public HiraganaAsrModel(
            ModelAsset modelAsset, KanaVocab vocab,
            BackendType backend = BackendType.GPUCompute)
        {
            if (modelAsset == null) throw new ArgumentNullException(nameof(modelAsset));
            _vocab = vocab ?? throw new ArgumentNullException(nameof(vocab));
            _model = ModelLoader.Load(modelAsset);
            _worker = new Worker(_model, backend);
        }

        /// <summary>
        /// Transcribe a 16 kHz mono utterance to a hiragana string.
        /// Blocking: <c>DownloadToArray</c> stalls the main thread until the GPU
        /// finishes. Prefer <see cref="TranscribeAsync"/> for live/interactive use.
        /// </summary>
        public string Transcribe(float[] audio16k)
        {
            if (_disposed || _disposeRequested) return string.Empty;
            if (audio16k == null || audio16k.Length == 0) return string.Empty;
            if (_busy)
            {
                Debug.LogWarning("[HiraganaAsr] Transcribe called while busy; ignored.");
                return string.Empty;
            }

            _busy = true;
            try
            {
                float[] input = BakeNorm ? audio16k : Wav2Vec2Preprocessor.Normalize(audio16k);

                using var inputTensor = new Tensor<float>(new TensorShape(1, input.Length), input);
                _worker.Schedule(inputTensor);

                var output = _worker.PeekOutput("kana_logits") as Tensor<float>;
                if (output == null) return string.Empty;

                TensorShape shape = output.shape; // (1, frames, vocabSize)
                int frames = shape[1];
                int vocabSize = shape[2];
                if (!ValidateVocab(vocabSize)) return string.Empty;

                // Blocking readback to CPU.
                float[] logits = output.DownloadToArray();

                return CtcDecoder.Decode(logits, frames, vocabSize, _vocab, DecodeMode, SwdWindow);
            }
            finally
            {
                _busy = false;
                if (_disposeRequested && !_disposed) DisposeWorkerNow();
            }
        }

        /// <summary>
        /// Non-blocking transcription: schedules inference and awaits an async GPU
        /// readback so the calling frame is not stalled. Use this for live preview
        /// and the per-utterance path in interactive scenes. The model serializes
        /// itself (one inference in flight); concurrent calls return empty.
        /// </summary>
        public async Awaitable<string> TranscribeAsync(float[] audio16k)
        {
            if (_disposed || _disposeRequested) return string.Empty;
            if (audio16k == null || audio16k.Length == 0) return string.Empty;
            if (_busy)
            {
                Debug.LogWarning("[HiraganaAsr] TranscribeAsync called while busy; ignored.");
                return string.Empty;
            }

            _busy = true;
            try
            {
                float[] input = BakeNorm ? audio16k : Wav2Vec2Preprocessor.Normalize(audio16k);

                using var inputTensor = new Tensor<float>(new TensorShape(1, input.Length), input);
                _worker.Schedule(inputTensor);

                var output = _worker.PeekOutput("kana_logits") as Tensor<float>;
                if (output == null) return string.Empty;

                TensorShape shape = output.shape; // (1, frames, vocabSize)
                int frames = shape[1];
                int vocabSize = shape[2];
                if (!ValidateVocab(vocabSize)) return string.Empty;

                using var cpuLogits = await output.ReadbackAndCloneAsync();
                float[] logits = cpuLogits.DownloadToArray(); // readback done: non-blocking

                return CtcDecoder.Decode(logits, frames, vocabSize, _vocab, DecodeMode, SwdWindow);
            }
            finally
            {
                _busy = false;
                // If Dispose() was requested while we were awaiting, free the worker now
                // that the in-flight inference and its readback have completed.
                if (_disposeRequested && !_disposed) DisposeWorkerNow();
            }
        }

        /// <summary>
        /// One-time guard that the loaded vocab matches the model's output classes.
        /// A mismatched kana_vocab.json otherwise decodes to silent garbage.
        /// </summary>
        private bool ValidateVocab(int vocabSize)
        {
            if (_vocabValidated) return true;

            if (vocabSize != _vocab.Size)
            {
                Debug.LogError(
                    $"[HiraganaAsr] Vocab/model mismatch: model outputs {vocabSize} classes " +
                    $"but kana_vocab.json has {_vocab.Size}. Wrong vocab file for this model?");
                return false;
            }
            if (_vocab.BlankIndex < 0 || _vocab.BlankIndex >= vocabSize)
            {
                Debug.LogError(
                    $"[HiraganaAsr] blank_index {_vocab.BlankIndex} is out of range " +
                    $"for vocab size {vocabSize}.");
                return false;
            }

            _vocabValidated = true;
            return true;
        }

        /// <summary>
        /// Disposes the worker. If an inference is in flight, disposal is deferred
        /// until that inference and its async readback complete, so the GPU readback
        /// never touches a freed worker.
        /// </summary>
        public void Dispose()
        {
            if (_disposed) return;
            if (_busy)
            {
                _disposeRequested = true;
                return;
            }
            DisposeWorkerNow();
        }

        private void DisposeWorkerNow()
        {
            _disposed = true;
            _worker?.Dispose();
        }
    }
}
