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
        /// </summary>
        public string Transcribe(float[] audio16k)
        {
            if (audio16k == null || audio16k.Length == 0) return string.Empty;

            float[] input = BakeNorm ? audio16k : Wav2Vec2Preprocessor.Normalize(audio16k);

            using var inputTensor = new Tensor<float>(new TensorShape(1, input.Length), input);
            _worker.Schedule(inputTensor);

            var output = _worker.PeekOutput("kana_logits") as Tensor<float>;
            if (output == null) return string.Empty;

            TensorShape shape = output.shape; // (1, frames, vocabSize)
            int frames = shape[1];
            int vocabSize = shape[2];

            // Blocking readback to CPU.
            float[] logits = output.DownloadToArray();

            return CtcDecoder.Decode(logits, frames, vocabSize, _vocab, DecodeMode, SwdWindow);
        }

        public void Dispose()
        {
            _worker?.Dispose();
        }
    }
}
