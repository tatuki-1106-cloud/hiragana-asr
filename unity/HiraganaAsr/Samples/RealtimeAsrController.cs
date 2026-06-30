using System.Collections.Generic;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.Events;

namespace HiraganaAsr.Samples
{
    /// <summary>UnityEvent carrying the recognized hiragana string (inspector-serializable).</summary>
    [System.Serializable]
    public sealed class HiraganaStringEvent : UnityEvent<string> { }

    /// <summary>
    /// End-to-end real-time hiragana ASR for Unity, mirroring
    /// <c>scripts/realtime_asr.py</c>: microphone -> VAD segmentation -> Sentis
    /// inference -> CTC decode -> finalized hiragana line.
    ///
    /// Setup:
    ///  1. Install the Sentis package (com.unity.sentis 2.1+ or com.unity.ai.inference).
    ///  2. Import the exported .onnx as a ModelAsset and assign it below.
    ///  3. Assign kana_vocab.json (as a TextAsset) below.
    ///  4. Add this component to a GameObject and press Play.
    /// </summary>
    public sealed class RealtimeAsrController : MonoBehaviour
    {
        [Header("Model")]
        [SerializeField] private ModelAsset modelAsset;
        [SerializeField] private TextAsset vocabJson;
        [SerializeField] private BackendType backend = BackendType.GPUCompute;
        [SerializeField] private CtcDecodeMode decodeMode = CtcDecodeMode.Swd;
        [SerializeField] private int swdWindow = 1;
        [Tooltip("Leave true unless the model was exported with --no-bake-norm.")]
        [SerializeField] private bool bakeNorm = true;

        [Header("Microphone")]
        [Tooltip("Empty = default device.")]
        [SerializeField] private string microphoneDevice = "";
        [SerializeField] private int requestedSampleRate = 48000;

        [Header("Segmentation")]
        [SerializeField] private float silenceTimeoutSec = 0.8f;
        [SerializeField] private float maxUtteranceSec = 12.0f;
        [SerializeField] private float prebufferSec = 0.5f;
        [SerializeField] private float speechFactor = 3.0f;
        [SerializeField] private float minUtteranceSec = 0.3f;

        [Header("Events")]
        public HiraganaStringEvent onUtteranceFinalized = new HiraganaStringEvent();

        private MicrophoneCapture _mic;
        private EnergyVadSegmenter _vad;
        private HiraganaAsrModel _model;
        private KanaVocab _vocab;
        private readonly List<string> _finalized = new List<string>();

        /// <summary>All finalized hiragana lines so far.</summary>
        public IReadOnlyList<string> Finalized => _finalized;

        private void Start()
        {
            if (modelAsset == null || vocabJson == null)
            {
                Debug.LogError("[HiraganaAsr] Assign both modelAsset and vocabJson.");
                enabled = false;
                return;
            }

            _vocab = KanaVocab.FromJson(vocabJson.text);
            _model = new HiraganaAsrModel(modelAsset, _vocab, backend)
            {
                BakeNorm = bakeNorm,
                DecodeMode = decodeMode,
                SwdWindow = swdWindow,
            };

            _vad = new EnergyVadSegmenter(silenceTimeoutSec, maxUtteranceSec, prebufferSec, speechFactor);

            _mic = new MicrophoneCapture();
            try
            {
                _mic.StartCapture(string.IsNullOrEmpty(microphoneDevice) ? null : microphoneDevice,
                    requestedSampleRate);
                Debug.Log($"[HiraganaAsr] Capturing at {_mic.SourceSampleRate} Hz -> 16 kHz. Speak!");
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[HiraganaAsr] Microphone start failed: {e.Message}");
                enabled = false;
            }
        }

        private void Update()
        {
            if (_mic == null || !_mic.IsCapturing) return;

            float[] samples = _mic.PollNewSamples();
            if (samples.Length == 0) return;

            if (_vad.Feed(samples, out float[] utterance))
            {
                float dur = utterance.Length / 16000f;
                if (dur < minUtteranceSec) return;

                string kana = _model.Transcribe(utterance);
                if (!string.IsNullOrEmpty(kana))
                {
                    _finalized.Add(kana);
                    onUtteranceFinalized.Invoke(kana);
                    Debug.Log($"[HiraganaAsr] ({dur:F1}s) {kana}");
                }
            }
        }

        private void OnDestroy()
        {
            _mic?.StopCapture();
            _model?.Dispose();
        }
    }
}
