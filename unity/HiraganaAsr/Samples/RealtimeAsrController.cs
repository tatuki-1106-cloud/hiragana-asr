using System;
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

        [Header("Live preview")]
        [Tooltip("Re-decode the in-progress utterance for incremental display (matches realtime_asr.py).")]
        [SerializeField] private bool enablePreview = true;
        [SerializeField] private float previewIntervalSec = 0.3f;
        [SerializeField] private float previewMinDeltaSec = 0.4f;
        [Tooltip("Cap preview decode to the last N seconds (0 = whole utterance).")]
        [SerializeField] private float previewMaxAudioSec = 6.0f;

        [Header("Events")]
        public HiraganaStringEvent onUtteranceFinalized = new HiraganaStringEvent();
        [Tooltip("Provisional, may rewrite as more audio arrives; prefixed with … when clipped.")]
        public HiraganaStringEvent onPreviewUpdated = new HiraganaStringEvent();

        private MicrophoneCapture _mic;
        private EnergyVadSegmenter _vad;
        private HiraganaAsrModel _model;
        private KanaVocab _vocab;
        private readonly List<string> _finalized = new List<string>();

        // Single-inference scheduling state.
        private bool _inflight;
        private readonly Queue<float[]> _pendingFinals = new Queue<float[]>();
        private int _utteranceSerial;     // bumped whenever an utterance finalizes
        private float _lastPreviewTime;
        private int _lastPreviewSamples;
        private const int SampleRate = 16000;

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
            if (samples.Length > 0 && _vad.Feed(samples, out float[] utterance))
            {
                // A new utterance boundary invalidates any in-flight/queued preview.
                _utteranceSerial++;
                _lastPreviewTime = 0f;
                _lastPreviewSamples = 0;

                float dur = utterance.Length / (float)SampleRate;
                if (dur >= minUtteranceSec)
                    _pendingFinals.Enqueue(utterance); // finals must never be dropped
            }

            TryDispatch();
        }

        /// <summary>
        /// Drive at most one inference at a time on the shared worker. Final
        /// utterances take priority and are never dropped; live preview is
        /// best-effort and discarded if the utterance changed while busy.
        /// </summary>
        private async void TryDispatch()
        {
            if (_inflight || _model == null) return;

            // 1) Final utterances (highest priority, queued so none are lost).
            //    A while-loop (not recursion) drains the queue so synchronous
            //    early-returns from TranscribeAsync can't grow the stack.
            while (_pendingFinals.Count > 0)
            {
                if (_model == null) return;
                float[] u = _pendingFinals.Dequeue();
                _inflight = true;
                try
                {
                    string kana = await _model.TranscribeAsync(u);
                    if (_model != null && !string.IsNullOrEmpty(kana))
                    {
                        _finalized.Add(kana);
                        onUtteranceFinalized.Invoke(kana);
                        Debug.Log($"[HiraganaAsr] ({u.Length / (float)SampleRate:F1}s) {kana}");
                    }
                }
                catch (Exception e)
                {
                    Debug.LogError($"[HiraganaAsr] final transcription failed: {e.Message}");
                }
                finally { _inflight = false; }
            }

            // 2) Live preview (provisional; safe to skip).
            if (!enablePreview || _model == null) return;

            float now = Time.time;
            if (now - _lastPreviewTime < previewIntervalSec) return;

            float[] current = _vad.GetCurrentAudio();
            if (current == null || current.Length / (float)SampleRate < minUtteranceSec) return;
            if (current.Length - _lastPreviewSamples < previewMinDeltaSec * SampleRate) return;

            float[] clip = current;
            bool clipped = false;
            int maxSamples = (int)(previewMaxAudioSec * SampleRate);
            if (previewMaxAudioSec > 0f && current.Length > maxSamples)
            {
                clip = new float[maxSamples];
                Array.Copy(current, current.Length - maxSamples, clip, 0, maxSamples);
                clipped = true;
            }

            int serial = _utteranceSerial;
            _lastPreviewTime = now;
            _lastPreviewSamples = current.Length;
            _inflight = true;
            try
            {
                string kana = await _model.TranscribeAsync(clip);
                // Discard if the utterance finalized or rolled over while we decoded.
                if (_model != null && serial == _utteranceSerial && _pendingFinals.Count == 0
                    && !string.IsNullOrEmpty(kana))
                {
                    onPreviewUpdated.Invoke(clipped ? "…" + kana : kana);
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"[HiraganaAsr] preview transcription failed: {e.Message}");
            }
            finally { _inflight = false; }
        }

        private void OnDestroy()
        {
            _mic?.StopCapture();
            // Null the field first so any in-flight TranscribeAsync that resumes
            // after disposal skips event invocation via its `_model != null` guard.
            var model = _model;
            _model = null;
            model?.Dispose();
        }
    }
}
