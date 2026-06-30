using System;
using Unity.Sentis;
using UnityEngine;

namespace HiraganaAsr.Samples
{
    /// <summary>
    /// One-shot validation that the imported ONNX model runs in Sentis and
    /// produces the same hiragana as the Python reference. Assign the model, the
    /// vocab, and the <c>*.parity.json</c> fixture emitted by
    /// <c>scripts/export_onnx.py</c>, then press Play and read the Console.
    ///
    /// This is the recommended first step after importing the model: it proves
    /// Sentis can execute the graph and that decoding matches PyTorch end to end.
    /// </summary>
    public sealed class SentisSmokeTest : MonoBehaviour
    {
        [SerializeField] private ModelAsset modelAsset;
        [SerializeField] private TextAsset vocabJson;
        [SerializeField] private TextAsset parityFixtureJson;
        [SerializeField] private BackendType backend = BackendType.GPUCompute;

        [Serializable]
        private sealed class Fixture
        {
            public bool bake_norm;
            public int num_samples;
            public float[] raw_waveform;
            public int[] logits_shape;
            public string expected_greedy;
            public string expected_swd;
        }

        private void Start()
        {
            if (modelAsset == null || vocabJson == null || parityFixtureJson == null)
            {
                Debug.LogError("[SmokeTest] Assign modelAsset, vocabJson and parityFixtureJson.");
                return;
            }

            Fixture fx = JsonUtility.FromJson<Fixture>(parityFixtureJson.text);
            if (fx == null || fx.raw_waveform == null || fx.raw_waveform.Length == 0)
            {
                Debug.LogError("[SmokeTest] Could not parse the parity fixture.");
                return;
            }

            var vocab = KanaVocab.FromJson(vocabJson.text);
            using var model = new HiraganaAsrModel(modelAsset, vocab, backend)
            {
                BakeNorm = fx.bake_norm,
            };

            model.DecodeMode = CtcDecodeMode.Greedy;
            string greedy = model.Transcribe(fx.raw_waveform);

            model.DecodeMode = CtcDecodeMode.Swd;
            string swd = model.Transcribe(fx.raw_waveform);

            bool greedyOk = greedy == fx.expected_greedy;
            bool swdOk = swd == fx.expected_swd;

            Debug.Log($"[SmokeTest] greedy expected='{fx.expected_greedy}' got='{greedy}' -> {(greedyOk ? "PASS" : "FAIL")}");
            Debug.Log($"[SmokeTest] swd    expected='{fx.expected_swd}' got='{swd}' -> {(swdOk ? "PASS" : "FAIL")}");

            if (greedyOk && swdOk)
                Debug.Log("[SmokeTest] ALL PASSED - Sentis output matches the PyTorch reference.");
            else
                Debug.LogWarning("[SmokeTest] Mismatch. Check backend/precision, opset support, and resampling.");
        }
    }
}
