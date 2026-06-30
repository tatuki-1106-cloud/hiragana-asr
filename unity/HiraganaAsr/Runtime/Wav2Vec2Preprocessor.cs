using System;

namespace HiraganaAsr
{
    /// <summary>
    /// wav2vec2 zero-mean / unit-variance normalization, matching the
    /// HuggingFace feature extractor (<c>do_normalize=true</c>) and
    /// <c>normalize_waveform</c> in <c>scripts/realtime_asr.py</c>:
    /// <c>(x - mean) / sqrt(var + 1e-5)</c> with population variance.
    ///
    /// NOTE: The default ONNX export bakes this normalization into the graph
    /// (<c>--bake-norm</c>), so you only need this class when you export with
    /// <c>--no-bake-norm</c>. Uses a numerically stable two-pass computation
    /// with double accumulation.
    /// </summary>
    public static class Wav2Vec2Preprocessor
    {
        public const float Epsilon = 1e-5f;

        public static float[] Normalize(float[] waveform)
        {
            var result = new float[waveform.Length];
            NormalizeInto(waveform, result);
            return result;
        }

        public static void NormalizeInto(float[] waveform, float[] destination)
        {
            if (destination.Length < waveform.Length)
                throw new ArgumentException("destination is smaller than waveform.");

            int n = waveform.Length;
            if (n == 0) return;

            double mean = 0.0;
            for (int i = 0; i < n; i++) mean += waveform[i];
            mean /= n;

            double variance = 0.0;
            for (int i = 0; i < n; i++)
            {
                double d = waveform[i] - mean;
                variance += d * d;
            }
            variance /= n; // population variance (unbiased = false)

            double invStd = 1.0 / Math.Sqrt(variance + Epsilon);
            for (int i = 0; i < n; i++)
                destination[i] = (float)((waveform[i] - mean) * invStd);
        }
    }
}
