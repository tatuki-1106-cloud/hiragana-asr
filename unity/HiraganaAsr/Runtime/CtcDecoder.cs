using System;
using System.Text;

namespace HiraganaAsr
{
    public enum CtcDecodeMode
    {
        Greedy,
        Swd, // Spike Window Decoding
    }

    /// <summary>
    /// CTC decoder for kana logits. Mirrors <c>src/asr/kana_vocab.py</c> (greedy
    /// collapse) and the Spike Window Decoding (SWD) used in
    /// <c>scripts/realtime_asr.py</c>.
    /// </summary>
    public static class CtcDecoder
    {
        /// <summary>
        /// Decode flattened logits of shape (frames * vocabSize), row-major.
        /// </summary>
        public static string Decode(
            float[] logits, int frames, int vocabSize, KanaVocab vocab,
            CtcDecodeMode mode = CtcDecodeMode.Swd, int swdWindow = 1)
        {
            int[] ids = mode == CtcDecodeMode.Swd
                ? SwdFrameIds(logits, frames, vocabSize, vocab.BlankIndex, swdWindow)
                : ArgmaxFrameIds(logits, frames, vocabSize);
            return CollapseToString(ids, vocab);
        }

        /// <summary>Per-frame argmax token ids.</summary>
        public static int[] ArgmaxFrameIds(float[] logits, int frames, int vocabSize)
        {
            var ids = new int[frames];
            for (int t = 0; t < frames; t++)
                ids[t] = Argmax(logits, t * vocabSize, vocabSize);
            return ids;
        }

        /// <summary>
        /// Spike Window Decoding: keep argmax only on frames near a CTC spike
        /// (blank probability &lt; 0.5), force blank elsewhere.
        /// </summary>
        public static int[] SwdFrameIds(
            float[] logits, int frames, int vocabSize, int blankIndex, int window)
        {
            var active = new bool[frames];
            bool anySpike = false;

            for (int t = 0; t < frames; t++)
            {
                int baseIdx = t * vocabSize;
                // softmax blank probability for this frame
                float max = float.NegativeInfinity;
                for (int v = 0; v < vocabSize; v++)
                {
                    float val = logits[baseIdx + v];
                    if (val > max) max = val;
                }
                double sum = 0.0;
                for (int v = 0; v < vocabSize; v++)
                    sum += Math.Exp(logits[baseIdx + v] - max);
                double blankProb = Math.Exp(logits[baseIdx + blankIndex] - max) / sum;

                if (blankProb < 0.5)
                {
                    anySpike = true;
                    int start = Math.Max(0, t - window);
                    int end = Math.Min(frames - 1, t + window);
                    for (int k = start; k <= end; k++) active[k] = true;
                }
            }

            if (!anySpike)
                return ArgmaxFrameIds(logits, frames, vocabSize);

            var ids = new int[frames];
            for (int t = 0; t < frames; t++)
                ids[t] = active[t] ? Argmax(logits, t * vocabSize, vocabSize) : blankIndex;
            return ids;
        }

        /// <summary>CTC collapse: drop blanks and consecutive repeats, map to chars.</summary>
        public static string CollapseToString(int[] frameIds, KanaVocab vocab)
        {
            var sb = new StringBuilder(frameIds.Length);
            int prev = -1;
            foreach (int id in frameIds)
            {
                if (vocab.IsBlank(id)) { prev = id; continue; }
                if (id == prev) continue;
                string token = vocab.TokenAt(id);
                if (!string.IsNullOrEmpty(token)) sb.Append(token);
                prev = id;
            }
            return sb.ToString();
        }

        private static int Argmax(float[] data, int offset, int count)
        {
            int best = 0;
            float bestVal = data[offset];
            for (int i = 1; i < count; i++)
            {
                float val = data[offset + i];
                if (val > bestVal) { bestVal = val; best = i; }
            }
            return best;
        }
    }
}
