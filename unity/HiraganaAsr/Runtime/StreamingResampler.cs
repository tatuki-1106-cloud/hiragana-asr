using System;
using System.Collections.Generic;

namespace HiraganaAsr
{
    /// <summary>
    /// Streaming resampler from an arbitrary source rate (typically the 44.1/48 kHz
    /// Unity microphone rate) to the 16 kHz the model expects.
    ///
    /// When downsampling, a stateful 2nd-order Butterworth low-pass filter is applied
    /// before linear interpolation to suppress aliasing of content above the target
    /// Nyquist frequency. Both the filter state and the fractional read position are
    /// carried across calls so chunk boundaries are seamless.
    /// </summary>
    public sealed class StreamingResampler
    {
        private readonly int _srcRate;
        private readonly int _dstRate;
        private readonly bool _passthrough;
        private readonly bool _applyFilter;

        // Biquad low-pass state (Direct Form II transposed).
        private readonly float _b0, _b1, _b2, _a1, _a2;
        private float _z1, _z2;

        // Linear interpolation state.
        private double _pos;          // fractional position in the filtered source stream
        private float _lastSample;    // last filtered source sample from previous chunk
        private bool _hasLast;
        private readonly double _step; // srcRate / dstRate

        public StreamingResampler(int srcRate, int dstRate)
        {
            _srcRate = srcRate;
            _dstRate = dstRate;
            _passthrough = srcRate == dstRate;
            _applyFilter = srcRate > dstRate; // anti-alias only matters when downsampling
            _step = (double)srcRate / dstRate;

            if (_applyFilter)
            {
                // Butterworth low-pass, cutoff = 0.45 * dstRate (just under Nyquist).
                double cutoff = 0.45 * dstRate;
                double w0 = 2.0 * Math.PI * cutoff / srcRate;
                double cosW0 = Math.Cos(w0);
                double sinW0 = Math.Sin(w0);
                double alpha = sinW0 / (2.0 * 0.70710678); // Q = 1/sqrt(2)

                double b0 = (1.0 - cosW0) / 2.0;
                double b1 = 1.0 - cosW0;
                double b2 = (1.0 - cosW0) / 2.0;
                double a0 = 1.0 + alpha;
                double a1 = -2.0 * cosW0;
                double a2 = 1.0 - alpha;

                _b0 = (float)(b0 / a0);
                _b1 = (float)(b1 / a0);
                _b2 = (float)(b2 / a0);
                _a1 = (float)(a1 / a0);
                _a2 = (float)(a2 / a0);
            }
            else
            {
                _b0 = _b1 = _b2 = _a1 = _a2 = 0f;
            }
        }

        /// <summary>Resample a chunk of source-rate mono samples to 16 kHz.</summary>
        public float[] Process(float[] src, int count)
        {
            if (count <= 0) return Array.Empty<float>();
            if (_passthrough)
            {
                var copy = new float[count];
                Array.Copy(src, copy, count);
                return copy;
            }

            // Low-pass filter the incoming samples (stateful), only when downsampling.
            var filtered = new float[count];
            if (_applyFilter)
            {
                for (int i = 0; i < count; i++)
                {
                    float x = src[i];
                    float y = _b0 * x + _z1;
                    _z1 = _b1 * x - _a1 * y + _z2;
                    _z2 = _b2 * x - _a2 * y;
                    filtered[i] = y;
                }
            }
            else
            {
                Array.Copy(src, filtered, count);
            }

            var output = new List<float>((int)(count / _step) + 2);

            if (!_hasLast)
            {
                _lastSample = filtered[0];
                _hasLast = true;
                _pos = 0.0;
            }

            // _pos is measured relative to the previous chunk's last sample = index -1.
            // Walk output positions, interpolating between filtered[idx-1] and filtered[idx].
            while (true)
            {
                int idx = (int)Math.Floor(_pos) + 1; // upper source index in current chunk
                if (idx >= count) break;

                double frac = _pos - Math.Floor(_pos);
                float a = idx - 1 < 0 ? _lastSample : filtered[idx - 1];
                float b = filtered[idx];
                output.Add((float)(a + (b - a) * frac));
                _pos += _step;
            }

            // Carry remainder past the chunk into the next call.
            _pos -= count;
            _lastSample = filtered[count - 1];

            return output.ToArray();
        }

        public void Reset()
        {
            _z1 = _z2 = 0f;
            _pos = 0.0;
            _hasLast = false;
            _lastSample = 0f;
        }
    }
}
