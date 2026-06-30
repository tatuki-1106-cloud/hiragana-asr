using System;
using System.Collections.Generic;

namespace HiraganaAsr
{
    /// <summary>
    /// Energy-based voice-activity segmenter operating on 16 kHz mono audio.
    ///
    /// This mirrors the segmentation behaviour of <c>scripts/realtime_asr.py</c>
    /// (rolling pre-buffer prepended on speech onset, finalize after a silence
    /// timeout, force-split overly long utterances) but replaces the Silero VAD
    /// with an RMS detector that tracks an adaptive noise floor. It is dependency
    /// free and robust in quiet rooms; for noisy environments consider exporting
    /// Silero VAD to ONNX as a drop-in upgrade (see docs/unity-integration.md).
    /// </summary>
    public sealed class EnergyVadSegmenter
    {
        private const int SampleRate = 16000;
        private const int FrameSize = 512; // 32 ms, matches the Silero chunk size

        private readonly int _silenceSamples;
        private readonly int _maxSamples;
        private readonly int _prebufferMax;
        private readonly int _warmupSamples;
        private readonly float _speechFactor;
        private readonly float _absoluteFloor;
        private readonly float _noiseAdapt;

        private readonly List<float> _utterance = new List<float>();
        private readonly Queue<float[]> _prebuffer = new Queue<float[]>();
        private int _prebufferLen;
        private float[] _leftover = Array.Empty<float>();

        private bool _isSpeaking;
        private int _silenceCount;
        private int _warmupRemaining;
        private bool _noiseInitialized;
        private float _noiseFloor = 1e-4f;

        public EnergyVadSegmenter(
            float silenceTimeoutSec = 0.8f,
            float maxUtteranceSec = 12.0f,
            float prebufferSec = 0.5f,
            float speechFactor = 3.0f,
            float absoluteFloor = 0.0025f,
            float noiseAdaptRate = 0.05f,
            float warmupSec = 0.3f)
        {
            _silenceSamples = (int)(silenceTimeoutSec * SampleRate);
            _maxSamples = (int)(maxUtteranceSec * SampleRate);
            _prebufferMax = (int)(prebufferSec * SampleRate);
            _warmupSamples = (int)(warmupSec * SampleRate);
            _speechFactor = speechFactor;
            _absoluteFloor = absoluteFloor;
            _noiseAdapt = noiseAdaptRate;
            _warmupRemaining = _warmupSamples;
        }

        public void Reset()
        {
            _utterance.Clear();
            _prebuffer.Clear();
            _prebufferLen = 0;
            _leftover = Array.Empty<float>();
            _isSpeaking = false;
            _silenceCount = 0;
            _warmupRemaining = _warmupSamples;
            _noiseInitialized = false;
            _noiseFloor = 1e-4f;
        }

        /// <summary>
        /// Feed new 16 kHz samples. Returns true and outputs a finalized utterance
        /// when an utterance boundary is reached; otherwise returns false.
        /// </summary>
        public bool Feed(float[] chunk, out float[] utterance)
        {
            utterance = null;

            float[] data;
            if (_leftover.Length == 0)
            {
                data = chunk;
            }
            else
            {
                data = new float[_leftover.Length + chunk.Length];
                Array.Copy(_leftover, 0, data, 0, _leftover.Length);
                Array.Copy(chunk, 0, data, _leftover.Length, chunk.Length);
            }
            _leftover = Array.Empty<float>();

            int pos = 0;
            while (pos + FrameSize <= data.Length)
            {
                var frame = new float[FrameSize];
                Array.Copy(data, pos, frame, 0, FrameSize);
                float rms = Rms(frame);

                // Calibrate the noise floor before allowing any speech detection.
                // Without this, detection latched on the initial 1e-4 floor could
                // treat room tone as speech and never finalize.
                if (!_noiseInitialized)
                {
                    _noiseFloor = rms;
                    _noiseInitialized = true;
                }
                if (_warmupRemaining > 0)
                {
                    _warmupRemaining -= FrameSize;
                    _noiseFloor += 0.3f * (rms - _noiseFloor);
                    PushPrebuffer(frame);
                    pos += FrameSize;
                    continue;
                }

                float speechThreshold = Math.Max(_absoluteFloor, _noiseFloor * _speechFactor);
                float releaseThreshold = Math.Max(_absoluteFloor * 0.6f, _noiseFloor * _speechFactor * 0.5f);
                bool isSpeech = _isSpeaking ? rms >= releaseThreshold : rms >= speechThreshold;

                if (isSpeech)
                {
                    if (!_isSpeaking)
                    {
                        _isSpeaking = true;
                        DrainPrebufferInto(_utterance);
                    }
                    _silenceCount = 0;
                    _utterance.AddRange(frame);
                }
                else
                {
                    // Adapt the noise floor only during non-speech frames.
                    _noiseFloor += _noiseAdapt * (rms - _noiseFloor);

                    if (_isSpeaking)
                    {
                        _silenceCount += FrameSize;
                        _utterance.AddRange(frame);

                        if (_silenceCount >= _silenceSamples)
                        {
                            utterance = _utterance.ToArray();
                            ResetUtterance();
                            _leftover = Slice(data, pos + FrameSize);
                            return true;
                        }
                    }
                    else
                    {
                        PushPrebuffer(frame);
                    }
                }

                if (_utterance.Count >= _maxSamples)
                {
                    utterance = _utterance.ToArray();
                    ResetUtterance();
                    _leftover = Slice(data, pos + FrameSize);
                    return true;
                }

                pos += FrameSize;
            }

            if (pos < data.Length)
                _leftover = Slice(data, pos);

            return false;
        }

        /// <summary>In-progress utterance audio for live preview, or null.</summary>
        public float[] GetCurrentAudio()
        {
            if (!_isSpeaking || _utterance.Count == 0) return null;
            return _utterance.ToArray();
        }

        private void ResetUtterance()
        {
            _utterance.Clear();
            _isSpeaking = false;
            _silenceCount = 0;
        }

        private void PushPrebuffer(float[] frame)
        {
            _prebuffer.Enqueue(frame);
            _prebufferLen += frame.Length;
            while (_prebufferLen > _prebufferMax && _prebuffer.Count > 0)
                _prebufferLen -= _prebuffer.Dequeue().Length;
        }

        private void DrainPrebufferInto(List<float> target)
        {
            while (_prebuffer.Count > 0)
                target.AddRange(_prebuffer.Dequeue());
            _prebufferLen = 0;
        }

        private static float[] Slice(float[] data, int start)
        {
            int len = data.Length - start;
            if (len <= 0) return Array.Empty<float>();
            var result = new float[len];
            Array.Copy(data, start, result, 0, len);
            return result;
        }

        private static float Rms(float[] frame)
        {
            double sum = 0.0;
            for (int i = 0; i < frame.Length; i++) sum += (double)frame[i] * frame[i];
            return (float)Math.Sqrt(sum / frame.Length);
        }
    }
}
