using System;
using UnityEngine;

namespace HiraganaAsr
{
    /// <summary>
    /// Captures microphone audio and streams it as 16 kHz mono float samples.
    ///
    /// Unity's <see cref="Microphone"/> usually records at 44.1/48 kHz, so each
    /// polled block is resampled to 16 kHz with <see cref="StreamingResampler"/>.
    /// Call <see cref="StartCapture"/> once, then <see cref="PollNewSamples"/>
    /// every frame (e.g. from <c>Update</c>).
    /// </summary>
    public sealed class MicrophoneCapture
    {
        private const int TargetRate = 16000;
        private const int ClipLengthSec = 5;

        private string _device;
        private AudioClip _clip;
        private int _clipSamples;
        private int _channels;
        private int _readHead;
        private StreamingResampler _resampler;

        public bool IsCapturing { get; private set; }
        public int SourceSampleRate { get; private set; }

        /// <summary>Begin recording. Pass null for the default device.</summary>
        public void StartCapture(string device = null, int requestedRate = 48000)
        {
            if (Microphone.devices.Length == 0)
                throw new InvalidOperationException("No microphone devices available.");

            _device = device ?? Microphone.devices[0];

            Microphone.GetDeviceCaps(_device, out int minFreq, out int maxFreq);
            int rate = requestedRate;
            if (maxFreq > 0) rate = Mathf.Clamp(rate, minFreq == 0 ? rate : minFreq, maxFreq);

            _clip = Microphone.Start(_device, true, ClipLengthSec, rate);
            if (_clip == null)
                throw new InvalidOperationException($"Failed to start microphone '{_device}'.");

            SourceSampleRate = _clip.frequency;
            _channels = Mathf.Max(1, _clip.channels);
            _clipSamples = _clip.samples; // per channel
            _readHead = 0;
            _resampler = new StreamingResampler(SourceSampleRate, TargetRate);
            IsCapturing = true;
        }

        /// <summary>
        /// Return any new microphone audio resampled to 16 kHz mono, or an empty
        /// array if nothing new is available yet.
        /// </summary>
        public float[] PollNewSamples()
        {
            if (!IsCapturing || _clip == null) return Array.Empty<float>();

            int pos = Microphone.GetPosition(_device); // per-channel frame index
            if (pos < 0) return Array.Empty<float>();

            int available = pos - _readHead;
            if (available < 0) available += _clipSamples; // wrapped
            if (available == 0) return Array.Empty<float>();

            // Guard against buffer overrun (we fell behind a full ring).
            if (available > _clipSamples) available = _clipSamples;

            // GetData fills the whole array, so size it to exactly the new audio,
            // reading `available` frames per channel starting at _readHead (wrapping).
            var interleaved = new float[available * _channels];
            _clip.GetData(interleaved, _readHead);
            _readHead = (_readHead + available) % _clipSamples;

            // Downmix to mono.
            var mono = new float[available];
            if (_channels == 1)
            {
                Array.Copy(interleaved, mono, available);
            }
            else
            {
                for (int i = 0; i < available; i++)
                {
                    float sum = 0f;
                    int baseIdx = i * _channels;
                    for (int c = 0; c < _channels; c++) sum += interleaved[baseIdx + c];
                    mono[i] = sum / _channels;
                }
            }

            return _resampler.Process(mono, mono.Length);
        }

        public void StopCapture()
        {
            if (!IsCapturing) return;
            Microphone.End(_device);
            IsCapturing = false;
            _clip = null;
            _resampler?.Reset();
        }
    }
}
