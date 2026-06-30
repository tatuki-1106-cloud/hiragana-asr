using System;
using UnityEngine;

namespace HiraganaAsr
{
    /// <summary>
    /// CTC kana vocabulary, loaded from the <c>kana_vocab.json</c> produced by
    /// <c>scripts/export_onnx.py</c>. The JSON stores an ordered <c>id_to_token</c>
    /// array so token ids map deterministically to characters (index 0 = blank).
    /// </summary>
    public sealed class KanaVocab
    {
        [Serializable]
        private sealed class VocabJson
        {
            public int blank_index;
            public string blank_token;
            public string[] id_to_token;
        }

        public int BlankIndex { get; }
        public string[] IdToToken { get; }
        public int Size => IdToToken.Length;

        private KanaVocab(int blankIndex, string[] idToToken)
        {
            BlankIndex = blankIndex;
            IdToToken = idToToken;
        }

        /// <summary>Parse a vocabulary from a kana_vocab.json string.</summary>
        public static KanaVocab FromJson(string json)
        {
            if (string.IsNullOrEmpty(json))
                throw new ArgumentException("kana_vocab.json content is empty.");

            VocabJson dto = JsonUtility.FromJson<VocabJson>(json);
            if (dto == null || dto.id_to_token == null || dto.id_to_token.Length == 0)
                throw new ArgumentException("kana_vocab.json is missing 'id_to_token'.");

            if (dto.blank_index < 0 || dto.blank_index >= dto.id_to_token.Length)
                throw new ArgumentException(
                    $"kana_vocab.json blank_index {dto.blank_index} is out of range " +
                    $"for {dto.id_to_token.Length} tokens.");

            return new KanaVocab(dto.blank_index, dto.id_to_token);
        }

        /// <summary>Token string for an id, or empty string if out of range.</summary>
        public string TokenAt(int id)
        {
            if (id < 0 || id >= IdToToken.Length) return string.Empty;
            return IdToToken[id];
        }

        public bool IsBlank(int id) => id == BlankIndex;
    }
}
