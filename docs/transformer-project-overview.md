# Transformer の基本を知っている人向け hiragana-asr 解説

この資料は、Transformer の基本概念（self-attention、encoder/decoder、CTC の概要）が頭に入っている読者向けに、このリポジトリが何を作っていて、どこに何が実装されているかを短時間で掴めるようにまとめたものです。

## まずこのプロジェクトが何を目指しているか

このプロジェクトは、**「もっとも自然な日本語文を生成する ASR」ではなく、「音声に含まれていた音を、ひらがな列としてできるだけ忠実に写す ASR」** を目指しています。

Whisper のような decoder / LM ベース ASR は、既知語への補完が強く、未知語や固有名詞でハルシネーションしやすいです。  
このプロジェクトはそこを避けるために、**wav2vec2 encoder + CTC** を採用しています。

- 出力は漢字かな混じり文ではなく **ひらがな**
- 意味理解や漢字変換は downstream の LLM に委ねる
- モデル自身は「それっぽい文章生成」をしない

設計思想は `/tmp/workspace/tatuki-1106-cloud/hiragana-asr/DESIGN.md` にまとまっています。

## 全体像

アーキテクチャの中心は `/tmp/workspace/tatuki-1106-cloud/hiragana-asr/src/asr/model.py` です。

```text
音声
  → wav2vec2 feature extractor (CNN)
  → wav2vec2 transformer encoder
      ├─ 中間層 hidden state → phoneme CTC head
      └─ 最終層 hidden state → kana CTC head
```

ポイントは **Dual CTC** です。

- **最終層**: ひらがなを出す主タスク
- **中間層**: 音素を出す補助タスク（InterCTC）

Transformer を知っている人向けに言うと、  
「encoder の final representation だけに全部押し込まず、中間層に音素 supervision を入れて、内部表現を音韻寄りに整える」設計です。

## なぜ decoder なしなのか

このプロジェクトでは decoder を持たないので、Whisper 系と比べると次の違いがあります。

| 観点 | decoder/LM ベース ASR | hiragana-asr |
|---|---|---|
| 何が得意か | 文として自然な出力 | 入力音声への忠実さ |
| 未知語 | 既知語へ引き寄せがち | 音として近い誤りになりやすい |
| 出力 | 漢字かな混じり文 | ひらがな列 + 音素列 |
| ハルシネーション耐性 | 低い | 高い |

つまりこのプロジェクトは、  
**言語モデルの賢さを捨てて、音響モデルとしての素直さを取りに行っている** と捉えると理解しやすいです。

## モデル実装の読み方

### 1. encoder 本体

`create_model()` が HuggingFace の `Wav2Vec2Model` をロードします。

- pretrained encoder は `reazon-research/japanese-wav2vec2-base` または `large`
- CNN feature extractor は基本的に freeze
- transformer encoder 側を fine-tune

ここで重要なのは、このリポジトリが **wav2vec2 を end-to-end で再実装しているわけではなく、既存 encoder を土台に task head を載せている** ことです。

### 2. kana / phoneme の二つの head

`DualCTCModel` は線形層を二つ持ちます。

- `kana_head`
- `phoneme_head`

どちらも hidden state から語彙空間への projection で、decoder は持ちません。  
最終的な文字列化は CTC の collapse に依存します。

### 3. 中間層 supervision

`default_inter_ctc_layer()` は encoder の真ん中付近を選びます。

- base: 12 層なら 6
- large: 24 層なら 12

中間層に音素 supervision を入れることで、後段の kana prediction が音韻情報を利用しやすくなる、というのがこの設計の狙いです。

## 語彙設計

語彙は次のファイルです。

- `/tmp/workspace/tatuki-1106-cloud/hiragana-asr/src/asr/kana_vocab.py`
- `/tmp/workspace/tatuki-1106-cloud/hiragana-asr/src/asr/phoneme_vocab.py`

かな語彙は「日本語文そのもの」ではなく、**CTC で扱うための最小限のひらがな記号集合** です。  
音素語彙は OpenJTalk 系の音素表現に寄っています。

ここでも重要なのは、**語彙サイズを大きくして日本語文の知識を持たせる設計ではない** という点です。

## 学習データがどう流れるか

学習の入口は `/tmp/workspace/tatuki-1106-cloud/hiragana-asr/scripts/01_train.py` です。  
データ関連の中心は `/tmp/workspace/tatuki-1106-cloud/hiragana-asr/src/asr/dataset.py` です。

流れはこうです。

1. ReazonSpeech / JSUT / JVS を前処理してローカル保存
2. 学習時に waveform を読む
3. 転写テキストを kana / phoneme に変換
4. wav2vec2 feature extractor で正規化
5. encoder に通して dual CTC loss を計算

## dataset.py の二つの経路

### `ASRDataset`

生データ寄りの経路です。

- 音声を読む
- 必要なら resample
- augmentation をかける
- テキストから kana / phoneme をその場で作る

つまり **柔軟だが重い** 経路です。

### `PreprocessedASRDataset`

前処理済みデータを読む高速経路です。

- waveform をそのまま読む
- 既に用意済みのラベルを使う
- 追加 augmentation だけ行う

大規模学習ではこちらが主役です。  
Transformer 自体より、**データ供給がボトルネックにならないようにする工夫** がこのリポジトリではかなり重要です。

## loss の見方

損失は `/tmp/workspace/tatuki-1106-cloud/hiragana-asr/src/asr/losses.py` にあります。

README と学習スクリプトから見ると、基本の考え方は次です。

- kana 側が主損失
- phoneme 側が補助損失
- kana 側には CR-CTC を使う

読むときのイメージとしては、

- **phoneme loss**: encoder の中間表現を音韻的に整える
- **kana loss**: 最終的に使いたい出力を直接最適化する
- **CR-CTC**: CTC のフレーム単位出力を少し安定化する

という役割分担です。

## 学習スクリプトで押さえるべき点

`scripts/01_train.py` で見ておくと理解が早いポイントは次です。

### 1. bucket batching

音声長がばらつくので、長さが近いサンプルをまとめて padding を減らしています。  
NLP の fixed-length mini-batch より、音声ではこの種の工夫の価値が大きいです。

### 2. mixed precision

BF16 / FP16 の切り替えがあります。  
大きめの wav2vec2 を回すための実運用上の工夫です。

### 3. encoder の規模拡大

このリポジトリは base から large へ移行しており、既存レポートでも large の改善が大きいです。

- `/tmp/workspace/tatuki-1106-cloud/hiragana-asr/docs/training-report-large-100h.md`
- `/tmp/workspace/tatuki-1106-cloud/hiragana-asr/docs/report-h100-medium-training.md`

つまり研究の主戦場は「decoder を足す」ことではなく、

- encoder を強くする
- データを増やす
- 補助 supervision を改善する

にあります。

## 推論パスの理解

### 単発推論

単発ファイル推論は `/tmp/workspace/tatuki-1106-cloud/hiragana-asr/scripts/03_infer.py` です。

流れは単純です。

1. checkpoint をロード
2. 音声を 16kHz に揃える
3. feature extractor に通す
4. `model()` で `kana_logits` と `phoneme_logits` を得る
5. greedy もしくは SWD で decode

Transformer に慣れている人なら、ここはほぼ「encoder-only ASR の最小構成」と見て問題ありません。

### Spike Window Decoding (SWD)

このプロジェクトの推論で少し特徴的なのが SWD です。

- blank 確率が十分下がるフレームを spike とみなす
- spike 周辺だけを active として decode する

CTC の出力は blank 優勢な時間が長いので、  
**本当に情報が出ている近辺だけを重視して decode する** 発想です。

## リアルタイム推論の全体像

リアルタイム処理は `/tmp/workspace/tatuki-1106-cloud/hiragana-asr/scripts/realtime_asr.py` です。

```text
Mic input
  → VAD
  → 発話切り出し
  → encoder 推論
  → kana 表示
```

ここで重要なのは、リアルタイム性を支えているのが Transformer の工夫だけではないことです。

- Silero VAD で発話境界を取る
- pre-buffer で話し始めの取りこぼしを減らす
- preview / final を分ける
- device ごとに dtype を切り替える

つまり実システムとして見ると、  
**ASR モデル本体 + 音声フロントエンド + セグメンテーション** が一体になっています。

## 既存レポートから見える、このプロジェクトの現在地

既存 docs を読むと、現時点の論点はかなり明確です。

- clean 音声ではかなり強い
- wild / noisy 条件ではまだ落ちる
- 長音、小書き仮名、母音、拗音が難所
- large 化とデータ増加は効く

特に `docs/report-h100-medium-training.md` は、
「このプロジェクトが何を改善軸として見ているか」を把握するのに有用です。

改善の主軸は、

1. データ量
2. encoder 規模
3. 補助タスク設計
4. 推論デコードの安定化

であって、decoder LM を足す方向ではありません。

## コードを読む順番のおすすめ

最短で掴むなら次の順です。

1. `/tmp/workspace/tatuki-1106-cloud/hiragana-asr/README.md`
2. `/tmp/workspace/tatuki-1106-cloud/hiragana-asr/DESIGN.md`
3. `/tmp/workspace/tatuki-1106-cloud/hiragana-asr/src/asr/model.py`
4. `/tmp/workspace/tatuki-1106-cloud/hiragana-asr/src/asr/dataset.py`
5. `/tmp/workspace/tatuki-1106-cloud/hiragana-asr/scripts/01_train.py`
6. `/tmp/workspace/tatuki-1106-cloud/hiragana-asr/scripts/03_infer.py`
7. `/tmp/workspace/tatuki-1106-cloud/hiragana-asr/scripts/realtime_asr.py`
8. 学習レポート群

## コードを見るときの観点（実務で役立つ見方）

単に「何をしているか」を追うより、次の観点で読むと理解と改善提案が速くなります。

1. **設計思想に一貫しているか**
  「音への忠実さを優先する」という方針に沿っているかを確認します。
  具体的には、decoder 的な補完を暗に強める処理が入っていないか、CTC 前提のシンプルさが崩れていないかを見ます。

2. **主タスクと補助タスクの役割分担が明確か**
  `kana_head` と `phoneme_head` が、
  - 最終出力品質の改善
  - 中間表現の音韻的な整形
  のどちらに効いている実装かを切り分けて見ます。
  変更を読むときも「これは kana 側を強くする変更か、intermediate supervision を強くする変更か」で分類すると追いやすいです。

3. **ボトルネックがモデルかデータ供給か**
  音声 ASR では、モデル本体より入出力パイプラインが律速になりやすいです。
  `dataset.py` や batching 周りは、精度だけでなくスループット・GPU 使用率の観点で読むと重要度を見誤りにくくなります。

4. **推論時の安定化ロジックが誤り傾向に対応しているか**
  SWD や VAD 前後の処理は、単なる後処理ではなく誤り分布を変える要素です。
  特に長音・小書き仮名・母音の誤りが、どの段で増減するかを意識してコードを追うと改善ポイントを特定しやすくなります。

5. **オフライン精度とリアルタイム要件のトレードオフ**
  `03_infer.py` と `realtime_asr.py` は目的が異なるため、同じ処理でも最適解が変わります。
  「最終精度優先」か「遅延・取りこぼし最小化優先」かを分けて読むと、実装意図を誤解しにくくなります。

6. **変更の評価軸が明文化されているか**
  変更を読むときは、
  - CER/WER などの精度
  - 難所（長音・拗音など）の改善
  - 速度・メモリ
  のどれを良くする変更かを先に仮説化してから読むと、レビュー品質が上がります。

## 普通の Transformer と何が違うか（処理フロー比較）

ここでは、NLP でよくある「encoder-decoder あるいは decoder-only Transformer」と、
このプロジェクト（wav2vec2 encoder + dual CTC）を処理単位で比較します。

### 1. 入力の違い: token 列ではなく生波形を扱う

一般的な Transformer（NLP）は、最初から離散 token ID を入力します。
一方このプロジェクトは、16kHz の生波形を入力し、まず CNN ベースの
feature extractor で時間方向を圧縮してから Transformer encoder に渡します。

つまり、Transformer 本体の前に

- 音声のサンプリング周波数統一
- モノラル化
- 正規化
- （学習時）速度摂動やノイズ付与

といった音声固有の前処理が必須になります。

この差分を見るときは、`model.py` だけでなく `dataset.py` も同じ重みで読むのが重要です。

### 2. 出力の違い: 自己回帰生成ではなく CTC 一括予測

一般的な Transformer では、次トークン予測を逐次繰り返す自己回帰が中心です。
このプロジェクトは decoder を持たず、encoder の各時刻フレームから CTC ロジットを出して、
blank 除去と collapse で最終列を得ます。

このため推論の性質も変わります。

- 一般 Transformer: 生成ステップ数に応じて遅延が増える
- CTC 系: フレーム列をまとめて推論できるため低遅延化しやすい

ただし CTC 特有の blank 優勢やスパイク分布を扱う必要があり、
デコード戦略（greedy / SWD）の設計が品質に直結します。

### 3. 監督信号の違い: final 層一本ではなく中間層にも supervision

一般的な Transformer の入門実装では、最終層の表現にのみ損失をかけることが多いです。
このプロジェクトでは次の二重監督を使います。

- 最終層 hidden state → kana head（主タスク）
- 中間層 hidden state → phoneme head（補助タスク, InterCTC）

狙いは、最終層にすべての責務を押し込むのではなく、
中間表現を音韻的に整えて最終 kana 予測を助けることです。

ここは「普通の Transformer」と比べたときに最も設計思想の差が出る部分です。

### 4. 損失設計の違い: Cross-Entropy 中心ではなく CR-CTC + InterCTC

NLP の標準的な Transformer は token-level Cross-Entropy が中心です。
このプロジェクトは CTC をベースにしていて、さらに kana 側には CR-CTC を採用しています。

- kana: CR-CTC（CTC + 隣接時刻の整合性正則化）
- phoneme: 標準 CTC
- total: kana_loss + inter_weight * phoneme_loss

CR-CTC は「CTC のスパイク偏重を緩和してフレーム予測を安定化する」方向の工夫です。
これは language modeling 的な賢さを足すのではなく、
音響整合性を上げるための工夫だと理解すると読みやすくなります。

### 5. 推論最適化の違い: 生成探索より spike 活用

一般 Transformer では beam search や sampling 温度など、
生成探索の設計が推論の中心になります。
このプロジェクトでは SWD（Spike Window Decoding）が特徴的で、
blank 確率が落ちたスパイク近傍フレームだけを重視して decode します。

要するに、

- 生成の多様性をどう制御するか

ではなく、

- どの時間フレームに有効情報が出ているか

を使って decode を安定化する設計です。

### 6. 実システム差分: モデル単体より front-end が強く効く

通常のテキスト Transformer では、推論パイプラインは比較的単純です。
リアルタイム ASR では、モデルの前後に

- VAD で発話境界を取る
- pre-buffer で話し始めを保護する
- preview / final を分ける

といった音声フロントエンド制御が必要で、
体感品質はこの層に強く依存します。

したがって `realtime_asr.py` を読むときは、
「Transformer をどう呼んでいるか」だけでなく、
「どのタイミングで音声断片を確定しているか」を主眼にすると理解が早いです。

### 7. コードレビュー時の比較観点（普通の Transformer とのズレを検知する）

変更を見るときは、次をチェックすると本質を外しにくくなります。

1. この変更は CTC 系の素直さを保っているか（decoder 的補完を暗に増やしていないか）
2. 変更の主効果はどこか（最終 kana 精度 / 中間音韻表現 / デコード安定化 / リアルタイム遅延）
3. 改善主張が難所誤り（長音・小書き仮名・母音・拗音）まで落ちているか
4. オフライン評価だけでなく、リアルタイム条件で破綻しないか

## 一言でまとめると

このリポジトリは、

**「Transformer encoder を使った日本語音響モデルを、LM 的な賢さではなく、音への忠実さとリアルタイム性のために最適化しているプロジェクト」**

です。

Transformer の基本を知っている人にとっては、

- encoder-only ASR
- dual supervision
- CTC 系の実運用
- リアルタイム音声フロントエンドとの接続

を具体例として読むと理解しやすいはずです。
