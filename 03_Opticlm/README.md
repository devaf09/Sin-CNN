# opticlm — 光学リザバー言語モデル

固定の「光学媒質」（乱数の位相板＋自由空間伝播＋強度検出）で直近の数文字を特徴にします。学習するのは線形の読み出しだけで、閉形式のリッジ回帰で解きます。
本体は単一ファイル `opticlm.py` です。numpy だけで動き、torch があれば GPU でも動きます。

光の部分は学習しないので、作り直さずに物理（光学素子）で再現できる可能性があります。
このフォルダは、その性質が「言語モデルとして役に立つか」「物理で作れるか」を検証した結果をまとめたものです。

## 結論（2026-10-13。CC-100 日本語 1000万文字で検証）

**言語モデルとしては、同じバイト数の n-gram に勝てませんでした。** 優位と言えるのは、物理実装に関わる性質だけです。

### 優位と言えること

| 内容 | 根拠の数字 | 比較相手と注意 |
|---|---|---|
| 追加学習・取り消し・融合・並列学習が厳密 | 誤差 1e-16（numpy）、GPU（fp32 積算）1e-6〜1e-7 | 勾配で学習する光ニューラルネットにはできない。ただし数える方式の n-gram でもできる |
| 光学の特徴は、デジタルのランダム特徴に見劣りしない | 10M, D=4097：光学 acc1 31.2 / nll 3.417、ランダムフーリエ特徴 30.9 / 3.429 | 差は標準誤差程度。「光だから良い」とは言えない。回折を外すと大きく悪化（24.8 → 16.7, D=1025） |
| 粗いアナログ素子で動く | W 5bit：nll +0.007、4bit：+0.025、素子ばらつき 10%：+0.014、検出器 光子100/画素＋ADC 6bit：+0.016、全部同時で約 +0.04 | 量子化の手法（グループごとのスケール）は既知。D が小さい（4097）と 4bit で +0.07 |

### 優位と言えないこと（不利な点）

| 内容 | 根拠の数字 |
|---|---|
| 同じバイト数での言語モデルの性能 | Kneser-Ney n-gram（文脈3, 5.5MB）acc1 40.4 / nll 2.875 に対し、リザバー（D=16385, 13MB）は 35.0 / 3.227。ほぼ全構成で支配される |
| データを増やしたときの伸び | 1M → 10M 文字で、n-gram は acc1 +3.6、リザバーは +0.8。読み出しの容量（D×語彙）で頭打ちになる |
| n-gram の混合補助 | 弱い n-gram（Witten-Bell）相手では効いたが、Kneser-Ney 相手では同じバイト数の n-gram を足すのと差がない |
| 特徴の増強（参照光・多面検出・長距離 bag）、2段カスケード（高次の相互作用）、確信度での混合の切り替え | いずれも効果なし、または悪化（RESULTS.md） |
| 学習の速さ | GPU で 1000万文字・D=16385 に約14分。CPU は GPU の約50倍遅い |

詳しい数字、実験の手順、うまくいかなかった実験はすべて [RESULTS.md](RESULTS.md) にあります。リポジトリ全体の自己批判は [REVIEW.md](REVIEW.md) にまとめています。

## 導入

```
pip install numpy
pip install torch      # GPU を使うとき（CUDA 版は pytorch.org の手順で）
```

## 使い方

```python
import opticlm
lm = opticlm.OpticalLM(grid=32, scales=(1, 2, 3, 4), backend="auto")   # auto: GPU があれば GPU
lm.fit(["文書1", "文書2"])          # 追加学習（リストの1要素＝1文書）
lm.solve(1e-3)                      # 読み出しを解く
opticlm.calibrate(lm, val_docs)     # 確率の較正（検証用の文書で1回）
lm.predict("今日は", k=5)           # [(文字, 確率), ...]
lm.generate("今日は", 80, temperature=0.8, forbid="〓")
lm.forget("取り消す文章")           # fit と同じ文章を渡すと厳密に取り消せる
lm.merge(other_lm)                  # 同じ物理設定のモデルの統計量を足す
lm.save("state.npz")                                     # 追加学習できる形
lm.save("infer.npz", stats=False, w_bits=4, w_group=256) # 推論専用（4bit＋グループごとのスケール）
lm = opticlm.load("state.npz", backend="auto")

lm, report = opticlm.train("corpus.txt", grid=64)        # ファイル（1行＝1文書）から分割・学習・較正まで
opticlm.fit_parallel(lm, docs, workers=3, threads=2)     # CPU の複数プロセスで学習して足す（優先度を下げて動く）
```

`example.py` に、そのまま動く例があります。

```
python opticlm.py selftest                                   # 動作確認（厳密性・GPU 版との照合）
python opticlm.py train corpus.txt model.npz --grid 64 --gpu # 学習（--workers 3 で CPU 並列）
python opticlm.py eval  model.npz corpus.txt
python opticlm.py gen   model.npz "今日は" --n 100
python opticlm.py bench corpus.txt --d 4096,16384 --gpu      # 同じバイト数での n-gram（既定 Kneser-Ney）との比較と、混合の比較
```

## 結果の再現

コーパスは各自で取得してください。CC-100 は Common Crawl 由来なので、本文はこのリポジトリに含めていません。

```
python tools/fetch_cc100_sample.py cc100_ja_10m.txt --chars 10000000           # 先頭の約8MB（圧縮）だけ取得
python tools/prepare_corpus.py cc100_ja_10m.txt cc100_v10m.txt --min-count 1000 # 整形（低頻度の文字を〓に）
python opticlm.py train cc100_v10m.txt model_10m_d16k.npz --grid 64 --no-stats --gpu   # 約15分（RTX 3060）
```

| 主張 | コマンド | RESULTS.md の節 |
|---|---|---|
| 同じバイト数で n-gram（KN）に負ける、混合補助の優位もない | `python experiments/exp_kn.py model_10m_d16k.npz cc100_v10m.txt --gpu` | 強い n-gram（補間 Kneser-Ney）でキュー1・2をやり直し |
| 同上（リザバーも学習し直す版） | `python opticlm.py bench cc100_v10m.txt --d 8192,16384 --gpu` | キュー1、キュー2 |
| 光学の特徴 ≈ ランダム特徴、回折は必須、カスケードは効かない | `python experiments/exp_features.py cc100_v10m.txt --d 4096 --modes optics,nodiff,cascade:0.5,rff:1,prod` | アイディア C3 |
| 4〜5bit・素子ばらつき・検出器雑音に強い | `python experiments/exp_robust.py model_10m_d16k.npz cc100_v10m.txt --gpu` | キュー6、量子化の頑健性 |
| 量子化の方式の比較（4bit が壊れる原因） | `python experiments/exp_quant.py model_10m_d16k.npz cc100_v10m.txt --gpu` | キュー6 |
| seed を変えても再現（CPU 並列で学習） | `python experiments/exp_seed_cpu.py cc100_v1m.txt --seeds 1,2 --d 4096` | 量子化の頑健性 (5) |
| 較正 binpow の効果 | `python experiments/exp_calib.py dump ...` → `run2` | キュー4 |
| 確信度での混合の切り替えは効かない | `python experiments/exp_mixgate.py scores.npz cc100_v10m.txt` | アイディア C1 |

検証は「7文書に1つを検証用」に分けて行っています（文書単位なので、同じ文書の文が学習と検証の両方に入ることはありません）。
窓の数が2万のとき、acc1 の標準誤差は約 ±0.3 ポイントです。それ以内の差には意味を持たせていません。

## 物理設定

| 引数 | 意味 | 光学での対応 |
|---|---|---|
| `grid`, `scales` | 露光ごとの画素数（一辺）/ 各露光が見る直近の文字数 | SLM・カメラの画素数、同時に使う光路の数 |
| `layers`, `dist`, `wavelength`, `pitch` | 位相板の枚数、層間距離、波長、画素ピッチ | 位相板（回折素子）の積層 |
| `seed` | 位相板と文字のホログラムの乱数の種 | 製造した素子そのもの（保存不要） |
| `detector` | `photons`, `read`, `adc_bits` | 検出器の雑音・量子化の模擬 |

総特徴数は D = 露光数 × grid² + 1 です。読み出し W は D × 語彙の行列で、これだけがデジタルで保存される部分です。
W を4〜5bit にするときは `w_group=256`（または露光の画素数）を指定します。列全体で1つのスケールにすると、バイアス行の大きさに引っぱられて 4bit 以下で壊れます。

## 注意

- **GPU メモリ**: A（D×D, float64）は D=16385 で 2.1GB、D=24337 で 4.7GB です。分解に足りないときは CPU（float64）で解きます。
- **CPU 並列** (`fit_parallel`): 子プロセスごとに A を持つので、メモリは「プロセス数 × A」が必要です。
- **検出器の雑音**: 雑音（photons / read）を入れると、取り消しは近似になります（測るたびに雑音が違うため）。
- **以前の版との違い**: 増強オプション（tails / ref / taps / bag）と ablate コマンドは、効果がなかったので削除しました。以前の版はコミット ec67335 にあります。
