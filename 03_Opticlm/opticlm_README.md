# opticlm — 光学リザバー言語モデル

固定の「光学媒質」で文脈を特徴にし、学習は線形の読み出しだけ(閉形式)にした小さな言語モデルのライブラリです。
単一ファイル `opticlm.py`(numpy のみで動作。torch があれば GPU でも動作)。

## 位置づけ (2026-10-13 に改訂。根拠の数字は RESULTS.md)

このモデルは **物理実装 (光・音) の研究用モデル** です。言語モデルとしての性能や軽さは主張しません。

- **できること (検証済み)**
  - 学習が足し算なので、追加学習・取り消し・融合が厳密にできる (誤差 1e-16)。CPU の複数プロセスで分けて学習して足すこともできる (`fit_parallel`)。
  - 光学の特徴は、同じ特徴数のデジタルのランダム特徴 (ランダムフーリエ特徴) と同等か、わずかに良い。「光で作れるランダム特徴」として使える。
    回折 (位相板と伝播) は必須で、無いと大きく悪化する。
  - **粗いアナログ素子で動く**: 読み出しの重み W は 4〜5bit (露光ごと+バイアス別のゲインを持つグループ量子化) で、素子のばらつき 10% でも
    nll の悪化は約 0.01。検出器は光子 100/画素・ADC 6bit で足りる。全部を同時にかけても nll +0.04 程度 (D=16385)。
- **主張しないこと**
  - **軽さ・固定メモリの優位**。同じバイト数では、n-gram (特に Kneser-Ney) が単独のリザバーにほぼ全構成で勝ちます。
    10M 文字で、13MB のリザバー (D=16385, W 8bit) は acc1 35.0 / nll 3.227、5.5MB の n-gram (Kneser-Ney 文脈3, 剪定2) は 40.4 / 2.875。
    データを10倍にしたときの伸びも n-gram の方が大きい (リザバーは読み出しの容量 D×語彙 で頭打ちになる)。
  - **n-gram の混合補助としての優位**。弱い n-gram (Witten-Bell) 相手では、小さなリザバーを混ぜると同じバイト数の n-gram より効いたが、
    Kneser-Ney 相手では差が無くなった。残るのは、合計 1〜3MB の小さな予算で強く剪定した n-gram を補う場合だけ (未確定)。
  - 特徴の増強 (参照光・多面検出・長距離 bag)、2段カスケードの効果。実日本語ではどれも効かなかった。
- 追加学習・取り消し・融合は、数える方式の n-gram でもできます。リザバーに固有なのは「物理で作れること」と「粗い素子への強さ」です。

## 導入

```
pip install numpy            # 必須
pip install torch            # GPU を使うとき (CUDA 版は pytorch.org の手順で)
```

`opticlm.py` を作業フォルダに置くだけです。

## 最初にやること (この順で)

```
python opticlm.py selftest                                   # 1. 動作確認 (GPU のあるマシンで)
python opticlm.py ablate cc100_v1m.txt --d 4096 --gpu        # 2. 特徴の増強オプションの効果を1表で
python opticlm.py bench  cc100_v1m.txt --d 2048,4096,8192,16384 --gpu    # 3. 同じバイト数での n-gram 比較
```

- 2 は約 12 通りの設定を同じ総特徴数で学習して並べます(GPU で十数分の見込み。開発側では実測できていません)。
- 3 の出力は「推論に必要なバイト数 vs 精度」の表、リザバーと n-gram を混ぜたときの NLL、混合の公平な比較 (`mix_compare`:
  n-gram にリザバーを足す場合と、同じバイト数の別の n-gram を足す場合の比較) です。`--ng-min 1,2,3,5,10,20,50,100` で
  小さい n-gram の比較点を増やせます。
- ablate は `--limit 0` で検証窓を全部使えます。`--variants 1,4` で比べる構成を絞れます (0 = 基本構成は常に含まれます)。

## Python から使う

```python
import opticlm
lm = opticlm.OpticalLM(grid=64, scales=(1, 2, 3, 4), backend="auto")   # auto: GPU があれば GPU
lm.fit(["文書1", "文書2"])          # 追加学習 (リストの1要素=1文書)
lm.solve(1e-3)                      # 読み出しを解く
opticlm.calibrate(lm, val_docs)     # 確率の較正 (検証用の文書で1回)
lm.predict("今日は", k=5)           # [(文字, 確率), ...]
lm.generate("今日は", 80, temperature=0.8, forbid="〓")
lm.forget("取り消す文章")           # fit と同じ文章を渡す
lm.save("state.npz")                # 追加学習できる形
lm.save("infer.npz", stats=False, w_bits=8)   # 推論専用 (W を 8bit に)
lm.save("infer4.npz", stats=False, w_bits=4, w_group=256)   # 4bit + 256行ごとのスケール (ほぼ無劣化)
lm = opticlm.load("state.npz", backend="auto")

lm, report = opticlm.train("corpus.txt", grid=64, scales=(1, 2, 3, 4))   # 分割・学習・正則化選択・較正まで一括
```

`opticlm_example.py` に動く例があります。

| 関数・メソッド | 内容 |
|---|---|
| `OpticalLM(...)` | モデル。`backend="numpy"/"torch"/"auto"` |
| `.fit(texts, weight=1)` / `.forget(texts)` | 追加学習 / 取り消し (足し算・引き算) |
| `.solve(lam_rel)` | 読み出し W を解く |
| `.merge(other)` | 別モデルの統計量を足す (cfg が同じものだけ) |
| `calibrate(lm, val)` / `evaluate(lm, val)` | 確率の較正 (softmax / power / binpow から交差当てはめの nll で選ぶ) / 検証 (acc1, acc5, nll) |
| `.predict` `.next_proba` `.nll` `.generate` | 予測・確率・生成 |
| `.save(path, stats, w_bits, w_group)` / `load(path)` | 保存 (w_bits: 32/16/8/6/5/4/3。w_group=256 でグループ量子化。4bit は w_group 必須と考えてよい) |
| `.infer_bytes(w_bits)` | 推論用の重みのバイト数の見積もり |
| `train(path, ...)` | ファイルから一括 (1行=1文書) |
| `NgramLM(order, min_count)` | 比較用の n-gram (サイズを数えられる) |
| `size_bench(...)` / `ablate(...)` | 上の bench / ablate |

## 物理設定と増強オプション

| 引数 | 意味 | 光学での対応 |
|---|---|---|
| `grid`, `scales` | 露光ごとの画素数 / 各露光が見る直近の文字数 | SLM の画素数、同時に使う独立な光路の数 |
| `tails=((T, γ[, "bag"]),)` | 長距離の露光。直近 T 文字を γ^k の振幅で重ねる | 損失のある空洞での積分 |
| `ref` | 参照光の強さ。>0 で `|U + ref·R|²` を読む | ホログラフィの干渉検出 (線形項も得る) |
| `taps` | 強度を読む層 (例 `(2,3)`) | 途中の面にもカメラを置く |
| `detector` | `photons`, `read`, `adc_bits` | 検出器の雑音・量子化の模擬 |

**`tails` の `"bag"`**: 位置ごとの位相板を使わず、文字の『集まり』(話題)だけを重ねます。遠い文字は
位置を区別しても雑音になるだけなので、長距離は `bag` + `ref` の組合せが理にかなう、という設計です。
合成データ(文書ごとに偏った文字集合)では、位置つきの長距離露光は効かず、bag + ref で acc5 が大きく伸びました。
**実際の日本語 (CC-100, D=4096) では、ref・taps・tails・bag のどれも効果がありませんでした (acc ±0.5 以内)。既定はすべてオフです。**

## 検証状況

確認済み(開発側・numpy):
- 追加学習の一括一致、取り消し、融合: 相対誤差 1e-16 程度(全オプション入りの設定でも)
- 既定設定の特徴が 08 の式と 1e-16 で一致、保存/読み込み、W の量子化保存
- n-gram の実装が 08 の実装と同じ数字を出す、リザバーと同じ評価窓を使う

- GPU 実機 (RTX 3060, torch 2.14 + CUDA 12.6) で selftest は全て OK。既定の `acc="fp32"` では、取り消し・融合の誤差が
  1e-6〜1e-7 程度になります (厳密さが要るなら `acc="fp64"`)。

未確認:
- 検出器の雑音がある場合の取り消しは近似になる(再測定のたびに雑音が違うため)。
- 実機の光学・音響での再現。読み出し W(D×語彙)の物理実装は未解決で、ここはデジタルのままです。

## 注意

- 評価の誤差: 検証窓 2万件で acc1 の標準誤差は約 ±0.3 ポイント。それ以内の差は意味を持たせないでください。
- `train(..., refit_all=True)` や CLI の `--refit` は検証文書も学習に足すので、そのモデルの検証値は甘くなります。
- コーパスは自分で用意してください(`fetch_cc100_sample.py`, `prepare_corpus.py`)。
