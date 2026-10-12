"""opticlm の使用例。  python opticlm_example.py [corpus.txt]   (省略時は組み込みの短い文で動作確認)
GPU があれば backend="auto" で自動的に GPU を使う。"""
import sys

import opticlm

path = sys.argv[1] if len(sys.argv) > 1 else None

if path:
    # 1行=1文書のファイルから: 分割 -> 学習 -> 正則化の選択 -> 確率の較正 まで一括
    lm, rep = opticlm.train(path, grid=32, scales=(1, 2, 3, 4), backend="auto")
    print(f"検証 acc1 {rep['acc1']:.1f}  acc5 {rep['acc5']:.1f}  nll {rep['nll']:.3f}")
else:
    docs = ["吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。" * 6,
            "今日はいい天気ですね。散歩に出かけましょう。公園の桜がきれいです。" * 6,
            "明日は雨が降るでしょう。傘を持って出かけてください。電車が遅れるかもしれません。" * 6]
    lm = opticlm.OpticalLM(grid=16, scales=(1, 2, 3), backend="auto")
    lm.fit(docs[:2])
    lm.solve(1e-3)
    opticlm.calibrate(lm, docs[2:])

print(lm)
print("次の文字の候補:", lm.predict("今日は", k=5))
print("生成:", lm.generate("今日は", 40, temperature=0.8))

# --- 追加学習 / 取り消し / 融合 (どれも再学習なし。足し算・引き算だけ) ---
extra = "追加で教える文章です。" * 10
lm.fit(extra)            # 追加学習
lm.solve(1e-3)           # 解き直し
lm.forget(extra)         # 取り消し (fit のときと同じ文章を渡す)
lm.solve(1e-3)

# --- 保存 ---
lm.save("example_state.npz")                           # 統計量つき (追加学習できる)
lm.save("example_infer.npz", stats=False, w_bits=8)    # 推論専用 (W を 8bit に)
small = opticlm.load("example_infer.npz")
print("8bit 版の予測:", small.predict("今日は", k=3))
