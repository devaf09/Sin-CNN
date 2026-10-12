"""exp_kn.py : 強い n-gram (補間 Kneser-Ney) を相手に、キュー1 (同じサイズでの単独比較) とキュー2 (混合補助) をやり直す。

python exp_kn.py model.npz corpus.txt [--gpu] [--limit 20000]

リザバーは保存済みのモデル (--w-bits 32) の W を量子化・低ランク化して使う (bench と同じ構成)。評価窓と混合の重みの選び方は bench と同じ。
"""
import argparse
import time

import numpy as np

import opticlm as O


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("corpus")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--limit", type=int, default=20000)
    ns = ap.parse_args()
    t0 = time.time()
    docs = O.read_docs(ns.corpus)
    tr, va = O.split_docs(docs)
    rows = []
    for sm in ("kn", "wb"):
        for o in (2, 3, 4):
            for mc in ((1, 2, 3, 5, 10, 20, 50, 100) if sm == "kn" else (1, 2, 10)):
                ng = O.NgramLM(o, mc, smoothing=sm).fit(tr)
                r = ng.eval(va, 4, ns.limit, with_py=True)
                rows.append(dict(name=f"n-gram{sm.upper()} 文脈{o} 剪定{mc}", b4=ng.bytes(True), b8=ng.bytes(False),
                                 kind="ngram" if sm == "kn" else "wb", **r))
        print(f"n-gram ({sm}) 完了 {time.time() - t0:.0f}s", flush=True)
    m = O.load(ns.model, backend="torch" if ns.gpu else "numpy")
    W0 = np.asarray(m.W, dtype=np.float64)
    V = W0.shape[1]
    variants = [("W 8bit", *O.quantize_matrix(W0, 8)), ("W 5bit g256", *O.quantize_matrix(W0, 5, 256)),
                ("W 4bit g256", *O.quantize_matrix(W0, 4, 256))]
    for rk in (32, 64, 128, 256):
        variants.append((f"W ランク{rk}(8bit)", *O.lowrank_matrix(W0, rk, 8)))
    for tag, Wv, nb in variants:
        m.W = Wv
        O.calibrate(m, va, ns.limit)
        r = O.evaluate(m, va, ns.limit, with_py=True)
        rows.append(dict(name=f"リザバー D={m.D} {tag}", b4=nb + 2 * V, b8=nb + 2 * V, kind="res", **r))
    print(f"リザバー完了 {time.time() - t0:.0f}s\n", flush=True)

    print(f"{'構成':<34}{'サイズ(4B/項目)':>14}{'acc1':>7}{'acc5':>7}{'nll':>7}")
    for r in sorted(rows, key=lambda r: r["b4"]):
        print(f"{r['name']:<34}{r['b4'] / 1e6:>12.2f}MB{r['acc1']:>7.1f}{r['acc5']:>7.1f}{r['nll']:>7.3f}")
    kn = [r for r in rows if r["kind"] == "ngram"]
    res = [r for r in rows if r["kind"] == "res"]
    print("\n[単独] 同じかより小さい KN n-gram に acc1・nll とも負ける (支配される) か:")
    for r in res:
        dom = [q for q in kn if q["b4"] <= r["b4"] and q["acc1"] >= r["acc1"] and q["nll"] <= r["nll"]]
        smaller = [q for q in kn if q["b4"] <= r["b4"]]
        best = min(smaller, key=lambda q: q["nll"]) if smaller else None
        print(f"  {r['name']:<30} {r['b4'] / 1e6:6.2f}MB acc1 {r['acc1']:.1f} nll {r['nll']:.3f} -> "
              + ("支配される" if dom else "支配されない")
              + (f" | 同サイズ以下の KN で nll 最良 {best['nll']:.3f} acc1 {best['acc1']:.1f} ({best['name']})" if best else " | 比べる KN なし"))
    # 混合: KN の最良 (nll) を基準に、mix_compare (kind=ngram の行を n-gram として扱う)
    base = min(kn, key=lambda q: q["nll"])
    print(f"\n基準 (KN で nll 最良): {base['name']}")
    O.mix_compare([r for r in rows if r["kind"] in ("ngram", "res")], base_name=base["name"])
    base3 = next(r for r in kn if r["name"] == "n-gramKN 文脈3 剪定1")
    if base3 is not base:
        print(f"\n基準を {base3['name']} にした場合")
        O.mix_compare([r for r in rows if r["kind"] in ("ngram", "res")], base_name=base3["name"])
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
