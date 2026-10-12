"""exp_quant.py : W の低ビット量子化の改善 (キュー6)。

python exp_quant.py model.npz corpus.txt [--gpu] [--bits 4,5,6,8] [--limit 20000]

model.npz は --w-bits 32 で保存したもの (量子化前の W が要る)。各方式で W を量子化し、較正し直して検証の acc1/acc5/nll を出す。
方式 (どれもクラス=列ごとに扱う。バイト数は W の符号+スケール+外れ値の保存を数える):
  col      : 列ごとの対称量子化 (現行。スケールは列の最大絶対値)
  clipP    : 列ごとのスケールを |W| の P パーセンタイルにして、超える値はクリップ
  groupG   : 各列を G 行ずつのグループに分け、グループごとにスケール (スケールは fp16)
  outK     : 列ごとに |W| の上位 K% を fp16 で別保存 (位置 4B + 値 2B)、残りを列ごとに量子化
  rowcol   : 行ごとのスケール r_i (fp16) で割ってから列ごとに量子化 (特徴ごとの大きさの違いを吸収)
"""
import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # opticlm.py は1つ上
import opticlm as O


def q_sym(W, bits, scale):
    qmax = 2 ** (bits - 1) - 1
    scale = np.where(scale == 0, 1.0, scale)
    return np.clip(np.rint(W / scale), -qmax, qmax) * scale


def m_col(W, bits):
    qmax = 2 ** (bits - 1) - 1
    return q_sym(W, bits, np.abs(W).max(0, keepdims=True) / qmax), W.size * bits / 8 + 4 * W.shape[1]


def m_clip(W, bits, pct):
    qmax = 2 ** (bits - 1) - 1
    return q_sym(W, bits, np.percentile(np.abs(W), pct, axis=0, keepdims=True) / qmax), W.size * bits / 8 + 4 * W.shape[1]


def m_group(W, bits, G):
    qmax = 2 ** (bits - 1) - 1
    D, V = W.shape
    pad = (-D) % G
    Wp = np.vstack([W, np.zeros((pad, V))]).reshape(-1, G, V)
    sc = np.abs(Wp).max(1, keepdims=True) / qmax
    sc = sc.astype(np.float16).astype(np.float64)
    Wq = q_sym(Wp, bits, sc).reshape(-1, V)[:D]
    return Wq, W.size * bits / 8 + 2 * sc.size


def m_out(W, bits, frac):
    D, V = W.shape
    k = max(1, int(round(D * frac)))
    A = np.abs(W)
    thr = -np.partition(-A, k - 1, axis=0)[k - 1:k]  # 列ごとの k 番目の大きさ
    out = A >= thr
    Wr = np.where(out, 0.0, W)
    qmax = 2 ** (bits - 1) - 1
    Wq = q_sym(Wr, bits, np.abs(Wr).max(0, keepdims=True) / qmax)
    Wq = np.where(out, W.astype(np.float16).astype(np.float64), Wq)
    return Wq, W.size * bits / 8 + 4 * V + int(out.sum()) * 6


def m_rowcol(W, bits):
    r = np.sqrt((W ** 2).mean(1, keepdims=True))
    r = np.where(r == 0, 1.0, r).astype(np.float16).astype(np.float64)
    Wn = W / r
    qmax = 2 ** (bits - 1) - 1
    Wq = q_sym(Wn, bits, np.abs(Wn).max(0, keepdims=True) / qmax) * r
    return Wq, W.size * bits / 8 + 4 * W.shape[1] + 2 * W.shape[0]


def m_bias(W, bits, G=0):
    """最後の行 (バイアス項: 特徴が定数1) だけ fp16 で別に持ち、残りを列ごと (G>0 ならグループごと) に量子化。"""
    body, last = W[:-1], W[-1:]
    Wq, nb = m_group(body, bits, G) if G else m_col(body, bits)
    return np.vstack([Wq, last.astype(np.float16).astype(np.float64)]), nb + 2 * W.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("corpus")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--bits", default="3,4,5,6,8")
    ap.add_argument("--limit", type=int, default=20000)
    ns = ap.parse_args()
    m = O.load(ns.model, backend="torch" if ns.gpu else "numpy")
    _, va = O.split_docs(O.read_docs(ns.corpus))
    W0 = np.asarray(m.W, dtype=np.float64)
    D, V = W0.shape
    # 外れ値の様子: 列ごとの 最大|W| / 標準偏差
    ratio = np.abs(W0).max(0) / (W0.std(0) + 1e-30)
    print(f"{ns.model}: D={D} V={V} | 列ごとの 最大/標準偏差: 中央値 {np.median(ratio):.1f} 90% {np.percentile(ratio, 90):.1f} "
          f"最大 {ratio.max():.1f} | 行ごとの rms の 最大/中央値 {np.sqrt((W0**2).mean(1)).max() / np.median(np.sqrt((W0**2).mean(1))):.1f}")
    rr = np.sqrt((W0 ** 2).mean(1))
    top = np.argsort(-rr)[:5]
    print(f"  rms の大きい行: {list(map(int, top))} (最後の行 = バイアス項 は {D - 1}) 値 {[round(float(rr[i] / np.median(rr)), 1) for i in top]}")
    rows = []

    def ev(name, Wq, nb):
        t0 = time.time()
        m.W = Wq
        O.calibrate(m, va, ns.limit)
        r = O.evaluate(m, va, ns.limit)
        rel = float(np.linalg.norm(Wq - W0) / np.linalg.norm(W0))
        rows.append((name, nb, r["acc1"], r["acc5"], r["nll"], rel))
        print(f"{name:<22}{nb / 1e6:>8.2f}MB{r['acc1']:>7.1f}{r['acc5']:>7.1f}{r['nll']:>8.3f}  相対誤差 {rel:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    print(f"{'方式':<22}{'サイズ':>10}{'acc1':>7}{'acc5':>7}{'nll':>8}")
    ev("fp32 (基準)", W0, W0.size * 4)
    for b in [int(x) for x in ns.bits.split(",")]:
        ev(f"{b}bit col(現行)", *m_col(W0, b))
        for p in (99.9, 99.0):
            ev(f"{b}bit clip{p}", *m_clip(W0, b, p))
        for G in (64, 256):
            ev(f"{b}bit group{G}", *m_group(W0, b, G))
        ev(f"{b}bit out0.5%", *m_out(W0, b, 0.005))
        ev(f"{b}bit rowcol", *m_rowcol(W0, b))
        ev(f"{b}bit bias16+col", *m_bias(W0, b))
        ev(f"{b}bit bias16+group256", *m_bias(W0, b, 256))
    m.W = W0
    return rows


if __name__ == "__main__":
    main()
