"""exp_robust.py : W の量子化の頑健性 (キュー6 の追加試験)。

python exp_robust.py model.npz corpus.txt [--gpu] [--limit 20000] [--parts group,noise,detector,big] [--seeds 2]

  group    : グループの行数を変える (4bit / 3bit)。スケールの数 (= 物理実装のゲインの数) と精度の関係
  noise    : W に雑音を乗せる (アナログの重みのばらつきの模擬)。乗算 (W*(1+σn)) と加算 (σ×グループ rms)。
             量子化なし / 4bit g256 の上に。雑音の乱数を seeds 通り変えて平均と幅を出す
  detector : 推論時だけ検出器を雑音ありにする (学習は理想検出。実機と学習環境がずれる場合)。W は fp32 / 4bit g256
  big      : 主な構成を、検証窓を --big-limit 文字 (既定 20万) にして評価し直す (標準誤差を小さくする)
各行で較正 (calibrate) はやり直す (実機でも検証データで較正できる想定)。
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # opticlm.py は1つ上
import opticlm as O


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("corpus")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--big-limit", type=int, default=200000)
    ap.add_argument("--parts", default="group,noise,detector,big")
    ap.add_argument("--seeds", type=int, default=2)
    ns = ap.parse_args()
    parts = set(ns.parts.split(","))
    m = O.load(ns.model, backend="torch" if ns.gpu else "numpy")
    _, va = O.split_docs(O.read_docs(ns.corpus))
    W0 = np.asarray(m.W, dtype=np.float64)
    D, V = W0.shape
    det0 = dict(m.cfg["detector"])
    print(f"{ns.model}: D={D} V={V} | 検証 {ns.limit} 文字 (big は {ns.big_limit})", flush=True)

    def ev(name, W, nbytes=None, limit=None, det=None):
        t0 = time.time()
        m.W = W
        m.cfg["detector"] = dict(det0, **(det or {}))
        lim = limit or ns.limit
        O.calibrate(m, va, lim)
        r = O.evaluate(m, va, lim)
        m.cfg["detector"] = dict(det0)
        sz = f"{nbytes / 1e6:7.2f}MB" if nbytes else " " * 9
        print(f"  {name:<34}{sz}{r['acc1']:>7.2f}{r['acc5']:>7.2f}{r['nll']:>8.3f}  n={r['n']}  ({time.time() - t0:.0f}s)", flush=True)
        return r

    print(f"  {'構成':<34}{'サイズ':>9}{'acc1':>7}{'acc5':>7}{'nll':>8}")
    ev("fp32 (基準)", W0, W0.size * 4)
    W4, b4 = O.quantize_matrix(W0, 4, 256)

    if "group" in parts:
        print("[group] グループの行数 (スケールの数 = ceil(D/行数) × V)")
        for bits in (4, 3):
            ev(f"{bits}bit 列ごと (グループ=D)", *O.quantize_matrix(W0, bits))
            for g in (4096, 1024, 512, 256, 128, 64, 32, 16):
                Wq, nb = O.quantize_matrix(W0, bits, g)
                ev(f"{bits}bit g{g} (スケール {-(-D // g) * V})", Wq, nb)

    if "noise" in parts:
        print(f"[noise] W に雑音 (乱数 {ns.seeds} 通りの平均 ± 幅)")
        g = 256
        Dp = -(-D // g) * g

        def grms(W):  # グループごとの rms を行に展開
            Wp = np.vstack([W, np.zeros((Dp - D, V))]).reshape(-1, g, V)
            return np.repeat(np.sqrt((Wp ** 2).mean(1)), g, axis=0)[:D]

        for base_name, Wb in (("fp32", W0), ("4bit g256", W4)):
            R = grms(Wb)
            for kind, sig in [("乗算", 0.01), ("乗算", 0.03), ("乗算", 0.1), ("乗算", 0.3),
                              ("加算", 0.03), ("加算", 0.1), ("加算", 0.3)]:
                rs = []
                for s in range(ns.seeds):
                    n = np.random.default_rng([7, s]).standard_normal(W0.shape)
                    Wn = Wb * (1 + sig * n) if kind == "乗算" else Wb + sig * R * n
                    rs.append(ev(f"{base_name} + {kind}雑音 σ={sig:g} (seed{s})", Wn))
                a1 = [r["acc1"] for r in rs]
                nl = [r["nll"] for r in rs]
                print(f"   -> 平均 acc1 {np.mean(a1):.2f} (幅 {np.ptp(a1):.2f}) nll {np.mean(nl):.3f} (幅 {np.ptp(nl):.3f})", flush=True)

    if "detector" in parts:
        print("[detector] 推論時だけ検出器に雑音 (学習は理想検出)")
        for det in (dict(photons=1000), dict(photons=100), dict(adc_bits=6), dict(adc_bits=4),
                    dict(photons=100, adc_bits=6)):
            for base_name, Wb in (("fp32", W0), ("4bit g256", W4)):
                ev(f"{base_name} + {det}", Wb, det=det)

    if "big" in parts:
        print(f"[big] 検証 {ns.big_limit} 文字で主な構成を評価し直す")
        ev("fp32", W0, W0.size * 4, limit=ns.big_limit)
        ev("8bit 列ごと", *O.quantize_matrix(W0, 8), limit=ns.big_limit)
        ev("5bit g256", *O.quantize_matrix(W0, 5, 256), limit=ns.big_limit)
        ev("4bit g256", W4, b4, limit=ns.big_limit)
        ev("4bit g64", *O.quantize_matrix(W0, 4, 64), limit=ns.big_limit)
        ev("3bit g64", *O.quantize_matrix(W0, 3, 64), limit=ns.big_limit)
        ev("4bit 列ごと (現行)", *O.quantize_matrix(W0, 4), limit=ns.big_limit)
    m.W = W0


if __name__ == "__main__":
    main()
