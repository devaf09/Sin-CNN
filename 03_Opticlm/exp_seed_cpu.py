"""exp_seed_cpu.py : 光学媒質の seed を変えて、量子化の結果が再現するかを CPU だけで確かめる (GPU の実験と並行で走らせる用)。
python exp_seed_cpu.py corpus.txt --seeds 1,2 --d 4096 [--workers 3 --threads 2]
学習は fit_parallel (子プロセスの優先度は低い)。このプロセス自身も優先度を下げる。"""
import argparse
import time

import numpy as np

import opticlm as O

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--d", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=20000)
    ns = ap.parse_args()
    O._low_priority()
    docs = O.read_docs(ns.corpus)
    tr, va = O.split_docs(docs)
    for seed in [int(s) for s in ns.seeds.split(",")]:
        t0 = time.time()
        m = O.OpticalLM(grid=O.grid_for_D(ns.d, 4), seed=seed, backend="numpy")
        O.fit_parallel(m, tr, workers=ns.workers, threads=ns.threads)
        t1 = time.time() - t0
        best = O.sweep_lam(m, va, (1e-3, 1e-2), ns.limit, verbose=False)
        W0 = np.asarray(m.W)
        print(f"seed={seed} D={m.D} 学習 {t1:.0f}s (並列 {ns.workers}x{ns.threads}) lam={best['lam']:g}", flush=True)
        for name, (Wq, nb) in [("fp32", (W0, W0.size * 4)), ("8bit 列ごと", O.quantize_matrix(W0, 8)),
                               ("4bit 列ごと", O.quantize_matrix(W0, 4)), ("5bit g256", O.quantize_matrix(W0, 5, 256)),
                               ("4bit g256", O.quantize_matrix(W0, 4, 256)), ("3bit g64", O.quantize_matrix(W0, 3, 64))]:
            m.W = Wq
            O.calibrate(m, va, ns.limit)
            r = O.evaluate(m, va, ns.limit)
            print(f"  seed={seed} {name:<12}{nb / 1e6:7.2f}MB acc1 {r['acc1']:6.2f} acc5 {r['acc5']:6.2f} nll {r['nll']:.3f}", flush=True)
        print(f"  seed={seed} 合計 {time.time() - t0:.0f}s", flush=True)
