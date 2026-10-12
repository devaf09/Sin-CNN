"""exp_mixgate.py : 確信度で混合の重みを切り替える (REVIEW_repo.md の C1 = README の「プレビュー層: 荒い→詳細」)。CPU のみ。

python exp_mixgate.py scores.npz corpus.txt [--limit 60000] [--bins 8]

scores.npz は exp_calib.py dump の出力 (リザバーの検証窓スコア)。n-gram は同じ窓で計算する (窓の集合は selftest で一致を確認済み)。
窓を交互に「当てはめ用 / 評価用」に分け、評価用の nll で比べる。公平のため、n-gram 同士の混合にも同じ切り替えを与える。
  固定      : 混合の重み λ を1つ
  ゲートR   : リザバーの確信度 (最大スコア) の分位ビンごとに λ
  ゲートN   : n-gram の確信度 (最大確率) の分位ビンごとに λ
  ゲート2D  : 両方の確信度 (各4分位 = 16ビン) ごとに λ
"""
import argparse
import time

import numpy as np

import exp_calib as C
import opticlm as O

LAMS = np.linspace(0.0, 1.0, 41)


def fit_gate(pa, pb, g, fit):
    """ビン g ごとに λ を選ぶ (当てはめ用の窓で)。返り値: ビン -> λ の配列"""
    lam = np.zeros(g.max() + 1)
    for k in range(len(lam)):
        s = fit & (g == k)
        if not s.any():
            lam[k] = 0.5
            continue
        v = [-np.log(l * pa[s] + (1 - l) * pb[s] + 1e-300).mean() for l in LAMS]
        lam[k] = LAMS[int(np.argmin(v))]
    return lam


def nll_gate(pa, pb, g, lam, ev):
    l = lam[g[ev]]
    return float(-np.log(l * pa[ev] + (1 - l) * pb[ev] + 1e-300).mean())


def qbins(x, fit, nb):
    edges = np.quantile(x[fit], np.linspace(0, 1, nb + 1)[1:-1])
    return np.searchsorted(edges, x)


def compare(name_a, pa, ca, name_b, pb, cb, fit, ev, nb):
    one = np.zeros(len(pa), dtype=np.int64)
    ga, gb = qbins(ca, fit, nb), qbins(cb, fit, nb)
    g2 = qbins(ca, fit, 4) * 4 + qbins(cb, fit, 4)
    out = {}
    for tag, g in (("固定", one), ("ゲートA", ga), ("ゲートB", gb), ("ゲート2D", g2)):
        lam = fit_gate(pa, pb, g, fit)
        out[tag] = nll_gate(pa, pb, g, lam, ev)
    base = min(-np.log(pa[ev]).mean(), -np.log(pb[ev]).mean())
    print(f"  {name_a} + {name_b}: 単独の良い方 {base:.4f} | " +
          " | ".join(f"{k} {v:.4f}" for k, v in out.items()), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("corpus")
    ap.add_argument("--limit", type=int, default=60000)
    ap.add_argument("--bins", type=int, default=8)
    ns = ap.parse_args()
    O._low_priority()
    t0 = time.time()
    d = np.load(ns.scores)
    S, y, uni = d["S"].astype(np.float64), d["y"], d["uni"]
    n = len(y)
    fit = (np.arange(n) % 2) == 0
    ev = ~fit
    # リザバーの確率: 当てはめ用の窓で binpow 相当 (確信度ビン × power × 頻度^β) を選ぶ
    fi = np.nonzero(fit)[0]
    mx = np.where(np.isfinite(S), S, -np.inf).max(1)
    edges = np.quantile(mx[fi], np.linspace(0, 1, 9)[1:-1])
    b = np.searchsorted(edges, mx)
    p_res = np.empty(n)
    for k in range(8):
        sf = fi[b[fi] == k]
        prm = C.fit_pou(S[sf], y[sf], uni)
        sel = b == k
        P = C.apply_pou(S[sel], uni, prm)
        p_res[sel] = P[np.arange(sel.sum()), y[sel]]
    print(f"リザバー {ns.scores}: 窓 {n}, 評価用の nll {-np.log(p_res[ev]).mean():.4f} ({time.time() - t0:.0f}s)", flush=True)

    docs = O.read_docs(ns.corpus)
    tr, va = O.split_docs(docs)
    T = 4
    ngs = {}
    for o, mc in ((3, 1), (2, 1), (4, 1), (4, 2), (2, 3)):
        ng = NG = O.NgramLM(o, mc).fit(tr)
        r = ng.eval(va, T, ns.limit, with_py=True)
        if r["n"] != n:
            raise SystemExit(f"窓の数が合いません: n-gram {r['n']} vs リザバー {n}")
        # n-gram の確信度 = 最大確率 (窓ごと)
        text = O._as_text(va)[:ns.limit]
        ids = np.array([ng.idx.get(c, -1) for c in text], dtype=np.int64)
        m_ = len(ids) - T
        X = ids[np.arange(m_)[:, None] + np.arange(T)[None, :]]
        yy = ids[T:]
        keep = (yy >= 0) & (X >= 0).all(1)
        X = X[keep]
        conf = np.concatenate([ng._dense(X[s:s + 2000]).max(1) for s in range(0, len(X), 2000)])
        ngs[f"n-gram{o}/剪定{mc}"] = (r["py"], conf, ng.bytes(True))
        print(f"  n-gram 文脈{o} 剪定{mc}: {ng.bytes(True) / 1e6:.2f}MB, 評価用の nll {-np.log(r['py'][ev]).mean():.4f}", flush=True)

    print(f"\n[混合] 当てはめ用の窓で λ を選び、評価用の窓の nll ({ns.bins} 分位ビン)。ゲートA/B は左/右のモデルの確信度")
    base_name = "n-gram3/剪定1"
    pb, cb, _ = ngs[base_name]
    res = {}
    res["res"] = compare("リザバー", p_res, mx, base_name, pb, cb, fit, ev, ns.bins)
    for k, (pa, ca, nb_) in ngs.items():
        if k != base_name:
            res[k] = compare(k, pa, ca, base_name, pb, cb, fit, ev, ns.bins)
    # 対比較の標準誤差 (固定の混合どうしの差、およびゲート2Dどうし)
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
