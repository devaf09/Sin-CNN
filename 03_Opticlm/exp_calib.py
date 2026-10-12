"""exp_calib.py : 較正 (スコア -> 確率) の写像を比べる実験 (キュー4)。

1. dump  : モデルを学習し、検証窓のスコア S (n,V)・正解 y・ユニグラムを npz に保存 (GPU 可)
   python exp_calib.py dump cc100_v1m.txt scores_1m_d4096.npz --d 4096 [--gpu] [--max-chars N] [--limit 0]
2. run   : 保存したスコアで写像を比べる (CPU のみ)。検証窓を交互に「当てはめ用 / 評価用」に分け、評価用の nll で比べる
   python exp_calib.py run scores_1m_d4096.npz
"""
import argparse
import sys
import time

import numpy as np

import opticlm as O


def dump(ns):
    docs = O.read_docs(ns.corpus, ns.max_chars)
    tr, va = O.split_docs(docs)
    g = O.grid_for_D(ns.d, 4)
    m = O.OpticalLM(grid=g, scales=(1, 2, 3, 4), backend="torch" if ns.gpu else "numpy")
    t0 = time.time()
    m.fit(tr)
    best = O.sweep_lam(m, va, (1e-3, 1e-2), ns.limit, verbose=True)
    print(f"D={m.D} {time.time() - t0:.0f}s acc1 {best['acc1']:.1f} nll {best['nll']:.3f} cal {m.cal}")
    X, y = O.eval_windows(m, va, ns.limit)
    S = np.concatenate([m.scores(X[s:s + 2000]) for s in range(0, len(y), 2000)])
    np.savez_compressed(ns.out, S=S.astype(np.float32), y=y, uni=m.unigram(), D=m.D, cal_nll=best["nll"])
    print(f"保存 {ns.out}: {S.shape}")


# ---------------------------------------------------------------- 写像
def power_q(S, eps, g):
    ok = np.isfinite(S)
    Q = np.where(ok, np.maximum(S, eps), 0.0) ** g
    return Q / Q.sum(1, keepdims=True)


def nll_of(p):
    return float(-np.log(p + 1e-300).mean())


def fit_power(S, y, uni):
    """既存の power (eps, gamma, a)。返り値 (params, 関数)"""
    best = (np.inf, None)
    uy = uni[y]
    for eps in O._EPS:
        for g in O._GAMMAS + (2.5, 5.0, 6.0):
            qy = power_q(S, eps, g)[np.arange(len(y)), y]
            for a in O._MIX:
                v = nll_of(a * qy + (1 - a) * uy)
                if v < best[0]:
                    best = (v, (eps, g, a))
    return best[1]


def apply_power(S, uni, prm):
    eps, g, a = prm
    return a * power_q(S, eps, g) + (1 - a) * uni[None, :]


def conf_bins(S, edges):
    """確信度 = 最大スコア。そのビン番号。"""
    mx = np.where(np.isfinite(S), S, -np.inf).max(1)
    return np.searchsorted(edges, mx)


def fit_binned(S, y, uni, nb=8):
    """最大スコアの分位でビンに分け、ビンごとに (eps,gamma,a) を選ぶ (文脈の確からしさで写像を切り替える)。"""
    mx = np.where(np.isfinite(S), S, -np.inf).max(1)
    edges = np.quantile(mx, np.linspace(0, 1, nb + 1)[1:-1])
    b = np.searchsorted(edges, mx)
    glob = fit_power(S, y, uni)
    prms = []
    for k in range(nb):
        sel = b == k
        prms.append(fit_power(S[sel], y[sel], uni) if sel.sum() > 200 else glob)
    return edges, prms


def apply_binned(S, uni, model):
    edges, prms = model
    b = np.searchsorted(edges, np.where(np.isfinite(S), S, -np.inf).max(1))
    P = np.empty_like(S, dtype=np.float64)
    for k, prm in enumerate(prms):
        sel = b == k
        if sel.any():
            P[sel] = apply_power(S[sel], uni, prm)
    return P


def fit_rank(S, y, uni, base, K=20, iters=30):
    """power の上に、順位ごとの倍率 c_r (r<K) を掛けて正規化し直す。c は当てはめ用で最尤 (乗法更新)。"""
    P0 = apply_power(S, uni, base)
    order = np.argsort(-np.where(np.isfinite(S), S, -np.inf), axis=1)[:, :K]
    rank_y = np.full(len(y), K)
    hit = order == y[:, None]
    rank_y[hit.any(1)] = hit.argmax(1)[hit.any(1)]
    Ptop = np.take_along_axis(P0, order, 1)  # (n,K)
    rest = 1.0 - Ptop.sum(1)
    c = np.ones(K + 1)  # c[K] = 圏外の倍率
    for _ in range(iters):
        Z = Ptop @ c[:K] + rest * c[K]
        # 期待値と実際の頻度の比で更新 (EM 的な固定点)
        cnt = np.bincount(rank_y, minlength=K + 1).astype(np.float64)
        exp_ = np.concatenate([(Ptop / Z[:, None]).sum(0), [(rest / Z).sum()]])
        c = cnt / np.maximum(exp_, 1e-12)
        c /= c[K]
    return base, order.shape[1], c


def apply_rank(S, uni, model):
    base, K, c = model
    P = apply_power(S, uni, base)
    order = np.argsort(-np.where(np.isfinite(S), S, -np.inf), axis=1)[:, :K]
    mult = np.full_like(P, c[K])
    np.put_along_axis(mult, order, c[:K][None, :].repeat(len(P), 0), 1)
    P = P * mult
    return P / P.sum(1, keepdims=True)


def run(ns):
    d = np.load(ns.scores)
    S, y, uni = d["S"].astype(np.float64), d["y"], d["uni"]
    n = len(y)
    fit_i, ev_i = np.arange(0, n, 2), np.arange(1, n, 2)
    Sf, yf, Se, ye = S[fit_i], y[fit_i], S[ev_i], y[ev_i]
    print(f"{ns.scores}: 窓 {n} (当てはめ {len(yf)} / 評価 {len(ye)}), D={int(d['D'])}, 保存時の nll(全窓で選択) {float(d['cal_nll']):.3f}")
    res = []
    t0 = time.time()
    pw = fit_power(Sf, yf, uni)
    Pe = apply_power(Se, uni, pw)
    res.append(("power (現行)", nll_of(Pe[np.arange(len(ye)), ye]), str(pw)))
    for nb in (4, 8, 16):
        mdl = fit_binned(Sf, yf, uni, nb)
        Pe = apply_binned(Se, uni, mdl)
        res.append((f"確信度ビン {nb}", nll_of(Pe[np.arange(len(ye)), ye]), " ".join(f"g{p[1]:g}/a{p[2]:g}" for p in mdl[1])))
    for K in (5, 20, 50):
        mdl = fit_rank(Sf, yf, uni, pw, K)
        Pe = apply_rank(Se, uni, mdl)
        assert np.allclose(Pe.sum(1), 1.0)
        res.append((f"順位補正 K={K}", nll_of(Pe[np.arange(len(ye)), ye]), " ".join(f"{v:.2f}" for v in mdl[2][:6]) + " ..."))
    mdl_b = fit_binned(Sf, yf, uni, 8)
    Pb_f = apply_binned(Sf, uni, mdl_b)
    # ビン + 順位補正の組合せ
    mdl = fit_rank_from_P(Sf, yf, Pb_f, 20)
    Pe = rank_apply_P(Se, apply_binned(Se, uni, mdl_b), mdl)
    res.append(("確信度ビン8 + 順位補正20", nll_of(Pe[np.arange(len(ye)), ye]), ""))
    base = res[0][1]
    print(f"{'写像':<26}{'評価nll':>9}{'差':>8}  パラメータ")
    for name, v, p in res:
        print(f"{name:<26}{v:>9.4f}{v - base:>+8.4f}  {p}")
    print(f"({time.time() - t0:.0f}s)")


def fit_rank_from_P(S, y, P0, K=20, iters=30):
    order = np.argsort(-np.where(np.isfinite(S), S, -np.inf), axis=1)[:, :K]
    hit = order == y[:, None]
    rank_y = np.full(len(y), K)
    rank_y[hit.any(1)] = hit.argmax(1)[hit.any(1)]
    Ptop = np.take_along_axis(P0, order, 1)
    rest = 1.0 - Ptop.sum(1)
    cnt = np.bincount(rank_y, minlength=K + 1).astype(np.float64)
    c = np.ones(K + 1)
    for _ in range(iters):
        Z = Ptop @ c[:K] + rest * c[K]
        exp_ = np.concatenate([(Ptop / Z[:, None]).sum(0), [(rest / Z).sum()]])
        c = cnt / np.maximum(exp_, 1e-12)
        c /= c[K]
    return K, c


def rank_apply_P(S, P, mdl):
    K, c = mdl
    order = np.argsort(-np.where(np.isfinite(S), S, -np.inf), axis=1)[:, :K]
    mult = np.full_like(P, c[K])
    np.put_along_axis(mult, order, np.broadcast_to(c[:K], order.shape).copy(), 1)
    P = P * mult
    return P / P.sum(1, keepdims=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    a = sub.add_parser("dump")
    a.add_argument("corpus")
    a.add_argument("out")
    a.add_argument("--d", type=int, default=4096)
    a.add_argument("--gpu", action="store_true")
    a.add_argument("--max-chars", type=int, default=0)
    a.add_argument("--limit", type=int, default=20000)
    b = sub.add_parser("run")
    b.add_argument("scores")
    ns = ap.parse_args()
    {"dump": dump, "run": run}.get(ns.cmd, lambda _: ap.print_help())(ns)


# ---------------------------------------------------------------- 追加: ユニグラムとの積 (product of experts) と細かい a
def fit_pou(S, y, uni, gammas=(1.0, 1.25, 1.5, 1.75, 2.0), betas=(-0.6, -0.4, -0.2, 0.0, 0.2), eps_l=(1e-4, 1e-3),
            mix=(0.8, 0.85, 0.9, 0.93, 0.96, 0.98, 0.99, 1.0)):
    """p = a * norm(clip(S,eps)^g * uni^beta) + (1-a) * uni"""
    best = (np.inf, None)
    uy = uni[y]
    ok = np.isfinite(S)
    for eps in eps_l:
        Q0 = np.where(ok, np.maximum(S, eps), 0.0)
        for g in gammas:
            Qg = Q0 ** g
            for bt in betas:
                Q = Qg * uni[None, :] ** bt
                qy = Q[np.arange(len(y)), y] / Q.sum(1)
                for a in mix:
                    v = nll_of(a * qy + (1 - a) * uy)
                    if v < best[0]:
                        best = (v, (eps, g, bt, a))
    return best[1]


def apply_pou(S, uni, prm):
    eps, g, bt, a = prm
    Q = np.where(np.isfinite(S), np.maximum(S, eps), 0.0) ** g * uni[None, :] ** bt
    return a * Q / Q.sum(1, keepdims=True) + (1 - a) * uni[None, :]


def run2(path):
    d = np.load(path)
    S, y, uni = d["S"].astype(np.float64), d["y"], d["uni"]
    n = len(y)
    fi, ei = np.arange(0, n, 2), np.arange(1, n, 2)
    t0 = time.time()
    pw = fit_power(S[fi], y[fi], uni)
    base = nll_of(apply_power(S[ei], uni, pw)[np.arange(len(ei)), y[ei]])
    prm = fit_pou(S[fi], y[fi], uni)
    Pe = apply_pou(S[ei], uni, prm)
    v = nll_of(Pe[np.arange(len(ei)), y[ei]])
    print(f"power {base:.4f} | power×uni^β {v:.4f} ({v - base:+.4f}) {prm}")
    # 確信度ビンごとに pou
    mx = np.where(np.isfinite(S), S, -np.inf).max(1)
    edges = np.quantile(mx[fi], np.linspace(0, 1, 9)[1:-1])
    b = np.searchsorted(edges, mx)
    Pe = np.empty((len(ei), S.shape[1]))
    for k in range(8):
        sf, se = fi[b[fi] == k], b[ei] == k
        p = fit_pou(S[sf], y[sf], uni)
        Pe[se] = apply_pou(S[ei][se], uni, p)
    assert np.allclose(Pe.sum(1), 1.0)
    v2 = nll_of(Pe[np.arange(len(ei)), y[ei]])
    print(f"確信度ビン8 × (power×uni^β) {v2:.4f} ({v2 - base:+.4f})  ({time.time() - t0:.0f}s)")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "run2":
    run2(sys.argv[2])
