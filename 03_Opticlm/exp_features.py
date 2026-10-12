"""exp_features.py : 「光学であることの価値」を測る対照実験 (REVIEW_repo.md の C3) と、2段カスケード (高次の相互作用) の試験。

python exp_features.py corpus.txt [--d 4096] [--modes optics,nodiff,cascade,rff,prod] [--max-chars 0] [--gpu]

同じ総特徴数 D・同じ露光 (scales=1,2,3,4 の入れ子の窓)・同じ閉形式の読み出し・同じ較正で、特徴の作り方だけを変える。
  optics  : 現行 (位相板 3層 + 伝播、強度検出)。電場について線形な変換のあとの |.|² なので、特徴は「文字の2つ組」の相互作用だけ
  nodiff  : 位相板も伝播もなし。|Σ 文字ホログラム × 位置の位相|² をそのまま読む (回折の構造が効いているかの対照)
  cascade : 2段。1段目の強度 (正規化) を α 倍して2段目の入力の位相にし (exp(iαI))、別の位相板 3層 + 伝播のあと検出。
            強度を位相に書き戻すので、3文字以上の相互作用が生まれる (物理: カメラ→SLM、または光電変換素子で実現できる)
  rff     : ランダムフーリエ特徴 cos(σ Σ_k g_k[文字] + b) (g はガウス乱数)。全次数の相互作用を含む「普通の」ランダム特徴
  prod    : cos(Σ_k θ_k[文字] + b) (θ は一様乱数の位相)。窓の文字列が完全に一致するときだけ似る = ハッシュ化した n-gram
"""
import argparse
import time

import numpy as np

import opticlm as O

torch = O.torch


class Variant(O.TorchOpticalLM):
    def __init__(self, *a, mode="optics", alpha=1.0, sigma=1.0, **kw):
        self.mode, self.alpha, self.sigma = mode, alpha, sigma
        super().__init__(*a, **kw)
        G, L, seed = self.G, self.cfg["layers"], self.seed
        rng = lambda *key: np.random.default_rng([seed, *key])
        self.mask2 = [self._tt(np.exp(1j * np.stack([rng(6, e, l).uniform(0, 2 * np.pi, (G, G)) for l in range(L)])))
                      for e in range(self.E)]
        self.bias_ph = [torch.as_tensor(rng(7, e).uniform(0, 2 * np.pi, self.N), dtype=torch.float32, device=self.dev)
                        for e in range(self.E)]
        self.gauss = None

    def _gauss_table(self):
        """文字ごとのガウス乱数 (露光 e, 位置 k ごとに別)。表は文字の追加に合わせて作り直す (文字ごとに決まった乱数)。"""
        V1 = self.table.shape[0]
        if self.gauss is None or self.gauss[0].shape[1] < V1:
            keys = [PADK] + [ord(c) for c in self.chars]
            self.gauss = [torch.as_tensor(np.stack([np.stack([np.random.default_rng([self.seed, 8, e, k, kk]).standard_normal(self.N)
                                                              for kk in keys]) for k in range(ex["T"])]),
                                          dtype=torch.float32, device=self.dev) for e, ex in enumerate(self.exps)]
        return self.gauss

    def _optics(self, xb):
        if self.mode == "optics":
            return super()._optics(xb)
        n, G, N = xb.shape[0], self.G, self.N
        out = self._empty((n, self.D))
        out[:, -1] = 1.0
        col = 0
        if self.mode == "rff":
            gt = self._gauss_table()
        for e, ex in enumerate(self.exps):
            Te = ex["T"]
            if self.mode in ("rff", "prod"):
                arg = self.bias_ph[e][None, :].expand(n, N).clone()
                for k in range(Te):
                    c = xb[:, -1 - k]
                    if self.mode == "rff":
                        arg = arg + self.sigma * gt[e][k][c]
                    else:
                        arg = arg + torch.angle(self.p_table[c] * self.p_pos[e][Te - 1 - k])
                out[:, col:col + N] = torch.cos(arg) * np.sqrt(2.0)
                col += N
                continue
            U = None
            for k in range(Te):
                term = self.p_table[xb[:, -1 - k]] * self.p_pos[e][Te - 1 - k]
                U = term if U is None else U + term
            U = U.reshape(n, G, G)
            m = ex["norm"]
            if self.mode == "nodiff":
                I = U.real ** 2 + U.imag ** 2
            else:  # cascade
                for l in range(self.cfg["layers"]):
                    U = self.ifft2(self.fft2(U * self.p_mask[e][l]) * self.p_H)
                I1 = ((U.real ** 2 + U.imag ** 2) - m) / m
                U = torch.exp(1j * self.alpha * I1)
                for l in range(self.cfg["layers"]):
                    U = self.ifft2(self.fft2(U * self.mask2[e][l]) * self.p_H)
                I = U.real ** 2 + U.imag ** 2
                m = 1.0
            out[:, col:col + N] = (I.reshape(n, -1) - m) / (ex["norm"] if self.mode == "nodiff" else 1.0)
            col += N
        return out


PADK = 0x7FFFFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--d", type=int, default=4096)
    ap.add_argument("--modes", default="optics,nodiff,cascade,rff,prod")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--max-chars", type=int, default=0)
    ap.add_argument("--limit", type=int, default=20000)
    ns = ap.parse_args()
    docs = O.read_docs(ns.corpus, ns.max_chars)
    tr, va = O.split_docs(docs)
    g = O.grid_for_D(ns.d, 4)
    print(f"{ns.corpus}: 文字数 {sum(map(len, docs))} | D={4 * g * g + 1} | 検証 {ns.limit} 文字", flush=True)
    print(f"{'特徴':<22}{'train':>7}{'acc1':>7}{'acc5':>7}{'nll':>8}{'秒':>6}")
    for mode in ns.modes.split(","):
        t0 = time.time()
        kw = {}
        name = mode
        if ":" in mode:  # 例 cascade:2.0 / rff:0.5
            mode, v = mode.split(":")
            kw = {"alpha" if mode == "cascade" else "sigma": float(v)}
        m = Variant(grid=g, scales=(1, 2, 3, 4), mode=mode, backend="torch", **kw)
        m.fit(tr)
        best = O.sweep_lam(m, va, (1e-3, 1e-2, 1e-1), ns.limit, verbose=False)
        t1 = O.evaluate(m, "".join(tr[:max(1, len(tr) // 4)])[:100000], 100000)["acc1"]
        print(f"{name:<22}{t1:>7.1f}{best['acc1']:>7.1f}{best['acc5']:>7.1f}{best['nll']:>8.3f}{time.time() - t0:>6.0f}", flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
