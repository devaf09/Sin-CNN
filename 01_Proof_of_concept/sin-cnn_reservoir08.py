"""
sin-cnn_reservoir08.py : 多重スケールの光学リザバー + 閉形式の読み出し (numpy。GPU(torch)でも動く)

07 からの変更点
  * 露光を複数並べる: 直近 1, 2, 4, 8 文字だけを見る別々の光学媒質(=露光)を用意し、
    カメラ強度を全部つなげて1本の読み出しで解く。
      短い文脈 -> 少ないデータでも汎化しやすい / 長い文脈 -> 暗記に強い
    をリッジ回帰が自動で使い分ける(n-gram のバックオフに近い働き)。
  * 大きなテキスト向けに、評価を先頭 limit 文字に制限、学習は逐次チャンク。

学習則は 07 と同じ(足し算だけ):
    A = sum phi^T phi ,  B = sum phi^T onehot(y) ,  (A + reg) W = B
  -> 追加学習 / 融合 / 取り消し / 語彙の追加 が厳密に成り立つ。

使い方:
    python sin-cnn_reservoir08.py corpus.txt [grid=32] [max_chars=0] [scales=1,2,4,8] [split=scatter|doc|tail] [gpu]
      grid      : 露光1枚あたりの画素の一辺 (総特徴数 = 露光数 x grid^2)
      max_chars : 先頭だけ使う文字数 (0=全部)
      scales    : 各露光が見る直近の文字数 (カンマ区切り。例 1,2,3,4,6,8)
      split     : scatter=検証を全体に散らす(既定) / doc=1行1文書で文書単位に分ける(CC-100向け) / tail=末尾15%(従来)
      gpu       : 最後に gpu と書くと、特徴の計算・統計量の積算・解くところを GPU(torch) で行う。
                  特徴数 D が大きいときの時間(約 D^2)を大きく減らせる。統計量は float64 のまま。
                  目安: 統計量 A は D^2 x 8 バイト。VRAM 8GB なら D = 16000 程度まで (D=16385 で約2.1GB。解くときにさらに同量)。
                  gpu 指定時は、[4]融合と[5]取り消し(メモリを多く使う検証)を省略する。D が 12000 を超えると
                  [3]の比較と統計量の保存も省略する。
    python sin-cnn_reservoir08.py selftest
      GPU版と numpy 版が同じ結果になるかを小さな設定で自己照合する (gpu 指定で走らせる前に1回実行してください)。
"""
import copy
import json
import os
import sys
import time

import numpy as np

try:
    import torch
except Exception:  # torch が無くても numpy 版は動く
    torch = None

PAD_KEY = 0x7FFFFFFF
BIG_D = 12000  # gpu 指定でこれを超える特徴数のときは、[3]の比較と統計量の保存を省略する


class OpticalReservoirLM:
    def __init__(self, grid=32, scales=(1, 2, 4, 8), layers=3, dist=5e-3,
                 wavelength=633e-9, pitch=20e-6, seed=0):
        scales = tuple(int(s) for s in scales)
        self.cfg = dict(grid=grid, scales=scales, layers=layers, dist=dist,
                        wavelength=wavelength, pitch=pitch, seed=seed)
        G = grid
        self.G, self.N, self.scales, self.seed = G, G * G, scales, seed
        self.E = len(scales)
        self.T = max(scales)             # 文脈の最大長
        self.D = self.E * self.N + 1     # 全特徴数 (+バイアス)
        rng = lambda *key: np.random.default_rng([seed, *key])

        # ---- 固定の物理 (学習しない): 露光ごとに独立な媒質 ----
        self.pos = [np.exp(1j * np.stack([rng(1, e, p).uniform(0, 2 * np.pi, self.N) for p in range(s)]))
                    for e, s in enumerate(scales)]
        self.mask = [np.exp(1j * np.stack([rng(2, e, l).uniform(0, 2 * np.pi, (G, G)) for l in range(layers)]))
                     for e in range(self.E)]
        f = np.fft.fftfreq(G, d=pitch)  # fft の並びそのまま
        FX, FY = np.meshgrid(f, f, indexing="ij")
        k = 2 * np.pi / wavelength
        kz = np.sqrt(np.maximum(k ** 2 - (2 * np.pi * FX) ** 2 - (2 * np.pi * FY) ** 2, 0.0))
        self.H = np.exp(1j * kz * dist)

        # ---- 語彙 (文字 -> 位相ホログラム。行0は PAD) ----
        self.chars, self.idx = [], {}
        self.table = np.exp(1j * rng(3, PAD_KEY).uniform(0, 2 * np.pi, self.N))[None, :]

        # ---- 学習する統計量 ----
        self._alloc_stats()
        self.count = np.zeros(0, dtype=np.int64)
        self.n = 0.0
        self.W = None
        self.tau = 0.05

    def _alloc_stats(self):
        self.A = np.zeros((self.D, self.D))
        self.Bt = np.zeros((0, self.D))

    def _grow_classes(self, k):
        self.Bt = np.vstack([self.Bt, np.zeros((k, self.D))])

    # ------------------------------------------------------------ 語彙
    def _add_chars(self, new):
        new = [c for c in dict.fromkeys(new) if c not in self.idx]
        if not new:
            return
        pats = [np.exp(1j * np.random.default_rng([self.seed, 3, ord(c)]).uniform(0, 2 * np.pi, self.N)) for c in new]
        for c in new:
            self.idx[c] = len(self.chars)
            self.chars.append(c)
        self.table = np.vstack([self.table, np.stack(pats)])
        self._grow_classes(len(new))
        self.count = np.concatenate([self.count, np.zeros(len(new), dtype=np.int64)])

    def encode(self, text):
        self._add_chars(list(text))
        return np.array([self.idx[c] for c in text], dtype=np.int64)

    def rows_for(self, text):
        return np.array([self.idx[c] + 1 if c in self.idx else 0 for c in text], dtype=np.int64)

    def _windows(self, rows):
        n = len(rows) - self.T
        if n <= 0:
            return np.zeros((0, self.T), dtype=np.int64), np.zeros(0, dtype=np.int64)
        i = np.arange(n)[:, None] + np.arange(self.T)[None, :]
        return rows[i], rows[self.T:self.T + n]

    # ------------------------------------------------------------ 光学 (固定)
    def features(self, rows, batch=128):
        """rows: (n, T) の table 行番号 -> 全露光のカメラ強度 (n, D)。最後の列は定数1。"""
        n = len(rows)
        out = np.empty((n, self.D))
        out[:, -1] = 1.0
        for s0 in range(0, n, batch):
            xb = rows[s0:s0 + batch]
            for e, s in enumerate(self.scales):
                U = (self.table[xb[:, -s:]] * self.pos[e][None]).sum(1).reshape(-1, self.G, self.G)
                for l in range(self.cfg["layers"]):
                    U = U * self.mask[e][l]
                    U = np.fft.ifft2(np.fft.fft2(U) * self.H)
                I = (U.real ** 2 + U.imag ** 2).reshape(len(xb), -1)
                out[s0:s0 + len(xb), e * self.N:(e + 1) * self.N] = I / s - 1.0
        return out

    # ------------------------------------------------------------ 学習 (足し算)
    def fit_text(self, text, weight=1.0, batch=128):
        """追加学習。weight=-1 でその文章を取り消す。"""
        ids = self.encode(text)
        X, y = self._windows(ids + 1)
        y = y - 1
        for s0 in range(0, len(X), batch):
            Phi = self.features(X[s0:s0 + batch])
            self.A += weight * (Phi.T @ Phi)
            np.add.at(self.Bt, y[s0:s0 + batch], weight * Phi)
            np.add.at(self.count, y[s0:s0 + batch], 1 if weight > 0 else -1)
        self.n += weight * len(X)
        return len(X)

    def solve(self, lam_rel=1e-2):
        scale = np.trace(self.A) / self.D
        reg = np.full(self.D, lam_rel * scale)
        reg[-1] = 1e-8 * scale
        M = self.A.copy()
        M[np.diag_indices(self.D)] += reg
        self.W = np.linalg.solve(M, self.Bt.T)
        return self.W

    # ------------------------------------------------------------ 予測
    def scores_from_features(self, Phi):
        S = Phi @ self.W
        if S.shape[1] < len(self.chars):
            S = np.hstack([S, np.zeros((len(S), len(self.chars) - S.shape[1]))])
        S[:, self.count[:S.shape[1]] <= 0] = -np.inf
        return S

    def scores(self, rows):
        return self.scores_from_features(self.features(rows))

    def generate(self, prompt, length=100, tau=None, top_k=5, seed=0):
        tau = self.tau if tau is None else tau
        rng = np.random.default_rng(seed)
        ctx, out = list(prompt), list(prompt)
        for _ in range(length):
            rows = self.rows_for("".join(ctx[-self.T:]))
            rows = np.concatenate([np.zeros(self.T - len(rows), dtype=np.int64), rows])[None, :]
            s = self.scores(rows)[0]
            if tau <= 0:
                j = int(np.argmax(s))
            else:
                top = np.argsort(s)[-top_k:]
                p = np.exp((s[top] - s[top].max()) / tau)
                j = int(rng.choice(top, p=p / p.sum()))
            out.append(self.chars[j])
            ctx.append(self.chars[j])
        return "".join(out)

    # ------------------------------------------------------------ 融合 / 保存
    def merge(self, other):
        assert self.cfg == other.cfg, "物理設定が違うモデルは融合できません"
        self._add_chars(other.chars)
        for j, c in enumerate(other.chars):
            self.Bt[self.idx[c]] += other.Bt[j]
            self.count[self.idx[c]] += other.count[j]
        self.A += other.A
        self.n += other.n
        return self

    def save(self, path, with_stats=True):
        """with_stats=False: 推論専用(W を float32)。True: 追加学習用(A, Bt を float32 で保存)。"""
        data = dict(cfg=json.dumps(self.cfg), chars=np.array(self.chars), count=self.count,
                    W=self.W.astype(np.float32), tau=self.tau)
        if with_stats:
            data.update(A=self.A.astype(np.float32), Bt=self.Bt.astype(np.float32), n=self.n)
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        m = cls(**json.loads(str(d["cfg"])))
        m._add_chars([str(c) for c in d["chars"]])
        m.count = d["count"].astype(np.int64)
        m.W = d["W"].astype(np.float64)
        m.tau = float(d["tau"])
        if "A" in d.files:
            m.A, m.Bt, m.n = d["A"].astype(np.float64), d["Bt"].astype(np.float64), float(d["n"])
        return m


class TorchReservoirLM(OpticalReservoirLM):
    """GPU(torch)版。光学の計算(位相板+伝搬+強度検出)、統計量 A/B の積算、解く処理を GPU で行う。
    学習則・API・保存形式は numpy 版と同じ。違いは次の2点だけ:
      * 特徴(カメラ強度)は float32 / complex64 で計算する (numpy 版は float64)。
      * 統計量 A, B と解は float64 のまま GPU 上に持つ (積算の丸めで融合や取り消しが 1e-15 より少し大きくずれうる)。
    ※ この環境には torch が無く、実機では未検証。まず `python sin-cnn_reservoir08.py selftest` で numpy 版と照合すること。"""

    def __init__(self, *a, device=None, **kw):
        if torch is None:
            raise RuntimeError("torch が見つかりません。 pip install torch で入れるか、gpu 指定なしで実行してください。")
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.Wt = None
        super().__init__(*a, **kw)
        tt = lambda x: torch.as_tensor(np.asarray(x).astype(np.complex64), device=self.dev)
        self.t_pos = [tt(p) for p in self.pos]
        self.t_mask = [tt(m) for m in self.mask]
        self.t_H = tt(self.H)
        self.t_table = tt(self.table)

    # ---- numpy 版の W を、GPU 上の Wt (float64) の窓口にする ----
    @property
    def W(self):
        return None if self.Wt is None else self.Wt.cpu().numpy()

    @W.setter
    def W(self, v):
        self.Wt = None if v is None else torch.as_tensor(np.asarray(v), dtype=torch.float64, device=self.dev)

    def _alloc_stats(self):
        self.A = torch.zeros((self.D, self.D), dtype=torch.float64, device=self.dev)
        self.Bt = torch.zeros((0, self.D), dtype=torch.float64, device=self.dev)

    def _grow_classes(self, k):
        self.Bt = torch.cat([self.Bt, torch.zeros((k, self.D), dtype=torch.float64, device=self.dev)], 0)

    def _add_chars(self, new):
        before = len(self.chars)
        super()._add_chars(new)
        if len(self.chars) != before:
            self.t_table = torch.as_tensor(self.table.astype(np.complex64), device=self.dev)

    # ---- 光学 (固定) ----
    def features_t(self, rows):
        """rows: (n, T) の table 行番号 (numpy) -> カメラ強度 (n, D) の float32 テンソル(GPU上)。"""
        xb = torch.as_tensor(np.asarray(rows), dtype=torch.long, device=self.dev)
        n = xb.shape[0]
        out = torch.empty((n, self.D), dtype=torch.float32, device=self.dev)
        out[:, -1] = 1.0
        for e, s in enumerate(self.scales):
            U = (self.t_table[xb[:, -s:]] * self.t_pos[e].unsqueeze(0)).sum(1).reshape(n, self.G, self.G)
            for l in range(self.cfg["layers"]):
                U = U * self.t_mask[e][l]
                U = torch.fft.ifft2(torch.fft.fft2(U) * self.t_H)
            I = (U.real ** 2 + U.imag ** 2).reshape(n, -1)
            out[:, e * self.N:(e + 1) * self.N] = I / s - 1.0
        return out

    # ---- 学習 (足し算) ----
    def fit_text(self, text, weight=1.0, batch=1024):
        ids = self.encode(text)
        X, y = self._windows(ids + 1)
        y = y - 1
        for s0 in range(0, len(X), batch):
            Phi = self.features_t(X[s0:s0 + batch]).double()
            yb = torch.as_tensor(y[s0:s0 + batch], dtype=torch.long, device=self.dev)
            self.A.addmm_(Phi.T, Phi, alpha=weight)
            self.Bt.index_add_(0, yb, Phi, alpha=weight)
            np.add.at(self.count, y[s0:s0 + batch], 1 if weight > 0 else -1)
        self.n += weight * len(X)
        return len(X)

    def solve(self, lam_rel=1e-2):
        """(A + reg) W = B をコレスキー分解で解く。メモリ節約のため A の対角を一時的に書き換えて戻す。"""
        scale = float(torch.trace(self.A).item()) / self.D
        reg = torch.full((self.D,), lam_rel * scale, dtype=torch.float64, device=self.dev)
        reg[-1] = 1e-8 * scale
        diag = self.A.diagonal()
        diag.add_(reg)
        try:
            L = torch.linalg.cholesky(self.A)
        finally:
            diag.sub_(reg)
        self.Wt = torch.cholesky_solve(self.Bt.T.contiguous(), L)
        del L
        return self.Wt

    # ---- 予測 ----
    def scores(self, rows):
        S = (self.features_t(rows).double() @ self.Wt).cpu().numpy()
        V = len(self.chars)
        if S.shape[1] < V:
            S = np.hstack([S, np.zeros((len(S), V - S.shape[1]))])
        S[:, self.count[:S.shape[1]] <= 0] = -np.inf
        return S

    # ---- 保存 / 読み込み (形式は numpy 版と同じ) ----
    def save(self, path, with_stats=True):
        data = dict(cfg=json.dumps(self.cfg), chars=np.array(self.chars), count=self.count,
                    W=self.Wt.cpu().numpy().astype(np.float32), tau=self.tau)
        if with_stats:
            data.update(A=self.A.cpu().numpy().astype(np.float32), Bt=self.Bt.cpu().numpy().astype(np.float32), n=self.n)
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path, **kw):
        d = np.load(path, allow_pickle=False)
        m = cls(**json.loads(str(d["cfg"])), **kw)
        m._add_chars([str(c) for c in d["chars"]])
        m.count = d["count"].astype(np.int64)
        m.W = d["W"].astype(np.float64)
        m.tau = float(d["tau"])
        if "A" in d.files:
            m.A = torch.as_tensor(d["A"].astype(np.float64), device=m.dev)
            m.Bt = torch.as_tensor(d["Bt"].astype(np.float64), device=m.dev)
            m.n = float(d["n"])
        return m


def selftest():
    """GPU版と numpy 版を、小さな設定で照合する。"""
    if torch is None:
        print("torch が見つかりません (pip install torch)。")
        return
    rng = np.random.default_rng(0)
    chars = [chr(0x3041 + i) for i in range(40)]
    text = "".join(chars[i] for i in rng.integers(0, 40, 4000))
    a = OpticalReservoirLM(grid=16, scales=(1, 2, 3))
    b = TorchReservoirLM(grid=16, scales=(1, 2, 3))
    print(f"torch {torch.__version__} | デバイス: {b.dev}")
    a.fit_text(text)
    b.fit_text(text)
    rel = lambda x, y: float(np.abs(x - y).max() / max(np.abs(x).max(), 1e-300))
    X, _ = a._windows(a.rows_for(text[:300]))
    fa, fb = a.features(X), b.features_t(X).cpu().numpy()
    r_f, r_A, r_B = rel(fa, fb), rel(a.A, b.A.cpu().numpy()), rel(a.Bt, b.Bt.cpu().numpy())
    a.solve(1e-2)
    b.solve(1e-2)
    Sa, Sb = a.scores(X), b.scores(X)
    ok = np.isfinite(Sa) & np.isfinite(Sb)
    agree = float((Sa.argmax(1) == Sb.argmax(1)).mean())
    r_S = float(np.abs(Sa[ok] - Sb[ok]).max())
    print(f"  特徴の相対差   {r_f:.2e}   (float32 なので 1e-5 程度までは正常)")
    print(f"  統計量 A の相対差 {r_A:.2e} | B の相対差 {r_B:.2e}")
    print(f"  スコアの最大差 {r_S:.2e} | argmax の一致率 {agree * 100:.1f}%")
    good = r_f < 1e-3 and r_A < 1e-3 and r_B < 1e-3 and agree >= 0.98
    print("  判定:", "OK (gpu 指定で実行して大丈夫です)" if good else "NG (数値が合いません。出力をそのまま共有してください)")


# ---------------------------------------------------------------- 評価
def softmax_nll(S, y, tau):
    Z = S / tau
    Z = Z - np.max(Z, axis=1, keepdims=True)
    logp = Z - np.log(np.exp(Z).sum(1, keepdims=True))
    return float(-logp[np.arange(len(y)), y].mean())


def best_tau(S, y):
    grid = np.logspace(-3, 0, 13)
    nlls = [softmax_nll(S, y, t) for t in grid]
    j = int(np.argmin(nlls))
    return float(grid[j]), nlls[j]


def _eval_windows(model, text, limit):
    """評価に使う窓。文脈の全文字と正解が学習語彙にある窓だけ(ブロックの継ぎ目 \\x00 は未知文字として自然に除かれる)。"""
    text = text[:limit]
    ids = np.array([model.idx.get(c, -1) for c in text], dtype=np.int64)
    X, y = model._windows(ids + 1)
    y = y - 1
    keep = (y >= 0) & (X > 0).all(1)
    return X[keep], y[keep]


def metrics(model, text, limit=20000, block=2000, nll=True):
    """返り値: (acc1, acc5, nll, tau, n)。スコア行列 (窓数 x 語彙) を block 行ずつ処理するので、語彙が数千でもメモリは小さい。"""
    X, y = _eval_windows(model, text, limit)
    n = len(X)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    taus = np.logspace(-3, 0, 13)
    c1 = c5 = 0
    nll_sum = np.zeros(len(taus))
    for s0 in range(0, n, block):
        Sb = model.scores(X[s0:s0 + block])
        yb = y[s0:s0 + block]
        V = Sb.shape[1]
        k = min(5, V)
        c1 += int((Sb.argmax(1) == yb).sum())
        top = np.argpartition(Sb, V - k, axis=1)[:, V - k:]
        c5 += int((top == yb[:, None]).any(1).sum())
        if not nll:
            continue
        Sf = np.where(np.isfinite(Sb), Sb, -1e3)
        rows = np.arange(len(yb))
        for t_i, t in enumerate(taus):
            Z = Sf / t
            mx = Z.max(1, keepdims=True)
            logZ = mx[:, 0] + np.log(np.exp(Z - mx).sum(1))
            nll_sum[t_i] += float((logZ - Z[rows, yb]).sum())
    if not nll:
        return 100.0 * c1 / n, 100.0 * c5 / n, float("nan"), float("nan"), n
    nlls = nll_sum / n
    j = int(np.argmin(nlls))
    return 100.0 * c1 / n, 100.0 * c5 / n, float(nlls[j]), float(taus[j]), n


def sweep(model, train, val, lams=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), verbose=True):
    """正則化の強さを、較正後の NLL (softmax と頻度のブレンド) で選ぶ。
    素の softmax の NLL は、スコアが確率でないため値が壊れやすく(一様分布並みになる、tau が探索上限に張り付く)、選択基準に向かない。"""
    best = None
    for lam in lams:
        model.solve(lam)
        a1, a5, _, _, _ = metrics(model, val, nll=False)
        t1 = metrics(model, train, nll=False)[0]
        nll, mix, tau = calibrated_nll(model, val)
        if verbose:
            print(f"  lam_rel={lam:<7g} train acc1 {t1:5.1f} | val acc1 {a1:5.1f} acc5 {a5:5.1f} nll {nll:.3f} (tau={tau:.3g}, 頻度とのブレンド a={mix})")
        if best is None or nll < best["nll"]:
            best = dict(lam=lam, nll=nll, tau=tau, mix=mix, acc1=a1, acc5=a5, train1=t1)
    model.solve(best["lam"])
    model.tau = best["tau"]
    return best


# ---------------------------------------------------------------- デモ
def unigram_nll(train, val):
    """参考値: 文脈を使わず出現頻度だけで予測したときの val の NLL (nat/文字)。"""
    from collections import Counter
    c = Counter(train.replace("\x00", ""))
    total, V = sum(c.values()), len(c)
    ys = [ch for ch in val if ch in c]
    return float(-np.mean([np.log((c[ch] + 0.5) / (total + 0.5 * V)) for ch in ys]))


def ngram_baseline(train, val, ctx_len, orders=(2, 3, 4, 6), limit=20000):
    """参考: 出現を数えるだけの n-gram (Witten-Bell 補間)。足し算で追加学習できる点はリザバーと同じ。
    評価する窓は metrics() と同じ(文脈の全文字と正解が学習語彙にある窓だけ)。"""
    from collections import Counter, defaultdict
    orders = tuple(n for n in orders if n <= ctx_len)  # 窓より長い文脈は使えない
    seams = "\x00" in train
    uni = Counter(train.replace("\x00", ""))
    vocab = set(uni)
    N, V = sum(uni.values()), len(vocab)
    top_uni = [c for c, _ in uni.most_common(5)]
    pu = lambda w: (uni[w] + 0.5) / (N + 0.5 * V)
    val = val[:limit]
    wins = [(val[i:i + ctx_len], val[i + ctx_len]) for i in range(len(val) - ctx_len)]
    wins = [(h, w) for h, w in wins if w in vocab and all(c in vocab for c in h)]
    rows = []
    for n in orders:
        cnt = {k: defaultdict(Counter) for k in range(1, n + 1)}
        for i in range(len(train) - n):
            if seams and "\x00" in train[i:i + n + 1]:
                continue
            for k in range(1, n + 1):
                cnt[k][train[i + n - k:i + n]][train[i + n]] += 1
        tot = {k: {h: sum(c.values()) for h, c in cnt[k].items()} for k in cnt}

        def prob(w, h):
            p = pu(w)
            for k in range(1, n + 1):
                c = cnt[k].get(h[-k:])
                if c is None:
                    break
                T = len(c)
                p = (c.get(w, 0) + T * p) / (tot[k][h[-k:]] + T)
            return p

        a1 = a5 = 0
        nll = 0.0
        for h, w in wins:
            cand = set(top_uni)
            for k in range(1, n + 1):
                c = cnt[k].get(h[-k:])
                if c is None:
                    break
                cand |= set(c)
            ranked = sorted(cand, key=lambda x: -prob(x, h))[:5]
            a1 += ranked[0] == w
            a5 += w in ranked
            nll -= np.log(prob(w, h))
        m = max(len(wins), 1)
        rows.append((n, 100.0 * a1 / m, 100.0 * a5 / m, nll / m))
    return rows


def calibrated_nll(model, val, limit=20000, block=2000):
    """リザバーのスコアを『softmax と頻度のブレンド』で確率にして NLL を測る(a, tau は val で選ぶ)。
    スコアは最小二乗の出力で確率ではないため、そのままの softmax は外した時に損が大きい。
    返り値: (nll, a, tau)。精度は変わらない(順位は同じ)。"""
    X, y = _eval_windows(model, val, limit)
    n = len(y)
    taus = np.logspace(-2.5, -0.5, 9)
    Sy = np.empty(n)
    logZ = np.empty((n, len(taus)))
    for s0 in range(0, n, block):
        Sb = model.scores(X[s0:s0 + block])
        Sb = np.where(np.isfinite(Sb), Sb, -1e3)
        yb = y[s0:s0 + block]
        b = len(yb)
        Sy[s0:s0 + b] = Sb[np.arange(b), yb]
        for t_i, t in enumerate(taus):
            Z = Sb / t
            mx = Z.max(1, keepdims=True)
            logZ[s0:s0 + b, t_i] = mx[:, 0] + np.log(np.exp(Z - mx).sum(1))
    V = len(model.chars)
    cnt = np.maximum(model.count[:V], 0).astype(np.float64)
    uy = ((cnt + 0.5) / (cnt.sum() + 0.5 * V))[y]
    best = (float("inf"), None, None)
    for t_i, t in enumerate(taus):
        ps = np.exp(Sy / t - logZ[:, t_i])
        for a in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0):
            nll = float(-np.log(a * ps + (1 - a) * uy + 1e-300).mean())
            if nll < best[0]:
                best = (nll, a, float(t))
    return best


def split_text(text, mode="scatter", k=4, block=500, every=7, docs=None):
    """学習/検証の分け方。
    scatter: 全体を block 文字ずつに区切り、every 個に1個を検証にする(約14%)。検証が全体に散らばるので、
             本の後半だけ文体や話題が違う、といった偏りの影響を受けにくい。
    doc    : ファイルの1行=1文書とみなし、文書単位で7個に1個を検証にする(Web由来のコーパス向け。
             同じ文書の断片が学習と検証の両方に入る漏れを防ぐ)。
    tail   : 従来どおり、末尾15%を検証にする。
    返り値: (chunks, train, val)。chunks は k 個のブロック列(各ブロックを別々に学習するので継ぎ目の偽の文脈が入らない)。
            train / val は評価用の文字列で、ブロックの継ぎ目に \\x00 が入る(未知の文字として評価から自動的に除かれる)。"""
    if mode == "doc":
        val_d = [d for i, d in enumerate(docs) if i % every == every // 2]
        train_d = [d for i, d in enumerate(docs) if i % every != every // 2]
        step = len(train_d) // k
        chunks = [train_d[i * step:(i + 1) * step if i < k - 1 else len(train_d)] for i in range(k)]
        return chunks, "\x00".join(train_d), "\x00".join(val_d)
    if mode == "tail":
        cut = int(len(text) * 0.85)
        step = cut // k
        chunks = [[text[i * step:(i + 1) * step if i < k - 1 else cut]] for i in range(k)]
        return chunks, text[:cut], text[cut:]
    blocks = [text[i:i + block] for i in range(0, len(text), block)]
    val_b = [b for i, b in enumerate(blocks) if i % every == every // 2]
    train_b = [b for i, b in enumerate(blocks) if i % every != every // 2]
    step = len(train_b) // k
    chunks = [train_b[i * step:(i + 1) * step if i < k - 1 else len(train_b)] for i in range(k)]
    return chunks, "\x00".join(train_b), "\x00".join(val_b)


def fit_chunk(model, blocks, weight=1.0):
    for b in blocks:
        model.fit_text(b, weight)


def demo(path, grid, max_chars, scales=(1, 2, 4, 8), mode="scatter", gpu=False):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    docs = None
    if mode == "doc":
        docs, tot = [], 0
        for ln in raw.splitlines():
            ln = ln.strip()
            if ln:
                docs.append(ln)
                tot += len(ln)
                if max_chars and tot >= max_chars:
                    break
        text = "".join(docs)
    else:
        text = raw.strip().replace("\n", "")
        if max_chars:
            text = text[:max_chars]
    k = 4
    chunks, train, val = split_text(text, mode, k, docs=docs)
    Model = TorchReservoirLM if gpu else OpticalReservoirLM
    mk = lambda **kw: Model(grid=grid, scales=scales, **kw)
    model = mk()
    big = gpu and model.D > BIG_D
    ntr, nva = len(train.replace("\x00", "")), len(val.replace("\x00", ""))
    print(f"文字数 {len(text)} | 学習 {ntr} (={k}分割) | 検証 {nva} | 検証の取り方: {dict(tail='末尾15%', doc='文書単位(1行=1文書)').get(mode, '全体に散らす(500文字ブロック)')}")
    print(f"露光 {model.scales} x {grid}x{grid}画素 = 特徴 {model.D - 1}個 | 参考: 頻度だけの予測 val nll = {unigram_nll(train, val):.3f}")

    print("\n[1] 逐次の追加学習 (チャンクを順に足す。各段階で解き直す)")
    print(f"{'段階':<10}{'val acc1':>9}{'val acc5':>9}{'val nll(較正後)':>16}{'所要s':>8}")
    for i, ch in enumerate(chunks, 1):
        t0 = time.time()
        fit_chunk(model, ch)
        model.solve(1e-2)
        a1, a5, _, _, _ = metrics(model, val, nll=False)
        nll = calibrated_nll(model, val)[0]
        print(f"+chunk{i:<5}{a1:>9.1f}{a5:>9.1f}{nll:>16.3f}{time.time() - t0:>8.1f}")
    print("  注: 解は「ここまでの全データで同時に学習した結果」と厳密に同じで、足す順序にも依存しない。")

    print("\n[2] 正則化の強さを検証データで選ぶ")
    best = sweep(model, train, val)
    print(f"  -> lam_rel={best['lam']:g} を採用")
    cn, ca, ct = best["nll"], best["mix"], best["tau"]
    print(f"  採用: 確率として較正(softmax と頻度のブレンド a={ca}, tau={ct:.3g}) -> val nll {cn:.3f}"
          f"  (参考: 頻度だけ {unigram_nll(train, val):.3f})")

    print("\n[2b] 参考: 数えるだけの n-gram (同じ学習/検証データ。これがリザバーの比較相手。『文脈k文字』で次の1文字を当てる)")
    print(f"{'構成':<22}{'val acc1':>9}{'val acc5':>9}{'val nll':>9}")
    for n, a1, a5, nl in ngram_baseline(train, val, model.T):
        print(f"{'文脈' + str(n) + '文字(WB補間)':<20}{a1:>9.1f}{a5:>9.1f}{nl:>9.3f}")
    print(f"{'リザバー(今回)':<20}{best['acc1']:>9.1f}{best['acc5']:>9.1f}{cn:>9.3f}")

    print("\n[3] 比較: 同じ総特徴数の『単一スケール(8文字だけ)』と何が違うか")
    g1 = int(round(grid * np.sqrt(model.E)))
    if big:
        print(f"  (特徴数 {model.D - 1} が大きいので、メモリ節約のため省略)")
    else:
        base = Model(grid=g1, scales=(model.T,))
        for ch in chunks:
            fit_chunk(base, ch)
        bb = sweep(base, train, val, verbose=False)
        print(f"{'構成':<26}{'特徴数':>8}{'train acc1':>11}{'val acc1':>9}{'val acc5':>9}{'val nll':>9}")
        print(f"{'単一(' + str(model.T) + ')':<26}{base.D - 1:>8}{bb['train1']:>11.1f}{bb['acc1']:>9.1f}{bb['acc5']:>9.1f}{bb['nll']:>9.3f}")
        print(f"{'多重' + str(model.scales):<24}{model.D - 1:>8}{best['train1']:>11.1f}{best['acc1']:>9.1f}{best['acc5']:>9.1f}{best['nll']:>9.3f}")
        del base

    if gpu:
        print("\n[4][5] 融合と取り消しの検証は、gpu 指定のときは省略します (メモリ節約。numpy 版で 1e-15 の一致を確認済み)")
    else:
        print("\n[4] 再学習なしの融合 (前半チャンクのモデルA + 後半チャンクのモデルB)")
        mA, mB = mk(), mk()
        for ch in chunks[:k // 2]:
            fit_chunk(mA, ch)
        for ch in chunks[k // 2:]:
            fit_chunk(mB, ch)
        merged = mA.merge(mB)
        del mB
        print(f"  統計量 A の相対差 {np.abs(merged.A - model.A).max() / np.abs(model.A).max():.2e}  (0に近いほど『全部で学習したのと同じ』)")
        del merged, mA

        print("\n[5] 取り消し (chunk4 を weight=-1 で引く -> chunk1〜3 だけで学習したモデルと一致するか)")
        m3 = mk()
        for ch in chunks[:3]:
            fit_chunk(m3, ch)
        m4 = copy.deepcopy(model)
        fit_chunk(m4, chunks[3], -1.0)
        print(f"  A の相対差 {np.abs(m4.A - m3.A).max() / np.abs(m3.A).max():.2e}")
        del m3, m4

    print("\n[6] 保存と読み込み")
    if big:
        print("  (特徴数が大きいので、統計量つきの保存は省略)")
    else:
        model.save("reservoir08_state.npz", with_stats=True)
        print(f"  追加学習用(統計量つき) {os.path.getsize('reservoir08_state.npz') / 1e6:.1f} MB")
    model.save("reservoir08_infer.npz", with_stats=False)
    print(f"  推論専用(Wのみ) {os.path.getsize('reservoir08_infer.npz') / 1e6:.1f} MB")
    re = Model.load("reservoir08_infer.npz")
    prompt = text[:model.T]
    print(f"  読み込み後の生成が一致: {re.generate(prompt, 30, tau=0) == model.generate(prompt, 30, tau=0)}")

    print(f"\n[7] 生成 (プロンプト『{prompt}』)")
    print("  argmax :", model.generate(prompt, 100, tau=0))
    print("  sample :", model.generate(prompt, 100, tau=model.tau, top_k=5))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "gpu"]
    if args and args[0] == "selftest":
        selftest()
    else:
        demo(args[0] if len(args) > 0 else "sample_corpus.txt",
             int(args[1]) if len(args) > 1 else 32,
             int(args[2]) if len(args) > 2 else 0,
             tuple(int(x) for x in args[3].split(",")) if len(args) > 3 else (1, 2, 4, 8),
             args[4] if len(args) > 4 else "scatter",
             gpu="gpu" in sys.argv[1:])
