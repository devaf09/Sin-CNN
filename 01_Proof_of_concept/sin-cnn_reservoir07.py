"""
sin-cnn_reservoir07.py : 学習しない光学リザバー + 閉形式の読み出し (numpy だけで動く)

設計の考え方
  [固定の物理]  文字ごとの位相ホログラム + 位置ごとの位相板 -> コヒーレントに重ね合わせ
                -> (ランダム位相板 -> 自由空間伝搬) x L 層 -> カメラ(強度検出)
                この部分は乱数で決めた固定のもの。学習しない。
  [学習する部分] カメラ画素の強度 phi から次の文字を当てる線形読み出し W だけ。
                最小二乗(リッジ)の正規方程式  (A + reg) W = B  を解く。
                    A = sum phi^T phi ,  B = sum phi^T onehot(y)
                A, B は「足し算」で更新できるので
                  * 追加学習   : 新しい文章の A, B を足すだけ(全データで学習し直した結果と厳密に一致)
                  * 忘却なし   : 過去の統計量は消えない
                  * 融合       : 2つのモデルの A, B を足すだけ(再学習なしで、両方のデータで学習したのと同じ)
                  * 取り消し   : weight=-1 で同じ文章を引き算できる
                  * 語彙の追加 : 文字パターンは文字コードから決まるので、新しい文字も再学習不要
  [物理への対応] 位相板=SLM/回折素子、伝搬=自由空間、強度検出=カメラ。
                音でも「遅延板 + 伝搬 + エネルギー検出」で同じ数式になる。

使い方:  python sin-cnn_reservoir07.py sample_corpus.txt [grid=48]
"""
import copy
import json
import math
import sys
import time

import numpy as np

PAD_KEY = 0x7FFFFFFF  # 「文脈が足りない」ことを表す予約パターンの乱数キー


class OpticalReservoirLM:
    def __init__(self, grid=48, ctx=8, layers=3, dist=5e-3, wavelength=633e-9, pitch=20e-6, seed=0):
        self.cfg = dict(grid=grid, ctx=ctx, layers=layers, dist=dist,
                        wavelength=wavelength, pitch=pitch, seed=seed)
        G, N, T = grid, grid * grid, ctx
        self.G, self.N, self.T, self.seed = G, N, T, seed
        rng = lambda *key: np.random.default_rng([seed, *key])

        # ---- 固定の物理 (学習しない) ----
        self.pos = np.exp(1j * np.stack([rng(1, p).uniform(0, 2 * np.pi, N) for p in range(T)]))   # 位置ごとの位相板
        self.mask = np.exp(1j * np.stack([rng(2, l).uniform(0, 2 * np.pi, (G, G)) for l in range(layers)]))
        f = np.fft.fftfreq(G, d=pitch)  # fft の並びそのままなので ifftshift 問題は起きない
        FX, FY = np.meshgrid(f, f, indexing="ij")
        k = 2 * np.pi / wavelength
        kz = np.sqrt(np.maximum(k ** 2 - (2 * np.pi * FX) ** 2 - (2 * np.pi * FY) ** 2, 0.0))
        self.H = np.exp(1j * kz * dist)

        # ---- 語彙 (文字 -> 位相ホログラム。行0は PAD) ----
        self.chars, self.idx = [], {}
        self.table = np.exp(1j * rng(3, PAD_KEY).uniform(0, 2 * np.pi, N))[None, :]

        # ---- 学習する統計量 ----
        self.A = np.zeros((N + 1, N + 1))       # sum phi^T phi   (最後の列/行はバイアス)
        self.Bt = np.zeros((0, N + 1))          # sum onehot^T phi (クラス x 特徴)
        self.count = np.zeros(0, dtype=np.int64)  # クラスごとの学習例数(符号付き)
        self.n = 0.0
        self.W = None                           # (N+1, V) 解いた読み出し重み
        self.tau = 0.05                         # softmax の温度(サンプリング用)

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
        self.Bt = np.vstack([self.Bt, np.zeros((len(new), self.N + 1))])
        self.count = np.concatenate([self.count, np.zeros(len(new), dtype=np.int64)])

    def encode(self, text):
        self._add_chars(list(text))
        return np.array([self.idx[c] for c in text], dtype=np.int64)

    def rows_for(self, text):  # 予測用: 知らない文字は PAD 扱い (語彙は増やさない)
        return np.array([self.idx[c] + 1 if c in self.idx else 0 for c in text], dtype=np.int64)

    def _windows(self, ids_rows):
        n = len(ids_rows) - self.T
        if n <= 0:
            return np.zeros((0, self.T), dtype=np.int64), np.zeros(0, dtype=np.int64)
        i = np.arange(n)[:, None] + np.arange(self.T)[None, :]
        return ids_rows[i], ids_rows[self.T:self.T + n]

    # ------------------------------------------------------------ 光学 (固定)
    def features(self, rows, batch=128):
        """rows: (n, T) の table 行番号 -> カメラ強度 (n, N+1)。最後の列は定数1。"""
        n = len(rows)
        out = np.empty((n, self.N + 1))
        out[:, -1] = 1.0
        for s in range(0, n, batch):
            xb = rows[s:s + batch]
            U = (self.table[xb] * self.pos[None]).sum(1).reshape(-1, self.G, self.G)  # コヒーレント重ね合わせ
            for l in range(self.cfg["layers"]):
                U = U * self.mask[l]
                U = np.fft.ifft2(np.fft.fft2(U) * self.H)
            I = (U.real ** 2 + U.imag ** 2).reshape(len(xb), -1)
            out[s:s + len(xb), :-1] = I / self.T - 1.0  # 期待強度 T を引いて平均0に
        return out

    # ------------------------------------------------------------ 学習 (足し算)
    def fit_text(self, text, weight=1.0, batch=128):
        """文章を追加学習する。weight=-1 でその文章を取り消す。"""
        ids = self.encode(text)
        X, y = self._windows(ids + 1)
        y = y - 1
        for s in range(0, len(X), batch):
            Phi = self.features(X[s:s + batch])
            self.A += weight * (Phi.T @ Phi)
            np.add.at(self.Bt, y[s:s + batch], weight * Phi)
            np.add.at(self.count, y[s:s + batch], 1 if weight > 0 else -1)
        self.n += weight * len(X)
        return len(X)

    def solve(self, lam_rel=1e-2):
        """正規方程式を解く。lam_rel は A の平均対角に対する相対的な正則化の強さ。"""
        scale = np.trace(self.A) / (self.N + 1)
        reg = np.full(self.N + 1, lam_rel * scale)
        reg[-1] = 1e-8 * scale
        self.W = np.linalg.solve(self.A + np.diag(reg), self.Bt.T)
        return self.W

    # ------------------------------------------------------------ 予測
    def scores_from_features(self, Phi):
        S = Phi @ self.W
        if S.shape[1] < len(self.chars):  # solve 後に語彙が増えた場合
            S = np.hstack([S, np.zeros((len(S), len(self.chars) - S.shape[1]))])
        S[:, self.count[:S.shape[1]] <= 0] = -np.inf
        return S

    def scores(self, rows):
        return self.scores_from_features(self.features(rows))

    def generate(self, prompt, length=100, tau=None, top_k=5, seed=0):
        tau = self.tau if tau is None else tau
        rng = np.random.default_rng(seed)
        ctx = list(prompt)
        out = list(prompt)
        for _ in range(length):
            window = ctx[-self.T:]
            rows = self.rows_for("".join(window))
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
        """other の知識を足し込む(再学習なし)。物理(seed等)が同じモデル同士のみ。"""
        assert self.cfg == other.cfg, "物理設定(grid/ctx/layers/dist/seed...)が違うモデルは融合できません"
        self._add_chars(other.chars)
        for j, c in enumerate(other.chars):
            self.Bt[self.idx[c]] += other.Bt[j]
            self.count[self.idx[c]] += other.count[j]
        self.A += other.A
        self.n += other.n
        return self

    def save(self, path, with_stats=True):
        """with_stats=False なら推論専用の軽量ファイル(W を float32 で保存)。"""
        data = dict(cfg=json.dumps(self.cfg), chars=np.array(self.chars), count=self.count,
                    W=self.W.astype(np.float32), tau=self.tau)
        if with_stats:
            data.update(A=self.A, Bt=self.Bt, n=self.n)
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
            m.A, m.Bt, m.n = d["A"], d["Bt"], float(d["n"])
        return m


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


def metrics(model, text):
    """知らない文字を含む窓は評価から除く。返り値: (acc1, acc5, nll, tau, n)"""
    ids = np.array([model.idx.get(c, -1) for c in text], dtype=np.int64)
    X, y = model._windows(ids + 1)
    y = y - 1
    keep = (y >= 0) & (X > 0).all(1)  # 文脈の全文字と正解が既知の窓だけ
    X, y = X[keep], y[keep]
    if len(X) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    S = model.scores(X)
    top5 = np.argsort(S, axis=1)[:, -5:]
    acc1 = float((S.argmax(1) == y).mean() * 100)
    acc5 = float((top5 == y[:, None]).any(1).mean() * 100)
    tau, nll = best_tau(np.where(np.isfinite(S), S, -1e3), y)
    return acc1, acc5, nll, tau, len(X)


# ---------------------------------------------------------------- デモ
def demo(path, grid):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip().replace("\n", "")
    cut = int(len(text) * 0.85)
    train, val = text[:cut], text[cut:]
    k = 4
    step = len(train) // k
    chunks = [train[i * step:(i + 1) * step if i < k - 1 else len(train)] for i in range(k)]
    print(f"文字数 {len(text)} | 学習 {len(train)} (={k}分割) | 検証 {len(val)} | grid={grid} ({grid * grid}画素)")

    model = OpticalReservoirLM(grid=grid)
    print("\n[1] 逐次の追加学習 (チャンクを順に足していく。各段階で解き直す)")
    print(f"{'段階':<10}{'val acc1':>9}{'val acc5':>9}{'val nll':>9}{'チャンク1 acc1':>16}{'所要s':>7}")
    for i, ch in enumerate(chunks, 1):
        t0 = time.time()
        model.fit_text(ch)
        model.solve(1e-2)
        a1, a5, nll, _, _ = metrics(model, val)
        c1, _, _, _, _ = metrics(model, chunks[0])
        print(f"+chunk{i:<5}{a1:>9.1f}{a5:>9.1f}{nll:>9.3f}{c1:>16.1f}{time.time() - t0:>7.1f}")
    print("  注: 解は「ここまでの全データで同時に学習した結果」と厳密に同じで、足す順序にも依存しない(忘却は起きない)。")
    print("      'チャンク1 acc1' が下がるのは忘却ではなく、画素数Nぶんの容量を新しいデータと分け合うため。grid を上げると緩和。")

    print("\n[2] 正則化の強さを検証データで選ぶ (A, B は共通なので解き直すだけ)")
    best = None
    for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        model.solve(lam)
        a1, a5, nll, tau, _ = metrics(model, val)
        tr1, *_ = metrics(model, train)
        print(f"  lam_rel={lam:<7g} train acc1 {tr1:5.1f} | val acc1 {a1:5.1f} acc5 {a5:5.1f} nll {nll:.3f} (tau={tau:.3g})")
        if best is None or nll < best[1]:
            best = (lam, nll, tau)
    model.solve(best[0])
    model.tau = best[2]
    print(f"  -> lam_rel={best[0]:g} を採用")

    print("\n[3] 再学習なしの融合 (前半チャンクのモデルA + 後半チャンクのモデルB)")
    mA, mB = OpticalReservoirLM(grid=grid), OpticalReservoirLM(grid=grid)
    for ch in chunks[:k // 2]:
        mA.fit_text(ch)
    for ch in chunks[k // 2:]:
        mB.fit_text(ch)
    merged = mA.merge(mB)
    dA = np.abs(merged.A - model.A).max() / np.abs(model.A).max()
    merged.solve(best[0])
    pairs = [(merged.idx[c], model.idx[c]) for c in model.chars if c in merged.idx]
    X, _ = model._windows(model.rows_for(val))
    Sm, Sj = merged.scores(X[:300]), model.scores(X[:300])
    dS = max(np.nanmax(np.abs(Sm[:, a] - Sj[:, b])[np.isfinite(Sm[:, a] - Sj[:, b])]) for a, b in pairs[:50])
    print(f"  統計量 A の相対差 {dA:.2e} | 予測スコアの最大差 {dS:.2e}  (0に近いほど『全部で学習したのと同じ』)")

    print("\n[4] 取り消し (chunk4 を weight=-1 で引き算 -> chunk1〜3 だけで学習したモデルと一致するか)")
    m3 = OpticalReservoirLM(grid=grid)
    for ch in chunks[:3]:
        m3.fit_text(ch)
    m4 = copy.deepcopy(model)
    m4.fit_text(chunks[3], weight=-1.0)
    print(f"  A の相対差 {np.abs(m4.A - m3.A).max() / np.abs(m3.A).max():.2e}")

    print("\n[5] 保存と読み込み")
    model.save("reservoir07_state.npz", with_stats=True)
    model.save("reservoir07_infer.npz", with_stats=False)
    import os
    print(f"  追加学習用(統計量つき) {os.path.getsize('reservoir07_state.npz') / 1e6:.1f} MB"
          f" | 推論専用(Wのみ) {os.path.getsize('reservoir07_infer.npz') / 1e6:.1f} MB")
    re = OpticalReservoirLM.load("reservoir07_infer.npz")
    prompt = text[:model.T]
    print(f"  読み込み後の生成が一致: {re.generate(prompt, 30, tau=0) == model.generate(prompt, 30, tau=0)}")

    print(f"\n[6] 生成 (プロンプト『{prompt}』)")
    print("  argmax :", model.generate(prompt, 100, tau=0))
    print("  sample :", model.generate(prompt, 100, tau=model.tau, top_k=5))


if __name__ == "__main__":
    demo(sys.argv[1] if len(sys.argv) > 1 else "sample_corpus.txt",
         int(sys.argv[2]) if len(sys.argv) > 2 else 48)
