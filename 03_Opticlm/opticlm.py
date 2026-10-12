"""
opticlm.py : 光学リザバー言語モデル (単一ファイル。numpy だけで動き、torch があれば GPU でも動く)

しくみ
  * 固定の「光学媒質」(乱数の位相板 + 自由空間伝播 + 強度検出) で直近の数文字を特徴 Φ にし、
    学習は線形の読み出し W だけ (閉形式のリッジ回帰: (A + λI) W = B、A = ΣΦᵀΦ, B = ΣΦᵀY)。
  * 学習が足し算なので、追加学習・取り消し・融合・並列学習が厳密にできる。
  * 光学媒質は seed から再現できるので、保存するのは読み出し W だけ。

検証の結論 (詳細は RESULTS.md。CC-100 日本語 1000万文字)
  * 言語モデルとしては n-gram (特に Kneser-Ney) に同じバイト数で負ける。読み出しの容量 (D×語彙) で頭打ちになる。
  * 光学の特徴は、同じ次元のデジタルのランダム特徴と同等か、わずかに良い。回折 (位相板 + 伝播) は必須。
  * 粗いアナログ素子で動く: W 4〜5bit (グループ量子化)、素子ばらつき 10%、検出器 光子100/画素・ADC 6bit で nll +0.04 程度。

使い方 (Python から)
    import opticlm
    lm = opticlm.OpticalLM(grid=32, scales=(1, 2, 3, 4), backend="auto")   # auto: GPU があれば GPU
    lm.fit(["文書1", "文書2"])            # 追加学習 (リストの1要素=1文書。文書をまたぐ文脈は作らない)
    lm.solve(1e-3)                        # 読み出しを解く。追加学習のたびに解き直す
    opticlm.calibrate(lm, val_docs)       # 確率の較正 (検証用の文書で1回)
    lm.predict("今日は", k=5)             # [(文字, 確率), ...]
    lm.generate("今日は", 80)
    lm.forget("取り消したい文章")         # fit と同じ文章を渡すと厳密に取り消せる
    lm.save("model.npz")                  # 統計量つき (追加学習できる)
    lm.save("infer.npz", stats=False, w_bits=4, w_group=256)   # 推論専用 (4bit + グループごとのスケール)
    lm = opticlm.load("model.npz")
    lm, report = opticlm.train("corpus.txt", grid=64)          # ファイル (1行=1文書) から分割・学習・較正まで

コマンドライン
    python opticlm.py selftest                                  # 動作確認 (GPU 実行の前に1回)
    python opticlm.py train corpus.txt model.npz [--grid 64] [--gpu | --workers 3]
    python opticlm.py eval  model.npz corpus.txt
    python opticlm.py gen   model.npz "プロンプト" [--n 100]
    python opticlm.py bench corpus.txt [--d 4096,16384] [--gpu]   # 同じバイト数での n-gram との比較と、混合の比較

物理設定 (cfg。同じ cfg のモデル同士だけ融合できる)
    grid, scales   : 露光ごとの画素の一辺 / 各露光が見る直近の文字数 (総特徴数 D = 露光数 × grid² + 1)
    layers, dist, wavelength, pitch, seed : 位相板の枚数、層間の距離、波長、画素ピッチ、乱数の種
    detector       : {"photons": 画素あたりの平均光子数, "read": 読み出し雑音 (平均強度比), "adc_bits": ADC のビット数}
                     None で理想的な検出。雑音 (photons / read) があると取り消しは「ほぼ」になる。

注意
  * GPU 版の既定 (acc="fp32") は 1バッチ分の ΦᵀΦ を float32 で計算して float64 の A に足す。取り消し・融合の誤差は 1e-6〜1e-7。
    1e-16 の厳密さが要るなら numpy 版か acc="fp64"。
  * 以前の版にあった増強オプション (tails / ref / taps / bag) は、実日本語で効果が無かったので削除した (RESULTS.md)。
    それらを使って保存したモデルは読み込めない (既定値のまま保存したモデルは読み込める)。
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np

try:
    import torch
except Exception:  # torch が無くても numpy 版は動く
    torch = None

__all__ = ["OpticalLM", "TorchOpticalLM", "NgramLM", "load", "train", "evaluate", "calibrate", "sweep_lam",
           "quantize_matrix", "lowrank_matrix", "size_bench", "mix_compare", "mixture_nll", "selftest", "split_docs", "read_docs",
           "fit_parallel", "grid_for_D"]

PAD_KEY = 0x7FFFFFFF
SEAM = "\x00"  # 文書の継ぎ目 (評価では未知文字として自動的に除かれる)
DET_DEFAULT = dict(photons=None, read=0.0, adc_bits=None, adc_fs=8.0)


def _canon(grid=32, scales=(1, 2, 3, 4), layers=3, dist=5e-3, wavelength=633e-9, pitch=20e-6, seed=0, detector=None,
           **legacy):
    """物理設定を正規化した dict にする。legacy: 以前の版の保存ファイルにある tails / ref / taps (既定値のときだけ受け付ける)。"""
    layers = int(layers)
    bad = {k: v for k, v in legacy.items()
           if not ((k == "tails" and not v) or (k == "ref" and not v) or (k == "taps" and (not v or list(v) == [layers])))}
    if bad:
        raise ValueError(f"この版では使えない設定です (増強オプションは削除済み): {bad}")
    det = dict(DET_DEFAULT)
    det.update(detector or {})
    return dict(grid=int(grid), scales=tuple(int(s) for s in scales), layers=layers, dist=float(dist),
                wavelength=float(wavelength), pitch=float(pitch), seed=int(seed), detector=det)


def grid_for_D(D, n_exposures):
    """総特徴数 D にほぼ等しくなる grid を返す。"""
    return max(2, int(round(math.sqrt((D - 1) / n_exposures))))


# ====================================================================== モデル本体 (numpy)
class OpticalLM:
    backend = "numpy"

    def __new__(cls, *a, backend="numpy", **kw):
        if cls is OpticalLM:
            b = backend
            if b == "auto":
                b = "torch" if (torch is not None and torch.cuda.is_available()) else "numpy"
            if b == "torch":
                cls = TorchOpticalLM
        return object.__new__(cls)

    def __init__(self, grid=32, scales=(1, 2, 3, 4), layers=3, dist=5e-3, wavelength=633e-9, pitch=20e-6,
                 seed=0, detector=None, backend="numpy", device=None, acc=None, **legacy):
        c = self.cfg = _canon(grid, scales, layers, dist, wavelength, pitch, seed, detector, **legacy)
        G, L, seed = c["grid"], c["layers"], c["seed"]
        self.G, self.N, self.seed = G, G * G, seed
        # 露光: 直近 T 文字を、位置ごとの位相板をかけて重ねる。norm = 期待強度 (= T)
        self.exps = [dict(T=s, norm=float(s)) for s in c["scales"]]
        self.E, self.T = len(self.exps), max(ex["T"] for ex in self.exps)
        self.D = self.E * self.N + 1
        rng = lambda *key: np.random.default_rng([seed, *key])

        # ---- 固定の物理 (学習しない) ----
        self.pos = [np.exp(1j * np.stack([rng(1, e, p).uniform(0, 2 * np.pi, self.N) for p in range(ex["T"])]))
                    for e, ex in enumerate(self.exps)]
        self.mask = [np.exp(1j * np.stack([rng(2, e, l).uniform(0, 2 * np.pi, (G, G)) for l in range(L)]))
                     for e in range(self.E)]
        f = np.fft.fftfreq(G, d=c["pitch"])  # fft の並びそのまま
        FX, FY = np.meshgrid(f, f, indexing="ij")
        k = 2 * np.pi / c["wavelength"]
        kz = np.sqrt(np.maximum(k ** 2 - (2 * np.pi * FX) ** 2 - (2 * np.pi * FY) ** 2, 0.0))
        self.H = np.exp(1j * kz * c["dist"])

        # ---- 語彙 (文字 -> 位相ホログラム。行0は PAD) ----
        self.chars, self.idx = [], {}
        self.table = np.exp(1j * rng(3, PAD_KEY).uniform(0, 2 * np.pi, self.N))[None, :]
        self._rng = np.random.default_rng([seed, 99])  # 検出器雑音用

        # ---- 学習する統計量 ----
        self._setup_backend()
        self._alloc_stats()
        self.count = np.zeros(0, dtype=np.int64)
        self.n = 0.0
        self.W = None
        self.lam = None
        self.cal = dict(kind="softmax", a=0.7, tau=0.03, nll=None)  # calibrate() で上書き
        self.calibrated = False

    # ------------------------------------------------------------ バックエンド (numpy)
    def _setup_backend(self):
        self.p_pos, self.p_mask, self.p_H = self.pos, self.mask, self.H
        self.p_table = self.table
        self.fft2, self.ifft2 = np.fft.fft2, np.fft.ifft2

    def _sync_table(self):
        self.p_table = self.table

    def _empty(self, shape):
        return np.empty(shape)

    def _to_idx(self, rows):
        return np.asarray(rows)

    def _poisson(self, x):
        return self._rng.poisson(x).astype(np.float64)

    def _randn(self, x):
        return self._rng.standard_normal(x.shape)

    def _clip(self, x, lo, hi):
        return np.clip(x, lo, hi)

    def _round(self, x):
        return np.rint(x)

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
        self._sync_table()
        self._grow_classes(len(new))
        self.count = np.concatenate([self.count, np.zeros(len(new), dtype=np.int64)])

    def encode(self, text):
        self._add_chars(list(text))
        return np.array([self.idx[c] for c in text], dtype=np.int64)

    def rows_for(self, text):
        return np.array([self.idx[c] + 1 if c in self.idx else 0 for c in text], dtype=np.int64)

    def _windows(self, rows, pad_start=False):
        """rows: 文字の table 行番号(1始まり)。返り値 X (n,T) と 正解 y (n,) (rows の値)。"""
        T = self.T
        if pad_start:  # 文書の先頭にも PAD の文脈つきで学習する (短いプロンプトからの予測・生成のため)
            P = np.concatenate([np.zeros(T, dtype=np.int64), rows])
            n = len(rows)
            y = rows
        else:
            P = rows
            n = len(rows) - T
            y = rows[T:]
        if n <= 0:
            return np.zeros((0, T), dtype=np.int64), np.zeros(0, dtype=np.int64)
        return P[np.arange(n)[:, None] + np.arange(T)[None, :]], y

    # ------------------------------------------------------------ 光学 (固定)。numpy / torch 共通の式
    def _detect(self, I, mean):
        d = self.cfg["detector"]
        if d["photons"]:
            ph = float(d["photons"])
            I = self._poisson(I / mean * ph) * (mean / ph)
        if d["read"]:
            I = I + self._randn(I) * (float(d["read"]) * mean)
        if d["adc_bits"]:
            lv = float(2 ** int(d["adc_bits"]) - 1)
            fs = float(d["adc_fs"]) * mean
            I = self._round(self._clip(I / fs * lv, 0.0, lv)) / lv * fs
        return I

    def _optics(self, xb):
        """xb: (n, T) の table 行番号 -> カメラ強度の特徴 (n, D)。最後の列は定数1 (バイアス)。"""
        c = self.cfg
        n, G, N = xb.shape[0], self.G, self.N
        out = self._empty((n, self.D))
        out[:, -1] = 1.0
        for e, ex in enumerate(self.exps):
            Te = ex["T"]
            U = None
            for k in range(Te):  # k=0 が直近の文字。位置の位相板の番号は古い方が 0 (08 と同じ)
                term = self.p_table[xb[:, -1 - k]] * self.p_pos[e][Te - 1 - k]
                U = term if U is None else U + term
            U = U.reshape(n, G, G)
            for l in range(c["layers"]):  # 位相板 -> 自由空間伝播 (角スペクトル法)
                U = self.ifft2(self.fft2(U * self.p_mask[e][l]) * self.p_H)
            I = self._detect(U.real ** 2 + U.imag ** 2, ex["norm"])
            out[:, e * N:(e + 1) * N] = (I.reshape(n, -1) - ex["norm"]) / ex["norm"]
        return out

    def features(self, rows, batch=128):
        rows = np.asarray(rows)
        n = len(rows)
        out = np.empty((n, self.D))
        for s0 in range(0, n, batch):
            out[s0:s0 + batch] = self._optics(self._to_idx(rows[s0:s0 + batch]))
        return out

    # ------------------------------------------------------------ 学習 (足し算)
    def _accumulate(self, Phi, y, weight):
        self.A += weight * (Phi.T @ Phi)
        Y = np.zeros((len(y), self.Bt.shape[0]))
        Y[np.arange(len(y)), y] = 1.0
        self.Bt += weight * (Y.T @ Phi)

    def _features_batch(self, X):
        return self._optics(self._to_idx(X))

    def _default_batch(self):
        return 128

    def fit(self, texts, weight=1.0, batch=None, pad_start=True):
        """追加学習。texts は文字列、または文書(文字列)のリスト。weight=-1 でその文章を取り消す。
        返り値: 使った窓の数。"""
        if isinstance(texts, str):
            texts = [texts]
        batch = batch or self._default_batch()
        buf_x, buf_y, nbuf, total = [], [], 0, 0

        def flush():
            nonlocal buf_x, buf_y, nbuf
            if not nbuf:
                return
            X, y = np.concatenate(buf_x), np.concatenate(buf_y)
            for s0 in range(0, len(X), batch):
                yb = y[s0:s0 + batch]
                self._accumulate(self._features_batch(X[s0:s0 + batch]), yb, weight)
            self.count += (1 if weight > 0 else -1) * np.bincount(y, minlength=len(self.count))
            buf_x, buf_y, nbuf = [], [], 0

        for t in texts:
            if not t:
                continue
            rows = self.encode(t) + 1
            for j0 in range(0, len(rows), 65536):  # 長い文書は区切って窓を作る (継ぎ目の文脈は保つ)
                seg = rows[max(0, j0 - self.T):j0 + 65536]
                if j0 == 0:
                    X, y = self._windows(seg, pad_start)
                else:
                    X, y = self._windows(seg, False)
                if len(X):
                    buf_x.append(X)
                    buf_y.append(y - 1)
                    nbuf += len(X)
                    total += len(X)
            if nbuf >= 8 * batch:
                flush()
        flush()
        self.n += weight * total
        return total

    def forget(self, texts, pad_start=True):
        """fit で足した文章を厳密に取り消す (同じ文章・同じ pad_start を渡すこと)。"""
        return self.fit(texts, weight=-1.0, pad_start=pad_start)

    fit_text = fit  # 08 互換

    def solve(self, lam_rel=1e-3):
        """(A + reg) W = B を解く。reg = lam_rel * trace(A)/D (バイアス項だけ 1e-8 倍)。"""
        scale = np.trace(self.A) / self.D
        reg = np.full(self.D, lam_rel * scale)
        reg[-1] = 1e-8 * scale
        M = self.A.copy()
        M[np.diag_indices(self.D)] += reg
        self.W = np.linalg.solve(M, self.Bt.T)
        self.lam = lam_rel
        return self.W

    # ------------------------------------------------------------ 予測
    def _fill_scores(self, S):
        V = len(self.chars)
        if S.shape[1] < V:
            S = np.hstack([S, np.zeros((len(S), V - S.shape[1]))])
        S[:, self.count[:S.shape[1]] <= 0] = -np.inf
        return S

    def scores(self, rows, batch=None):
        """rows: (n, T) の table 行番号 -> スコア (n, V)。最小二乗の出力で、確率ではない(順位づけ用)。未学習の文字は -inf。"""
        rows = np.asarray(rows)
        batch = batch or 512
        out = [self._scores_batch(rows[s0:s0 + batch]) for s0 in range(0, len(rows), batch)]
        S = np.concatenate(out) if out else np.zeros((0, self.W.shape[1]))
        return self._fill_scores(S)

    def _scores_batch(self, rows):
        return self._features_batch(rows) @ self.W

    def unigram(self):
        V = len(self.chars)
        cnt = np.maximum(self.count[:V], 0).astype(np.float64)
        return (cnt + 0.5) / (cnt.sum() + 0.5 * V)

    def probs_from_scores(self, S):
        """スコア -> 確率 (較正器で)。softmax(S/tau) または clip(S)^gamma を、頻度(ユニグラム)とブレンドする。"""
        c = self.cal
        valid = np.isfinite(S)
        if c["kind"] == "binpow":
            u = self.unigram()
            k = np.searchsorted(np.asarray(c["edges"]), np.where(valid, S, -np.inf).max(1))
            P = np.empty(S.shape)
            for j, (eps, g, beta, a) in enumerate(c["params"]):
                sel = k == j
                if sel.any():
                    q = np.where(valid[sel], np.maximum(S[sel], eps), 0.0) ** g * u[None, :] ** beta
                    P[sel] = a * q / q.sum(1, keepdims=True) + (1 - a) * u[None, :]
            return P
        if c["kind"] == "softmax":
            Z = np.where(valid, S, -np.inf) / c["tau"]
            Z = Z - Z.max(1, keepdims=True)
            q = np.exp(Z)
        else:
            q = np.where(valid, np.maximum(S, c["eps"]), 0.0) ** c["gamma"]
        q = q / q.sum(1, keepdims=True)
        return c["a"] * q + (1 - c["a"]) * self.unigram()[None, :]

    def _context_rows(self, prompt):
        rows = self.rows_for(prompt[-self.T:])
        return np.concatenate([np.zeros(self.T - len(rows), dtype=np.int64), rows])[None, :]

    def next_proba(self, prompt, forbid=""):
        """プロンプトの次の1文字の確率分布 (長さ V の配列)。forbid の文字は 0 にして正規化。"""
        p = self.probs_from_scores(self.scores(self._context_rows(prompt)))[0]
        for ch in forbid:
            if ch in self.idx:
                p[self.idx[ch]] = 0.0
        return p / p.sum()

    def predict(self, prompt, k=5, forbid=""):
        p = self.next_proba(prompt, forbid)
        top = np.argsort(-p)[:k]
        return [(self.chars[j], float(p[j])) for j in top]

    def nll(self, text, pad_start=True):
        """文章の平均 NLL (nat/文字) と評価した文字数。語彙に無い文字を含む位置は除く。"""
        rows = self.rows_for(text)
        X, y = self._windows(rows, pad_start)
        keep = (y > 0) & (X > 0).all(1) if not pad_start else (y > 0)
        X, y = X[keep], y[keep] - 1
        if len(y) == 0:
            return float("nan"), 0
        tot = 0.0
        for s0 in range(0, len(X), 1000):
            P = self.probs_from_scores(self.scores(X[s0:s0 + 1000]))
            tot += float(-np.log(P[np.arange(len(P)), y[s0:s0 + 1000]] + 1e-300).sum())
        return tot / len(y), len(y)

    def generate(self, prompt, length=100, temperature=1.0, top_k=0, forbid="", seed=0):
        """temperature=0 で最頻、>0 で確率 p^(1/T) からサンプリング。top_k>0 なら上位 k 個に限る。"""
        rng = np.random.default_rng(seed)
        ctx = list(prompt)
        for _ in range(length):
            p = self.next_proba("".join(ctx), forbid)
            if temperature <= 0:
                j = int(np.argmax(p))
            else:
                q = p ** (1.0 / temperature)
                if top_k and top_k < len(q):
                    q[np.argsort(-q)[top_k:]] = 0.0
                j = int(rng.choice(len(q), p=q / q.sum()))
            ctx.append(self.chars[j])
        return "".join(ctx)

    # ------------------------------------------------------------ サイズ
    def infer_bytes(self, w_bits=16, w_group=0):
        """推論に必要な重みのバイト数の見積もり (W と語彙)。光学媒質は seed から再現できるので 0 とする。"""
        V = len(self.chars)
        if w_bits < 16 and w_group:
            return int(math.ceil(self.D * V * w_bits / 8)) + 2 * V * -(-self.D // w_group) + 2 * V
        return int(math.ceil(self.D * V * w_bits / 8)) + (4 * V if w_bits < 16 else 0) + 2 * V

    # ------------------------------------------------------------ 融合 / 保存
    def merge(self, other):
        """別のモデルの統計量を足す (再学習なし)。cfg が同じモデル同士だけ。"""
        if self.cfg != other.cfg:
            raise ValueError("物理設定(cfg)が違うモデルは融合できません")
        self._add_chars(other.chars)
        A, Bt = self._np(other.A), self._np(other.Bt)
        self._merge_stats(A, Bt, other)
        self.n += other.n
        return self

    def _np(self, x):
        return x

    def _merge_stats(self, A, Bt, other):
        for j, c in enumerate(other.chars):
            self.Bt[self.idx[c]] += Bt[j]
            self.count[self.idx[c]] += other.count[j]
        self.A += A

    def _W_np(self):
        return None if self.W is None else np.asarray(self.W)

    def save(self, path, stats=True, w_bits=32, stats_dtype=np.float32, with_stats=None, w_group=0):
        """stats=True: 追加学習用に A, Bt も保存 (大きい)。False: 推論専用。w_bits: 32/16/8/6/4 (W の精度)。
        w_bits<32 は量子化後の値で保存され、読み込むと量子化された W で動く (stats=True と併用すると W は再計算で戻せる)。
        w_group>0 (w_bits<16 のとき): グループ量子化 (quantize_matrix を参照。4bit なら w_group=256 を推奨)。"""
        stats = stats if with_stats is None else with_stats
        W = self._W_np()
        if W is None:
            raise RuntimeError("solve() してから保存してください")
        data = dict(cfg=json.dumps(self.cfg), chars=np.array(self.chars), count=self.count, lam=-1.0 if self.lam is None else self.lam,
                    cal=json.dumps(self.cal), calibrated=int(self.calibrated), w_bits=int(w_bits))
        if w_bits >= 32:
            data["W"] = W.astype(np.float32)
        elif w_bits == 16:
            data["W"] = W.astype(np.float16)
        elif w_group:
            data["Wq"], data["Wgscale"] = _group_codes(np.asarray(W, dtype=np.float64), w_bits, w_group)
            data["w_group"] = int(w_group)
        else:
            qmax = 2 ** (w_bits - 1) - 1
            sc = np.abs(W).max(0) / qmax
            sc[sc == 0] = 1.0
            data["Wq"] = np.clip(np.rint(W / sc), -qmax, qmax).astype(np.int8)
            data["Wscale"] = sc.astype(np.float32)
        if stats:
            data.update(A=self._np(self.A).astype(stats_dtype), Bt=self._np(self.Bt).astype(stats_dtype), n=self.n)
        np.savez_compressed(path, **data)

    def _load_state(self, d):
        self._add_chars([str(c) for c in d["chars"]])
        self.count = d["count"].astype(np.int64)
        if "Wgscale" in d.files:
            self.W = _group_decode(d["Wq"], d["Wgscale"], int(d["w_group"]))
        elif "Wq" in d.files:
            self.W = d["Wq"].astype(np.float64) * d["Wscale"].astype(np.float64)[None, :]
        else:
            self.W = d["W"].astype(np.float64)
        if "cal" in d.files:
            self.cal = json.loads(str(d["cal"]))
            self.calibrated = bool(d["calibrated"])
        elif "tau" in d.files:  # 08 の保存形式
            self.cal = dict(kind="softmax", a=0.7, tau=float(d["tau"]), nll=None)
        if "lam" in d.files and float(d["lam"]) >= 0:
            self.lam = float(d["lam"])
        if "A" in d.files:
            self._set_stats(d["A"].astype(np.float64), d["Bt"].astype(np.float64))
            self.n = float(d["n"])

    def _set_stats(self, A, Bt):
        self.A, self.Bt = A, Bt

    @classmethod
    def load(cls, path, backend="numpy", **kw):
        d = np.load(path, allow_pickle=False)
        m = OpticalLM(**json.loads(str(d["cfg"])), backend=backend, **kw)
        m._load_state(d)
        return m

    def __repr__(self):
        return (f"<{type(self).__name__} 露光{self.E}x{self.G}x{self.G}画素 D={self.D} 窓{self.T}文字 "
                f"語彙{len(self.chars)} 学習窓{self.n:.0f}>")


# ====================================================================== GPU 版
class TorchOpticalLM(OpticalLM):
    """GPU(torch)版。光学の計算・統計量の積算・解く処理を GPU で行う。API と保存形式は numpy 版と同じ。
    違い: 特徴は float32/complex64。acc="fp32"(既定) は 1 バッチ分の ΦᵀΦ を float32 で計算して float64 の A に足す
    (消費者向け GPU では float64 の行列積が非常に遅いので数十倍速い)。acc="fp64" は従来どおり全部 float64。
    統計量 A, B と解 W は float64 のまま持つ。"""
    backend = "torch"

    def __init__(self, *a, device=None, acc="fp32", **kw):
        if torch is None:
            raise RuntimeError("torch が見つかりません。 pip install torch で入れるか、backend='numpy' で実行してください。")
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.acc = acc
        self.Wt = None
        super().__init__(*a, **kw)

    def _tt(self, x):
        return torch.as_tensor(np.asarray(x).astype(np.complex64), device=self.dev)

    def _setup_backend(self):
        self.p_pos = [self._tt(p) for p in self.pos]
        self.p_mask = [self._tt(m) for m in self.mask]
        self.p_H = self._tt(self.H)
        self.p_table = self._tt(self.table)
        self.fft2, self.ifft2 = torch.fft.fft2, torch.fft.ifft2

    def _sync_table(self):
        self.p_table = self._tt(self.table)

    def _empty(self, shape):
        return torch.empty(shape, dtype=torch.float32, device=self.dev)

    def _to_idx(self, rows):
        return torch.as_tensor(np.asarray(rows), dtype=torch.long, device=self.dev)

    def _poisson(self, x):
        return torch.poisson(x)

    def _randn(self, x):
        return torch.randn_like(x)

    def _clip(self, x, lo, hi):
        return x.clamp(lo, hi)

    def _round(self, x):
        return torch.round(x)

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

    def _np(self, x):
        return x.cpu().numpy() if hasattr(x, "cpu") else x

    def _W_np(self):
        return self.W

    def _set_stats(self, A, Bt):
        self.A = torch.as_tensor(A, device=self.dev)
        self.Bt = torch.as_tensor(Bt, device=self.dev)

    def _default_batch(self):
        return 2048 if self.acc == "fp32" else 1024

    def features_t(self, rows):
        """GPU 上の float32 テンソル (n, D)。"""
        return self._optics(self._to_idx(rows))

    def features(self, rows, batch=512):
        rows = np.asarray(rows)
        out = [self.features_t(rows[s0:s0 + batch]).double().cpu().numpy() for s0 in range(0, len(rows), batch)]
        return np.concatenate(out) if out else np.zeros((0, self.D))

    def _features_batch(self, X):
        return self._optics(self._to_idx(X))

    def _accumulate(self, Phi, y, weight):
        yb = torch.as_tensor(np.asarray(y), dtype=torch.long, device=self.dev)
        if self.acc == "fp32":
            self.A.add_(torch.mm(Phi.T, Phi), alpha=weight)
            Y = torch.zeros((len(y), self.Bt.shape[0]), dtype=torch.float32, device=self.dev)
            Y[torch.arange(len(y), device=self.dev), yb] = 1.0
            self.Bt.add_(torch.mm(Y.T, Phi), alpha=weight)
        else:
            Pd = Phi.double()
            self.A.addmm_(Pd.T, Pd, alpha=weight)
            self.Bt.index_add_(0, yb, Pd, alpha=weight)

    def solve(self, lam_rel=1e-3):
        """コレスキー分解で解く。メモリ節約のため A の対角を一時的に書き換えて戻す。
        GPU の空きメモリが分解 (A と同じ大きさの L) に足りないときは、CPU (float64) で解いて W だけ GPU に戻す。"""
        scale = float(torch.trace(self.A).item()) / self.D
        if self.dev.type == "cuda":
            free = torch.cuda.mem_get_info(self.dev)[0]
            if free < 1.15 * self.D * self.D * 8:
                M = self.A.cpu()
                reg = torch.full((self.D,), lam_rel * scale, dtype=torch.float64)
                reg[-1] = 1e-8 * scale
                M.diagonal().add_(reg)
                L = torch.linalg.cholesky(M)
                del M
                self.Wt = torch.cholesky_solve(self.Bt.T.contiguous().cpu(), L).to(self.dev)
                del L
                self.lam = lam_rel
                return self.Wt
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
        self.lam = lam_rel
        return self.Wt

    def _scores_batch(self, rows):
        Phi = self._features_batch(rows)
        if self.acc == "fp32":
            return (Phi @ self.Wt.float()).double().cpu().numpy()
        return (Phi.double() @ self.Wt).cpu().numpy()

    def _merge_stats(self, A, Bt, other):
        for j, c in enumerate(other.chars):
            self.Bt[self.idx[c]] += torch.as_tensor(Bt[j], device=self.dev)
            self.count[self.idx[c]] += other.count[j]
        self.A += torch.as_tensor(A, device=self.dev)

    @classmethod
    def load(cls, path, **kw):
        return OpticalLM.load(path, backend="torch", **kw)


def load(path, backend="numpy", **kw):
    """保存したモデルを読み込む。backend="auto" で GPU があれば GPU。"""
    return OpticalLM.load(path, backend=backend, **kw)


# ====================================================================== CPU の並列学習 (足し算なので分割して足せば厳密に同じ)
_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _low_priority():
    """この (子) プロセスの優先度を下げる。PC の操作や GPU の処理を邪魔しないため。失敗しても続ける。"""
    try:
        if sys.platform == "win32":
            import ctypes
            k = ctypes.windll.kernel32
            k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
        else:
            os.nice(10)
    except Exception:
        pass


def _fit_worker(args):
    cfg, texts, weight, threads, low = args
    if low:
        _low_priority()
    if torch is not None:
        torch.set_num_threads(threads)
    m = OpticalLM(backend="numpy", **cfg)
    m.fit(texts, weight)
    return dict(chars=m.chars, count=m.count, A=m.A, Bt=m.Bt, n=m.n)


def fit_parallel(model, texts, workers=2, threads=2, weight=1.0, low_priority=True, chunks_per_worker=2, verbose=False):
    """CPU の複数プロセスで文書を分けて学習し、model に足す (結果は model.fit(texts) と丸め誤差の範囲で同じ)。
    workers × threads が使うコア数の目安 (既定 2×2=4 コア)。low_priority=True で子プロセスの優先度を下げる。
    メモリ: 子プロセスごとに A (D×D×8 バイト) を持つ。D=4096 で 134MB、D=16384 で 2.1GB/プロセス。
    検出器の雑音がある設定では、雑音の乱数が一括学習と違うので一致は近似になる。model は numpy 版でも GPU 版でもよい
    (GPU 版でも学習は CPU で行い、統計量だけ GPU に足す)。返り値: 使った窓の数。"""
    import multiprocessing as mp
    if isinstance(texts, str):
        texts = [texts]
    texts = [t for t in texts if t]
    if not texts:
        return 0
    n_chunks = max(1, min(len(texts), workers * chunks_per_worker))
    tot = sum(map(len, texts))
    chunks, cur, acc = [], [], 0
    for t in texts:  # 文字数がほぼ均等になるように連続した文書で分ける
        cur.append(t)
        acc += len(t)
        if acc >= tot * (len(chunks) + 1) / n_chunks and len(chunks) < n_chunks - 1:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    saved = {k: os.environ.get(k) for k in _THREAD_VARS}
    for k in _THREAD_VARS:  # 子プロセスは numpy を読み込む前にこの値を見る
        os.environ[k] = str(threads)
    total = 0
    t0 = time.time()
    try:
        with mp.get_context("spawn").Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_fit_worker, [(model.cfg, c, weight, threads, low_priority) for c in chunks])):
                o = OpticalLM(backend="numpy", **model.cfg)
                o._add_chars(r["chars"])
                o.count, o.A, o.Bt, o.n = r["count"], r["A"], r["Bt"], r["n"]
                model.merge(o)
                total += r["n"]
                if verbose:
                    print(f"  並列学習 {i + 1}/{len(chunks)} 完了 ({time.time() - t0:.0f}s)", flush=True)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return total


# ====================================================================== 評価 / 較正
def _as_text(val):
    return SEAM.join(val) if not isinstance(val, str) else val


def eval_windows(model, val, limit=20000):
    """検証用の窓: 文脈の全文字と正解が学習語彙にある窓だけ (継ぎ目 \\x00 は未知文字として除かれる)。"""
    text = _as_text(val)
    if limit:
        text = text[:limit]
    ids = np.array([model.idx.get(c, -1) for c in text], dtype=np.int64)
    X, y = model._windows(ids + 1)
    y = y - 1
    keep = (y >= 0) & (X > 0).all(1)
    return X[keep], y[keep]


_TAUS = np.logspace(-2.5, -0.5, 9)
_MIX = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99)  # 1.0 は除く (頻度の床を残し、極端な NLL を防ぐ)
_GAMMAS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
_EPS = (1e-5, 1e-4, 1e-3)
_BP_GAMMAS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
_BP_BETAS = (-0.6, -0.4, -0.2, 0.0, 0.2)
_BP_MIX = (0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99)
_BP_BINS = 8


def _binpow_choose(qy, z, uy, sel):
    """binpow の1ビン分: qy, z (n, E, G, B) から nll 最小の (e, g, b, a) と nll を返す。"""
    q = qy[sel] / z[sel]
    u = uy[sel][:, None, None, None]
    best = (float("inf"), None)
    for a in _BP_MIX:
        v = -np.log(a * q + (1 - a) * u + 1e-300).mean(0)
        i = np.unravel_index(int(np.argmin(v)), v.shape)
        if v[i] < best[0]:
            best = (float(v[i]), (*i, a))
    return best


def calibrate(model, val, limit=20000, block=2000, kinds=("softmax", "power", "binpow"), verbose=False):
    """スコアを確率にする較正器を、検証データで選んで model.cal に設定する。
      softmax : p = a*softmax(S/tau) + (1-a)*頻度
      power   : p = a*clip(S,eps)^gamma/正規化 + (1-a)*頻度   (スコアは確率のスケールなので素直な写像)
      binpow  : 最大スコア (確信度) で窓を分位ビンに分け、ビンごとに
                p = a*norm(clip(S,eps)^gamma * 頻度^beta) + (1-a)*頻度   (パラメータが多いので、nll は2分割の交差当てはめで
                見積もって他と比べる。選ばれたら全窓で当てはめ直す)
    順位は変わらない(精度はそのまま)。返り値: 選ばれた較正器の dict。"""
    X, y = eval_windows(model, val, limit)
    n = len(y)
    if n == 0:
        raise ValueError("較正に使える窓がありません (検証データが短すぎる、または語彙が合わない)")
    V = len(model.chars)
    uy = model.unigram()[y]
    sm_y, sm_z = np.empty((n, len(_TAUS))), np.empty((n, len(_TAUS)))
    pw_y, pw_z = np.empty((n, len(_EPS), len(_GAMMAS))), np.empty((n, len(_EPS), len(_GAMMAS)))
    do_bp = "binpow" in kinds and n >= 50 * _BP_BINS
    if do_bp:
        bshape = (n, len(_EPS), len(_BP_GAMMAS), len(_BP_BETAS))
        bp_y, bp_z, smax = np.empty(bshape), np.empty(bshape), np.empty(n)
        ub = model.unigram()[None, :, None] ** np.array(_BP_BETAS)[None, None, :]  # (1, V, B)
    for s0 in range(0, n, block):
        S = model.scores(X[s0:s0 + block])
        yb = y[s0:s0 + block]
        b = len(yb)
        ok = np.isfinite(S)
        Sf = np.where(ok, S, -1e3)
        Sy = Sf[np.arange(b), yb]
        for t_i, t in enumerate(_TAUS):
            Z = Sf / t
            mx = Z.max(1, keepdims=True)
            sm_z[s0:s0 + b, t_i] = mx[:, 0] + np.log(np.exp(Z - mx).sum(1))
            sm_y[s0:s0 + b, t_i] = Sy / t
        for e_i, eps in enumerate(_EPS):
            Q = np.where(ok, np.maximum(S, eps), 0.0)
            qy = Q[np.arange(b), yb]
            for g_i, g in enumerate(_GAMMAS):
                pw_z[s0:s0 + b, e_i, g_i] = (Q ** g).sum(1)
                pw_y[s0:s0 + b, e_i, g_i] = qy ** g
            if do_bp:
                for g_i, g in enumerate(_BP_GAMMAS):
                    Qg = Q ** g
                    bp_z[s0:s0 + b, e_i, g_i] = Qg @ ub[0]
                    bp_y[s0:s0 + b, e_i, g_i] = (qy ** g)[:, None] * ub[0, yb]
        if do_bp:
            smax[s0:s0 + b] = Sf.max(1)
    res = {}
    if "softmax" in kinds:
        best = (float("inf"), None)
        for t_i, t in enumerate(_TAUS):
            q = np.exp(sm_y[:, t_i] - sm_z[:, t_i])
            for a in _MIX:
                v = float(-np.log(a * q + (1 - a) * uy + 1e-300).mean())
                if v < best[0]:
                    best = (v, dict(kind="softmax", a=a, tau=float(t), nll=v))
        res["softmax"] = best[1]
    if "power" in kinds:
        best = (float("inf"), None)
        for e_i, eps in enumerate(_EPS):
            for g_i, g in enumerate(_GAMMAS):
                q = pw_y[:, e_i, g_i] / pw_z[:, e_i, g_i]
                for a in _MIX:
                    v = float(-np.log(a * q + (1 - a) * uy + 1e-300).mean())
                    if v < best[0]:
                        best = (v, dict(kind="power", a=a, gamma=g, eps=eps, nll=v))
        res["power"] = best[1]
    if do_bp:
        edges = np.quantile(smax, np.linspace(0, 1, _BP_BINS + 1)[1:-1])
        bins = np.searchsorted(edges, smax)
        fold = np.arange(n) % 2
        tot = 0.0
        for f in (0, 1):  # 交差当てはめ: 片方で選び、もう片方で評価
            for k in range(_BP_BINS):
                fit_s, ev_s = (bins == k) & (fold == f), (bins == k) & (fold != f)
                if not ev_s.any():
                    continue
                _, (e_i, g_i, b_i, a) = _binpow_choose(bp_y, bp_z, uy, fit_s if fit_s.any() else (bins == k))
                q = bp_y[ev_s, e_i, g_i, b_i] / bp_z[ev_s, e_i, g_i, b_i]
                tot += float(-np.log(a * q + (1 - a) * uy[ev_s] + 1e-300).sum())
        params = []
        glob = _binpow_choose(bp_y, bp_z, uy, np.ones(n, dtype=bool))[1]
        for k in range(_BP_BINS):  # 同じスコアが多いと分位の境界が重なり、空のビンができる -> 全体で選んだ値を使う
            e_i, g_i, b_i, a = _binpow_choose(bp_y, bp_z, uy, bins == k)[1] if (bins == k).any() else glob
            params.append([_EPS[e_i], _BP_GAMMAS[g_i], _BP_BETAS[b_i], a])
        res["binpow"] = dict(kind="binpow", edges=[float(x) for x in edges], params=params, nll=tot / n)
    chosen = min(res.values(), key=lambda c: c["nll"])
    model.cal = chosen
    model.calibrated = True
    model.cal_table = res
    if verbose:
        for k, c in res.items():
            print(f"    較正 {k:<8} nll {c['nll']:.3f}  {c}")
    return chosen


def evaluate(model, val, limit=20000, block=2000, with_py=False):
    """検証。返り値 dict: acc1, acc5 (スコア順位), nll (較正後の確率), n, [py: 正解の確率の配列]。"""
    X, y = eval_windows(model, val, limit)
    n = len(y)
    if n == 0:
        return dict(acc1=float("nan"), acc5=float("nan"), nll=float("nan"), n=0)
    c1 = c5 = 0
    py = np.empty(n)
    for s0 in range(0, n, block):
        S = model.scores(X[s0:s0 + block])
        yb = y[s0:s0 + block]
        V = S.shape[1]
        k = min(5, V)
        c1 += int((S.argmax(1) == yb).sum())
        top = np.argpartition(S, V - k, axis=1)[:, V - k:]
        c5 += int((top == yb[:, None]).any(1).sum())
        P = model.probs_from_scores(S)
        py[s0:s0 + len(yb)] = P[np.arange(len(yb)), yb]
    r = dict(acc1=100.0 * c1 / n, acc5=100.0 * c5 / n, nll=float(-np.log(py + 1e-300).mean()), n=n)
    if with_py:
        r["py"] = py
    return r


def sweep_lam(model, val, lams=(1e-4, 1e-3, 1e-2, 1e-1), limit=20000, verbose=True):
    """正則化の強さを、較正後の NLL で選ぶ (素の softmax の NLL は壊れやすいので基準にしない)。選んだ値で解き直し、較正も済ませる。"""
    best = None
    for lam in lams:
        model.solve(lam)
        cal = calibrate(model, val, limit)
        r = evaluate(model, val, limit)
        if verbose:
            print(f"  lam_rel={lam:<7g} val acc1 {r['acc1']:5.1f} acc5 {r['acc5']:5.1f} nll {r['nll']:.3f}  ({cal['kind']})")
        if best is None or r["nll"] < best["nll"]:
            best = dict(lam=lam, cal=cal, **r)
    model.solve(best["lam"])
    model.cal = best["cal"]
    model.calibrated = True
    return best


# ====================================================================== データ
def read_docs(source, max_chars=0):
    """source: ファイルパス / 文字列 / 文書のリスト -> 文書のリスト。ファイルは 1行=1文書 (空行は除く)。"""
    if isinstance(source, (list, tuple)):
        docs = [d for d in source if d]
    elif isinstance(source, str) and os.path.exists(source):
        with open(source, "r", encoding="utf-8-sig") as f:
            docs = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
    else:
        if isinstance(source, str) and len(source) < 260 and source.lower().endswith((".txt", ".text", ".md")):
            raise FileNotFoundError(f"ファイルが見つかりません: {source}")
        docs = [str(source)]
    if max_chars:
        out, tot = [], 0
        for d in docs:
            out.append(d)
            tot += len(d)
            if tot >= max_chars:
                break
        docs = out
    return docs


def split_docs(docs, every=7, block=500):
    """検証の分け方。文書が十分あれば文書単位 (7文書に1つを検証)、少なければ 500文字ブロック単位で散らす。"""
    if len(docs) >= 3 * every:
        return ([d for i, d in enumerate(docs) if i % every != every // 2],
                [d for i, d in enumerate(docs) if i % every == every // 2])
    text = "".join(docs)
    blocks = [text[i:i + block] for i in range(0, len(text), block)]
    return ([b for i, b in enumerate(blocks) if i % every != every // 2],
            [b for i, b in enumerate(blocks) if i % every == every // 2])


def train(source, val_every=7, refit_all=False, lams=(1e-4, 1e-3, 1e-2, 1e-1), max_chars=0, chunks=1, verbose=True,
          backend="auto", workers=0, threads=2, **cfg):
    """ファイル(1行=1文書)から、分割・追加学習・正則化の選択・較正まで行う。cfg は OpticalLM の引数 (grid, scales, ...)。
    refit_all=True: 選んだ設定で検証用の文書も学習に足して解き直す (データが約14%増える。較正はそのまま。
                    ただしこのモデルに対する検証値は、検証文書を学習済みなので甘くなる。最終版を作るときだけ使う)。
    返り値: (model, report)。report は検証での acc1/acc5/nll など。"""
    docs = read_docs(source, max_chars)
    tr, va = split_docs(docs, val_every)
    model = OpticalLM(backend=backend, **cfg)
    if verbose:
        print(f"{model} | 学習文書 {len(tr)} / 検証文書 {len(va)} | 文字数 {sum(map(len, docs))}")
    step = max(1, len(tr) // chunks)
    for i in range(chunks):
        t0 = time.time()
        part = tr[i * step:(i + 1) * step if i < chunks - 1 else len(tr)]
        if workers > 1:
            fit_parallel(model, part, workers=workers, threads=threads, verbose=verbose)
        else:
            model.fit(part)
        if verbose:
            print(f"  学習 {i + 1}/{chunks}: {time.time() - t0:.1f}s")
    best = sweep_lam(model, va, lams, verbose=verbose)
    report = dict(lam=best["lam"], acc1=best["acc1"], acc5=best["acc5"], nll=best["nll"], cal=model.cal, n_val=best["n"],
                  D=model.D, vocab=len(model.chars))
    if refit_all and va:
        model.fit(va)
        model.solve(best["lam"])
    return model, report


# ====================================================================== 重みの圧縮 (小さくする / 実機の精度制限の模擬)
def _group_codes(W, bits, group):
    """グループ量子化の符号。各列を group 行ずつに分け、グループごとのスケール (fp16) で対称量子化。
    返り値 (codes int8 (D,V), scales float16 (グループ数, V))。"""
    qmax = 2 ** (bits - 1) - 1
    D, V = W.shape
    ng = -(-D // group)
    Wp = np.vstack([W, np.zeros((ng * group - D, V))]).reshape(ng, group, V)
    sc = (np.abs(Wp).max(1) / qmax).astype(np.float16)
    sc[sc == 0] = 1.0
    q = np.clip(np.rint(Wp / sc.astype(np.float64)[:, None, :]), -qmax, qmax).reshape(-1, V)[:D]
    return q.astype(np.int8), sc


def _group_decode(codes, sc, group):
    rep = np.repeat(sc.astype(np.float64), group, axis=0)[:codes.shape[0]]
    return codes.astype(np.float64) * rep


def quantize_matrix(W, bits, group=0):
    """W (D,V) をクラス (列) ごとの対称量子化。返り値 (量子化後の W, 必要バイト数)。
    group>0: 各列を group 行ずつのグループに分け、グループごとにスケール (fp16) を持つ。列ごとのスケールは
    バイアス行や一部の大きな値で決まってしまい 4bit 以下で壊れるが、group=256 なら 4bit でもほぼ劣化しない (RESULTS.md キュー6)。"""
    W = np.asarray(W, dtype=np.float64)
    if bits >= 32:
        return W.copy(), W.size * 4
    if bits == 16:
        return W.astype(np.float16).astype(np.float64), W.size * 2
    if group:
        codes, sc = _group_codes(W, bits, group)
        return _group_decode(codes, sc, group), int(math.ceil(W.size * bits / 8)) + 2 * sc.size
    qmax = 2 ** (bits - 1) - 1
    sc = np.abs(W).max(0) / qmax
    sc[sc == 0] = 1.0
    q = np.clip(np.rint(W / sc), -qmax, qmax)
    return q * sc, int(math.ceil(W.size * bits / 8)) + 4 * W.shape[1]


def lowrank_matrix(W, rank, bits=16):
    """W ≈ P Q (ランク rank)。P (D,r), Q (r,V) をそれぞれ bits で量子化。返り値 (近似 W, バイト数)。"""
    U, s, Vt = np.linalg.svd(np.asarray(W, dtype=np.float64), full_matrices=False)
    P, Q = U[:, :rank] * s[:rank], Vt[:rank]
    Pq, bp = quantize_matrix(P, bits)
    Qq, bq = quantize_matrix(Q, bits)
    return Pq @ Qq, bp + bq


# ====================================================================== n-gram (比較相手。サイズを数えられる)
class NgramLM:
    """数えるだけの n-gram。文脈は最大 4 文字。min_count>1 で『出現が少ない項目』を捨てて小さくできる
    (2文字以上の文脈にだけ適用)。足し算で追加学習できる点はリザバーと同じ。評価する窓は eval_windows と同じ。
    smoothing: "wb" = Witten-Bell 補間 (既定、従来どおり) / "kn" = 補間 Kneser-Ney (階層ごとに割引 D = n1/(n1+2n2)、
    最上位以外の階層と頻度の床は『続き方の数』(その並びの前に来た異なる文字の数) で数える。剪定に強く、普通はこちらが強い)。"""

    def __init__(self, order=3, min_count=1, smoothing="wb"):
        if not 1 <= order <= 4:
            raise ValueError("order は 1〜4")
        if smoothing not in ("wb", "kn"):
            raise ValueError("smoothing は 'wb' か 'kn'")
        self.order, self.min_count, self.smoothing = order, min_count, smoothing

    def fit(self, docs):
        docs = [d for d in docs if d]
        self.chars = sorted(set("".join(docs)))
        self.idx = {c: i for i, c in enumerate(self.chars)}
        V = self.V = len(self.chars)
        seq = np.concatenate([np.concatenate([[-1], np.array([self.idx[c] for c in d], dtype=np.int64)]) for d in docs])
        uni = np.bincount(seq[seq >= 0], minlength=V).astype(np.float64)
        self.pu = (uni + 0.5) / (uni.sum() + 0.5 * V)
        raw = []  # 階層ごとの (文脈キー, 次の文字, 回数)。文脈キーは直近の文字が下位 16bit
        for k in range(1, self.order + 1):
            pos = np.arange(k, len(seq))
            ok = np.ones(len(pos), dtype=bool)
            for j in range(k + 1):
                ok &= seq[pos - j] >= 0
            pos = pos[ok]
            key = np.zeros(len(pos), dtype=np.uint64)
            for j in range(k):
                key |= seq[pos - 1 - j].astype(np.uint64) << np.uint64(16 * j)
            w = seq[pos]
            ukeys, cid = np.unique(key, return_inverse=True)
            upk, cnt = np.unique(cid.astype(np.int64) * V + w, return_counts=True)
            raw.append((ukeys[upk // V], upk % V, cnt.astype(np.float64)))
        self.disc = [0.0] * self.order
        raw_n = list(raw)  # 剪定は生の出現回数で判定する (KN でも同じ項目を残す)
        if self.smoothing == "kn":
            # 続き方の数: 階層 k の (文脈, 文字) の組から、文脈の一番古い文字を落として数える -> 階層 k-1 の回数にする
            for k in range(self.order, 1, -1):
                ck, wk, _ = raw[k - 1]
                low = ck & np.uint64((1 << (16 * (k - 1))) - 1)
                uk, cid = np.unique(low, return_inverse=True)
                upk, cnt = np.unique(cid.astype(np.int64) * V + wk, return_counts=True)
                raw[k - 2] = (uk[upk // V], upk % V, cnt.astype(np.float64))
            cont = np.bincount(raw[0][1], minlength=V).astype(np.float64)  # 頻度の床も続き方の数 (前に来た異なる文字の数)
            self.pu = (cont + 0.5) / (cont.sum() + 0.5 * V)
            for k in range(self.order):
                c = raw[k][2]
                n1, n2 = float((c == 1).sum()), float((c == 2).sum())
                self.disc[k] = min(0.95, max(0.1, n1 / (n1 + 2 * n2))) if n1 + n2 > 0 else 0.5
        self.levels = []
        for k in range(1, self.order + 1):
            ctx, w, cnt = raw[k - 1]
            if k >= 2 and self.min_count > 1:
                rc, rw, rn = raw_n[k - 1]
                if rc is ctx:
                    keep = rn >= self.min_count
                else:  # KN の下位階層: 同じ (文脈, 文字) の生の回数を引いて判定
                    uk, inv = np.unique(np.concatenate([rc, ctx]), return_inverse=True)
                    a_ = inv[:len(rc)].astype(np.int64) * V + rw
                    b_ = inv[len(rc):].astype(np.int64) * V + w
                    o_ = np.argsort(a_)
                    j = np.minimum(np.searchsorted(a_[o_], b_), len(a_) - 1)
                    keep = (a_[o_][j] == b_) & (rn[o_][j] >= self.min_count)
                ctx, w, cnt = ctx[keep], w[keep], cnt[keep]
            order_ = np.lexsort((w, ctx))
            ctx, w, cnt = ctx[order_], w[order_], cnt[order_]
            ukeys, newc = np.unique(ctx, return_inverse=True)
            tot = np.bincount(newc, weights=cnt)
            Tn = np.bincount(newc)
            self.levels.append(dict(keys=ukeys, ptr=np.concatenate([[0], np.cumsum(Tn)]), w=w,
                                    c=cnt, tot=tot, T=Tn.astype(np.float64), d=self.disc[k - 1]))
        return self

    def entries(self):
        return int(sum(len(lv["w"]) for lv in self.levels))

    def bytes(self, compact=False):
        """推定サイズ。単純な表: 1項目 8 バイト (4Bのハッシュ+2Bの次文字+2Bのカウント)。
        工夫した形式 (文脈ごとにまとめる等): 1項目 4 バイト (2B次文字+1Bカウント+1B文脈ポインタ)。語彙は 2B/文字。"""
        return self.entries() * (4 if compact else 8) + 2 * self.V

    def _dense(self, Xb):
        b, T = Xb.shape
        V = self.V
        P = np.tile(self.pu, (b, 1))
        alive = np.ones(b, dtype=bool)
        for k, lv in enumerate(self.levels, start=1):
            key = np.zeros(b, dtype=np.uint64)
            for j in range(k):
                key |= Xb[:, T - 1 - j].astype(np.uint64) << np.uint64(16 * j)
            if len(lv["keys"]) == 0:  # 剪定で全部消えた階層
                break
            pos = np.minimum(np.searchsorted(lv["keys"], key), len(lv["keys"]) - 1)
            found = alive & (lv["keys"][pos] == key)
            alive = found
            if not found.any():
                break
            rows = np.nonzero(found)[0]
            cid = pos[rows]
            starts = lv["ptr"][cid]
            lens = lv["ptr"][cid + 1] - starts
            off = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens) + np.repeat(starts, lens)
            C = np.zeros((b, V))
            C[np.repeat(rows, lens), lv["w"][off]] = lv["c"][off]
            Tt, tt = lv["T"][cid][:, None], lv["tot"][cid][:, None]
            if self.smoothing == "kn":
                d = lv["d"]
                P[rows] = (np.maximum(C[rows] - d, 0.0) + d * Tt * P[rows]) / tt
            else:
                P[rows] = (C[rows] + Tt * P[rows]) / (tt + Tt)
        return P

    def eval(self, val, T, limit=20000, block=2000, with_py=False):
        """リザバーの evaluate と同じ窓・同じ指標。T はリザバーの窓長 (窓の集合を揃えるため)。"""
        text = _as_text(val)
        if limit:
            text = text[:limit]
        ids = np.array([self.idx.get(c, -1) for c in text], dtype=np.int64)
        n = len(ids) - T
        if n <= 0:
            return dict(acc1=float("nan"), acc5=float("nan"), nll=float("nan"), n=0)
        X = ids[np.arange(n)[:, None] + np.arange(T)[None, :]]
        y = ids[T:]
        keep = (y >= 0) & (X >= 0).all(1)
        X, y = X[keep], y[keep]
        n = len(y)
        c1 = c5 = 0
        py = np.empty(n)
        for s0 in range(0, n, block):
            P = self._dense(X[s0:s0 + block])
            yb = y[s0:s0 + block]
            c1 += int((P.argmax(1) == yb).sum())
            k = min(5, self.V)
            top = np.argpartition(P, self.V - k, axis=1)[:, self.V - k:]
            c5 += int((top == yb[:, None]).any(1).sum())
            py[s0:s0 + len(yb)] = P[np.arange(len(yb)), yb]
        r = dict(acc1=100.0 * c1 / n, acc5=100.0 * c5 / n, nll=float(-np.log(py).mean()), n=n)
        if with_py:
            r["py"] = py
        return r


def mixture_nll(py_a, py_b):
    """2つのモデルの確率を重み λ で混ぜたときの最良 NLL。返り値 (nll, λ)。λ は a の重み。"""
    best = (float("inf"), None)
    for lam in np.linspace(0.0, 1.0, 21):
        v = float(-np.log(lam * py_a + (1 - lam) * py_b + 1e-300).mean())
        if v < best[0]:
            best = (v, float(lam))
    return best


# ====================================================================== 同じバイト数での比較 (bench)
def _fmt_bytes(b):
    return f"{b / 1e6:.2f}MB" if b >= 1e6 else f"{b / 1e3:.0f}KB"


def size_bench(source, d_list=(1024, 2048, 4096), scales=(1, 2, 3, 4), backend="auto", max_chars=0, chunks=1,
               w_bits=(16, 8, 6, 4), ranks=(64, 128, 256), lams=(1e-3, 1e-2), ng_orders=(2, 3, 4),
               ng_min=(1, 2, 3, 5, 10), limit=20000, mix=True, w_group_bits=(5, 4, 3), ng_smoothing="kn", **cfg):
    """リザバー (D・W の精度・低ランク) と n-gram (文脈長・剪定) を、推論に必要なバイト数 vs 精度で並べる。
    n-gram のバイト数は『単純な表(8B/項目)』と『工夫した形式(4B/項目)』の2通りの見積もり。"""
    docs = read_docs(source, max_chars)
    tr, va = split_docs(docs)
    T = max(scales)
    rows = []
    print(f"文書 {len(docs)} (学習 {len(tr)} / 検証 {len(va)}) | 文字数 {sum(map(len, docs))} | 窓 {T}文字")

    ng_main = None
    for o in ng_orders:
        if o > T:
            continue
        for mc in ng_min:
            if mc > 1 and o == 2 and False:
                continue
            ng = NgramLM(o, mc, smoothing=ng_smoothing).fit(tr)
            r = ng.eval(va, T, limit, with_py=True)
            rows.append(dict(name=f"n-gram 文脈{o}文字 剪定{mc}", b8=ng.bytes(False), b4=ng.bytes(True), kind="ngram", **r))
            if o == min(3, T) and mc == 1:
                ng_main = r["py"]
    base_py = None
    for D in d_list:
        g = grid_for_D(D, len(scales))
        t0 = time.time()
        m = OpticalLM(grid=g, scales=scales, backend=backend, **cfg)
        step = max(1, len(tr) // chunks)
        for i in range(chunks):
            m.fit(tr[i * step:(i + 1) * step if i < chunks - 1 else len(tr)])
        best = sweep_lam(m, va, lams, limit, verbose=False)
        W0 = m.W.copy()
        print(f"  D={m.D} 学習+選択 {time.time() - t0:.0f}s: acc1 {best['acc1']:.1f} acc5 {best['acc5']:.1f} nll {best['nll']:.3f} (lam={best['lam']:g})")
        variants = [("W 32bit", W0, m.D * len(m.chars) * 4)]
        for wb in w_bits:
            Wq, nb = quantize_matrix(W0, wb)
            variants.append((f"W {wb}bit", Wq, nb))
        for wb in w_group_bits:
            Wq, nb = quantize_matrix(W0, wb, 256)
            variants.append((f"W {wb}bit g256", Wq, nb))
        for r_ in ranks:
            if r_ < min(W0.shape):
                Wl, nb = lowrank_matrix(W0, r_, 8)
                variants.append((f"W ランク{r_}(8bit)", Wl, nb))
        for tag, Wv, nb in variants:
            m.W = Wv
            calibrate(m, va, limit)
            r = evaluate(m, va, limit, with_py=True)
            nb += 2 * len(m.chars)
            rows.append(dict(name=f"リザバー D={m.D} {tag}", b8=nb, b4=nb, kind="res", **r))
            if tag == "W 16bit" or (tag == "W 32bit" and base_py is None):
                base_py = r["py"]
        m.W = W0
        del m
    print("\n推論に必要なバイト数 (n-gram は 単純な表/工夫した形式 の2通りの見積もり) と精度。バイト数の小さい順。")
    print("  印: a=その大きさ以下で acc1 が最良 / n=その大きさ以下で nll が最良 (どの方式でも)")
    print(f"{'構成':<34}{'サイズ(8B/項目)':>16}{'サイズ(4B/項目)':>16}{'acc1':>7}{'acc5':>7}{'nll':>7}  印")
    for r in sorted(rows, key=lambda r: r["b4"]):
        smaller = [q for q in rows if q["b4"] <= r["b4"]]
        mark = ("a" if r["acc1"] >= max(q["acc1"] for q in smaller) else "") + \
               ("n" if r["nll"] <= min(q["nll"] for q in smaller) else "")
        print(f"{r['name']:<34}{_fmt_bytes(r['b8']):>16}{_fmt_bytes(r['b4']):>16}{r['acc1']:>7.1f}{r['acc5']:>7.1f}{r['nll']:>7.3f}  {mark}")
    res_rows = [r for r in rows if r["kind"] == "res"]
    ng_rows = [r for r in rows if r["kind"] == "ngram"]
    print("\nリザバーの各構成が、n-gram の構成に『サイズ・acc1・nll の全てで負けている(支配される)』か:")
    for lab, key in (("n-gram を工夫した形式 4B/項目 と見積もる(n-gram に有利)", "b4"), ("n-gram を単純な表 8B/項目 と見積もる(n-gram に不利)", "b8")):
        free = []
        for r in res_rows:
            if not any(q[key] <= r["b4"] and q["acc1"] >= r["acc1"] and q["nll"] <= r["nll"] for q in ng_rows):
                free.append(r)
        print(f"  {lab}: 支配された構成 {len(res_rows) - len(free)}/{len(res_rows)}")
        for r in free:
            print(f"     支配されない: {r['name']} ({_fmt_bytes(r['b4'])}, acc1 {r['acc1']:.1f}, nll {r['nll']:.3f})")
        if not free:
            print("     -> どのリザバー構成も、同じかより小さい n-gram のどれかに全指標で負けている")
    top = max(ng_rows, key=lambda q: q["b4"]) if ng_rows else None
    if top:
        print(f"  (注: n-gram の最大構成は {_fmt_bytes(top['b4'])}。それより大きい帯は n-gram の点が無いので、勝敗は言えません)")
    if ng_main is not None and base_py is not None and len(ng_main) == len(base_py):
        nl, lam = mixture_nll(base_py, ng_main)
        n_res, n_ng = float(-np.log(base_py).mean()), float(-np.log(ng_main).mean())
        gain = min(n_res, n_ng) - nl
        print(f"\n混合 (リザバー D最大 W16bit と n-gram文脈3): 混合 nll {nl:.3f} (リザバーの重み {lam:.2f}) | "
              f"リザバー単独 {n_res:.3f} | n-gram単独 {n_ng:.3f}")
        print("  -> " + (f"混合が良い方の単独より {gain:.3f} 小さい: リザバーは n-gram に無い情報を持っている (補い合う)" if gain > 0.02
                         else "混合しても良い方の単独からほぼ改善しない: 2つは同じ情報を見ている"))
    if mix:
        mix_compare(rows)
    return rows


def mix_compare(rows, base_name="n-gram 文脈3文字 剪定1"):
    """混合の公平な比較。(1) 基準の n-gram に『リザバー』を足す vs 『別の n-gram』を足す (追加バイト数で揃える)。
    (2) 2モデルの混合を全ペアで作り、リザバー入りの混合が、同じか小さい合計サイズの n-gram 同士の混合に nll で負けるか。
    サイズは n-gram に有利な 4B/項目 の見積もり。rows は size_bench の返り値 (py つき)。"""
    rows = [r for r in rows if "py" in r]
    base = next((r for r in rows if r["name"] == base_name), None)
    if base is None or any(len(r["py"]) != len(base["py"]) for r in rows):
        print("\n[混合の比較] 基準の n-gram が無いか、評価窓が揃っていないので省略")
        return None
    ng = [r for r in rows if r["kind"] == "ngram"]
    res = [r for r in rows if r["kind"] == "res"]
    nll1 = lambda r: float(-np.log(r["py"] + 1e-300).mean())
    print(f"\n[混合の比較1] 基準 {base_name} ({_fmt_bytes(base['b4'])}, nll {nll1(base):.3f}) に1つ足す。追加サイズ(4B/項目)の小さい順")
    print(f"{'足すもの':<34}{'追加サイズ':>11}{'混合nll':>9}{'基準からの差':>12}{'重み':>6}  印")
    add = []
    for r in ng + res:
        if r is base:
            continue
        nl, lam = mixture_nll(base["py"], r["py"])
        lp = np.log(lam * base["py"] + (1 - lam) * r["py"] + 1e-300)
        add.append(dict(name=r["name"], kind=r["kind"], b=r["b4"], nll=nl, lam=1 - lam, lp=lp))
    for a in sorted(add, key=lambda a: a["b"]):
        best = min(q["nll"] for q in add if q["b"] <= a["b"])
        print(f"{a['name']:<34}{_fmt_bytes(a['b']):>11}{a['nll']:>9.3f}{a['nll'] - nll1(base):>+12.3f}{a['lam']:>6.2f}  {'n' if a['nll'] <= best else ''}")
    print("  印 n = その追加サイズ以下で混合 nll が最良")
    for a in [a for a in add if a["kind"] == "res"]:
        rivals = [q for q in add if q["kind"] == "ngram" and q["b"] <= a["b"]]
        if rivals:
            q = min(rivals, key=lambda q: q["nll"])
            dlt = q["lp"] - a["lp"]  # 窓ごとの nll の差 (リザバー側 - n-gram側)。対比較なので標準誤差が小さい
            se = float(dlt.std() / math.sqrt(len(dlt)))
            print(f"  {a['name']:<34} {a['nll']:.3f} vs 同じか小さい n-gram の追加で最良 {q['nll']:.3f} ({q['name']}) "
                  f"差 {a['nll'] - q['nll']:+.3f}±{se:.3f} "
                  f"-> {'リザバーが良い' if a['nll'] < q['nll'] - max(0.01, 2 * se) else '差なし' if abs(a['nll'] - q['nll']) <= max(0.01, 2 * se) else 'n-gram が良い'}")
    pairs = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if a["kind"] == "res" and b["kind"] == "res":
                continue
            nl, _ = mixture_nll(a["py"], b["py"])
            pairs.append(dict(name=f"{a['name']} + {b['name']}", res=a["kind"] == "res" or b["kind"] == "res",
                              b=a["b4"] + b["b4"], nll=nl))
    nn = [p for p in pairs if not p["res"]]
    rn = [p for p in pairs if p["res"]]
    free = [p for p in rn if not any(q["b"] <= p["b"] and q["nll"] <= p["nll"] for q in nn)]
    print(f"\n[混合の比較2] 2モデル混合の全ペア: リザバー入り {len(rn)} 通り中 {len(rn) - len(free)} 通りが、"
          f"合計サイズが同じか小さい n-gram 同士の混合に nll で負ける")
    for p in sorted(free, key=lambda p: p["nll"])[:10]:
        rivals = [q for q in nn if q["b"] <= p["b"]]
        q = min(rivals, key=lambda q: q["nll"]) if rivals else None
        print(f"   負けない: {p['name']} ({_fmt_bytes(p['b'])}, nll {p['nll']:.3f})"
              + (f" | 同サイズ以下の n-gram 同士の最良 {q['nll']:.3f} ({q['name']})" if q else " | 同サイズ以下の n-gram 同士の混合なし"))
    if nn:
        q = min(nn, key=lambda q: q["nll"])
        print(f"  n-gram 同士の混合で最良: {q['name']} ({_fmt_bytes(q['b'])}, nll {q['nll']:.3f})")
    if rn:
        q = min(rn, key=lambda q: q["nll"])
        print(f"  リザバー入りの混合で最良: {q['name']} ({_fmt_bytes(q['b'])}, nll {q['nll']:.3f})")
    return dict(add=add, pairs=pairs)


# ====================================================================== 動作確認
def _ref08_features(m, X, grid, scales, layers=3, dist=5e-3, wavelength=633e-9, pitch=20e-6, seed=0):
    """08 の features() と同じ式を、このファイルの実装に依存せずに書いたもの (後方互換の照合用)。"""
    G, N = grid, grid * grid
    rg = lambda *key: np.random.default_rng([seed, *key])
    pos = [np.exp(1j * np.stack([rg(1, e, p).uniform(0, 2 * np.pi, N) for p in range(s)])) for e, s in enumerate(scales)]
    mask = [np.exp(1j * np.stack([rg(2, e, l).uniform(0, 2 * np.pi, (G, G)) for l in range(layers)])) for e in range(len(scales))]
    f = np.fft.fftfreq(G, d=pitch)
    FX, FY = np.meshgrid(f, f, indexing="ij")
    k = 2 * np.pi / wavelength
    H = np.exp(1j * np.sqrt(np.maximum(k ** 2 - (2 * np.pi * FX) ** 2 - (2 * np.pi * FY) ** 2, 0.0)) * dist)
    out = np.empty((len(X), len(scales) * N + 1))
    out[:, -1] = 1.0
    for e, s in enumerate(scales):
        U = (m.table[X[:, -s:]] * pos[e][None]).sum(1).reshape(-1, G, G)
        for l in range(layers):
            U = U * mask[e][l]
            U = np.fft.ifft2(np.fft.fft2(U) * H)
        out[:, e * N:(e + 1) * N] = ((U.real ** 2 + U.imag ** 2).reshape(len(X), -1)) / s - 1.0
    return out


def selftest(verbose=True):
    """厳密性・数値の一致・保存読み込み・(torch があれば)GPU版との照合を、小さな設定で確認する。"""
    ok_all = True

    def check(name, cond, detail=""):
        nonlocal ok_all
        ok_all &= bool(cond)
        print(f"  [{'OK' if cond else 'NG'}] {name} {detail}")

    rng = np.random.default_rng(0)
    chars = [chr(0x3041 + i) for i in range(30)]
    mk_text = lambda n: "".join(chars[i] for i in rng.integers(0, 30, n))
    docs = [mk_text(300) for _ in range(8)]
    rel = lambda a, b: float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-300))
    cfgs = {
        "基本": dict(grid=8, scales=(1, 2, 3)),
        "検出器あり": dict(grid=8, scales=(1, 2), detector=dict(adc_bits=8)),
    }
    for tag, cfg in cfgs.items():
        print(f"[{tag}] {cfg}")
        a = OpticalLM(**cfg)
        a.fit(docs)
        full = OpticalLM(**cfg)
        full.fit(docs[:5])
        part = OpticalLM(**cfg)
        part.fit(docs[:3])
        other = OpticalLM(**cfg)
        other.fit(docs[3:5])
        part.merge(other)
        check("融合 = 一括学習", rel(full.A, part.A) < 1e-12, f"(相対差 {rel(full.A, part.A):.1e})")
        if tag == "基本":
            par = OpticalLM(**cfg)
            fit_parallel(par, docs[:5], workers=2, threads=1, chunks_per_worker=2)
            ok_p = rel(full.A, par.A) < 1e-12 and rel(full.Bt[[full.idx[c] for c in full.chars]], par.Bt[[par.idx[c] for c in full.chars]]) < 1e-12                 and all(full.count[full.idx[c]] == par.count[par.idx[c]] for c in full.chars) and par.n == full.n
            check("CPU 並列学習 (2プロセス) = 一括学習", ok_p, f"(相対差 {rel(full.A, par.A):.1e})")
        b = copy.deepcopy(a)
        b.forget(docs[5:])
        check("取り消し = 取り消した分を学習していないモデル", rel(full.A, b.A) < 1e-12 and rel(full.Bt, b.Bt) < 1e-12,
              f"(相対差 {rel(full.A, b.A):.1e})")
        a.solve(1e-3)
        c = calibrate(a, docs[:2], limit=0)
        S = a.scores(a._windows(a.rows_for(docs[0]))[0][:20])
        P = a.probs_from_scores(S)
        check("確率の和が1", np.allclose(P.sum(1), 1.0), f"(較正 {c['kind']})")
        keep_cal = a.cal
        a.cal = calibrate(a, docs[:2], limit=0, kinds=("binpow",))
        Pb = a.probs_from_scores(S)
        check("確率の和が1 (binpow)", np.allclose(Pb.sum(1), 1.0) and (Pb > 0).all(),
              f"(交差 nll {a.cal['nll']:.3f}, ビン {len(a.cal['params'])})")
        W_keep = a.W
        a.W = np.zeros_like(np.asarray(W_keep))  # 全スコアが同じ (空のビンができる極端な場合)
        c0 = calibrate(a, docs[:2], limit=0, kinds=("binpow",))
        P0 = a.probs_from_scores(a.scores(a._windows(a.rows_for(docs[0]))[0][:20]))
        check("binpow: スコアが全て同じでも動く", np.allclose(P0.sum(1), 1.0), f"(交差 nll {c0['nll']:.3f})")
        a.W = W_keep
        a.cal = keep_cal
        pr = a.predict(docs[0][:3], 3)
        check("predict が確率つきで返る", len(pr) == 3 and all(0 <= p <= 1 for _, p in pr))
        g = a.generate(docs[0][:3], 20, temperature=0.7, seed=1)
        check("generate が指定長", len(g) == 23)
        path = "_selftest_model.npz"
        a.save(path, stats=True)
        r = load(path)
        check("保存→読み込みで同じ予測", np.allclose(r.scores(a._windows(a.rows_for(docs[0]))[0][:5]),
                                                  a.scores(a._windows(a.rows_for(docs[0]))[0][:5]), atol=1e-3, equal_nan=True))
        for wb in (16, 8, 4):
            a.save(path, stats=False, w_bits=wb)
            r = load(path)
            err = rel(a.W, r.W)
            check(f"W {wb}bit の保存読み込み", err < {16: 1e-2, 8: 0.2, 4: 1.0}[wb], f"(相対誤差 {err:.2e})")
        a.save(path, stats=False, w_bits=4, w_group=16)
        r = load(path)
        Wg, _ = quantize_matrix(a.W, 4, 16)
        check("W 4bit グループ量子化の保存読み込み", np.allclose(r.W, Wg) and rel(a.W, r.W) < 0.3,
              f"(相対誤差 {rel(a.W, r.W):.2e})")
        os.remove(path)

    print("[後方互換] 08 (sin-cnn_reservoir08.py) の式を独立に書き直したものと同じ特徴か")
    text = mk_text(400)
    grid, sc = 8, (1, 2, 4, 8)
    m = OpticalLM(grid=grid, scales=sc)
    m.fit(text, pad_start=False)
    X, _ = m._windows(m.rows_for(text[:150]))
    F_new = m.features(X)
    F_ref = _ref08_features(m, X, grid, sc)
    check("特徴 (08 の式)", rel(F_ref, F_new) < 1e-9, f"(相対差 {rel(F_ref, F_new):.1e})")

    print("[n-gram] 窓の集合がリザバーの評価と一致し、確率が正規化されているか")
    tr, va = docs[:6], docs[6:]
    ng = NgramLM(3).fit(tr)
    m = OpticalLM(grid=8, scales=(1, 2, 3, 4))
    m.fit(tr)
    m.solve(1e-3)
    X, y = eval_windows(m, va, 0)
    rn = ng.eval(va, 4, 0)
    check("評価窓の数", rn["n"] == len(y), f"({rn['n']} vs {len(y)})")
    Pn = ng._dense(np.array([[ng.idx[c] for c in va[0][i:i + 4]] for i in range(10)]))
    check("n-gram の確率の和が1", np.allclose(Pn.sum(1), 1.0))
    for mc in (1, 2):
        nk = NgramLM(3, mc, smoothing="kn").fit(tr)
        Pk = nk._dense(np.array([[nk.idx[c] for c in va[0][i:i + 4]] for i in range(10)]))
        check(f"n-gram (Kneser-Ney, 剪定{mc}) の確率の和が1", np.allclose(Pk.sum(1), 1.0) and (Pk > 0).all())

    if torch is None:
        print("[GPU版] torch が見つかりません (pip install torch)。numpy 版の確認だけ行いました。")
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[GPU版] torch {torch.__version__} | デバイス: {dev} | numpy 版と照合")
        for tag, cfg in cfgs.items():
            cfg = dict(cfg)
            cfg["detector"] = None  # 雑音の乱数は照合できないので ADC なしの理想検出で
            a = OpticalLM(**cfg)
            for acc in ("fp32", "fp64"):
                b = OpticalLM(backend="torch", acc=acc, **cfg)
                a2 = OpticalLM(**cfg)
                a2.fit(docs)
                b.fit(docs)
                X, _ = a2._windows(a2.rows_for(docs[0][:200]))
                r_f = rel(a2.features(X), b.features(X))
                r_A = rel(a2.A, b._np(b.A))
                r_B = rel(a2.Bt, b._np(b.Bt))
                a2.solve(1e-2)
                b.solve(1e-2)
                Sa, Sb = a2.scores(X), b.scores(X)
                okm = np.isfinite(Sa) & np.isfinite(Sb)
                agree = float((Sa.argmax(1) == Sb.argmax(1)).mean())
                check(f"{tag} acc={acc}", r_f < 1e-3 and r_A < 1e-3 and r_B < 1e-3 and agree >= 0.98,
                      f"(特徴 {r_f:.1e} / A {r_A:.1e} / B {r_B:.1e} / argmax 一致 {agree * 100:.0f}%)")
                bb = copy.deepcopy(b)
                bb.forget(docs[5:])
                f5 = OpticalLM(backend="torch", acc=acc, **cfg)
                f5.fit(docs[:5])
                check(f"{tag} acc={acc} 取り消し", rel(b._np(f5.A), b._np(bb.A)) < 1e-5, f"({rel(b._np(f5.A), b._np(bb.A)):.1e})")
    if not ok_all:
        print("\n判定: NG (出力をそのまま共有してください)")
    elif torch is None:
        print("\n判定: numpy 版は OK。GPU 版は未確認です (torch を入れて、GPU のあるマシンでもう一度実行してください)")
    else:
        print("\n判定: OK (--gpu を付けて実行して大丈夫です)")
    return ok_all


# ====================================================================== CLI
def _cli(argv):
    ap = argparse.ArgumentParser(prog="opticlm", description="光学リザバー言語モデル")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("selftest")
    t = sub.add_parser("train")
    t.add_argument("corpus")
    t.add_argument("out")
    for p in (t,):
        p.add_argument("--grid", type=int, default=32)
        p.add_argument("--scales", default="1,2,3,4")
        p.add_argument("--max-chars", type=int, default=0)
        p.add_argument("--chunks", type=int, default=1)
        p.add_argument("--w-bits", type=int, default=32)
        p.add_argument("--w-group", type=int, default=0, help="W のグループ量子化の行数 (例 256。--w-bits 6 以下で有効)")
        p.add_argument("--no-stats", action="store_true")
        p.add_argument("--refit", action="store_true", help="検証用の文書も学習に足して解き直す (最終版用。eval の数字は甘くなる)")
        p.add_argument("--gpu", action="store_true")
        p.add_argument("--workers", type=int, default=0, help="CPU の並列学習のプロセス数 (2以上で有効。優先度を下げて動く)")
        p.add_argument("--threads", type=int, default=2, help="並列学習の1プロセスあたりのスレッド数")
    e = sub.add_parser("eval")
    e.add_argument("model")
    e.add_argument("corpus")
    e.add_argument("--gpu", action="store_true")
    g = sub.add_parser("gen")
    g.add_argument("model")
    g.add_argument("prompt")
    g.add_argument("--n", type=int, default=100)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--forbid", default="〓")
    g.add_argument("--gpu", action="store_true")
    b = sub.add_parser("bench")
    b.add_argument("corpus")
    b.add_argument("--d", default="1024,2048,4096")
    b.add_argument("--scales", default="1,2,3,4")
    b.add_argument("--max-chars", type=int, default=0)
    b.add_argument("--no-mix", action="store_true", help="混合の公平な比較 (mix_compare) を省く")
    b.add_argument("--ng-min", default="1,2,3,5,10", help="n-gram の剪定 (出現がこの回数未満の項目を捨てる) の一覧")
    b.add_argument("--ng-smoothing", default="kn", choices=("wb", "kn"),
                   help="比較相手の n-gram の平滑化 (kn: Kneser-Ney、wb: Witten-Bell。kn の方が強い)")
    b.add_argument("--gpu", action="store_true")
    ns = ap.parse_args(argv)
    be = lambda gpu: "torch" if gpu else "numpy"
    if ns.cmd == "selftest":
        selftest()
    elif ns.cmd == "train":
        t0 = time.time()
        model, rep = train(ns.corpus, grid=ns.grid, scales=tuple(int(x) for x in ns.scales.split(",")),
                           max_chars=ns.max_chars, chunks=ns.chunks, backend=be(ns.gpu), refit_all=ns.refit,
                           workers=ns.workers, threads=ns.threads)
        print(f"検証 acc1 {rep['acc1']:.1f} acc5 {rep['acc5']:.1f} nll {rep['nll']:.3f} (較正 {rep['cal']['kind']}) | {time.time() - t0:.0f}s")
        model.save(ns.out, stats=not ns.no_stats, w_bits=ns.w_bits, w_group=ns.w_group)
        print(f"保存: {ns.out} ({os.path.getsize(ns.out) / 1e6:.1f}MB) | 推論用の重み {model.infer_bytes(ns.w_bits, ns.w_group) / 1e6:.2f}MB")
    elif ns.cmd == "eval":
        m = load(ns.model, backend=be(ns.gpu))
        docs = read_docs(ns.corpus)
        _, va = split_docs(docs)
        r = evaluate(m, va)
        print(f"検証 {r['n']}窓: acc1 {r['acc1']:.1f} acc5 {r['acc5']:.1f} nll {r['nll']:.3f}")
    elif ns.cmd == "gen":
        m = load(ns.model, backend=be(ns.gpu))
        print(m.generate(ns.prompt, ns.n, temperature=ns.temperature, forbid=ns.forbid))
    elif ns.cmd == "bench":
        size_bench(ns.corpus, d_list=tuple(int(x) for x in ns.d.split(",")), scales=tuple(int(x) for x in ns.scales.split(",")),
                   backend=be(ns.gpu), max_chars=ns.max_chars, mix=not ns.no_mix,
                   ng_min=tuple(int(x) for x in ns.ng_min.split(",")), ng_smoothing=ns.ng_smoothing)
    else:
        ap.print_help()


if __name__ == "__main__":
    _cli(sys.argv[1:])
