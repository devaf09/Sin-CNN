"""
sin-cnn_diag06.py : 06 の「ロスが ln(V) 付近から下がらない」原因を切り分ける診断スクリプト。

使い方:
    python sin-cnn_diag06.py sample_corpus.txt 6
        第1引数: コーパス (省略時 sample_corpus.txt)
        第2引数: 各設定のエポック数 (省略時 6)

各設定を同じシード・同じデータで短く学習し、最後に比較表を出す。
  - 検証データ (コーパス後半20%) の損失と、最終位置の次文字予測精度を出す
  - 参考値として ln(V) と、ユニグラム(出現頻度だけ)予測のクロスエントロピーも出す
    -> これより損失が高いなら、文脈どころか頻度すら学べていない(最適化の失敗)

注意: このスクリプトは作成環境で実行できていない(構文チェックのみ)。
"""
import math
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

G = 48
NUM_FREQS = 256
NUM_LAYERS = 6
HARM = 3
WAVELENGTH = 633e-9
DX = 20e-6
PROP_DIST = 0.002
CTX = 8
BATCH = 64
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------ data
def make_windows(ids, ctx):
    ids = torch.tensor(ids, dtype=torch.long)
    n = len(ids) - ctx
    idx = torch.arange(ctx + 1).unsqueeze(0) + torch.arange(n).unsqueeze(1)
    w = ids[idx]
    return w[:, :-1], w[:, 1:]


# ------------------------------------------------------------ model parts
class PhaseLayer(nn.Module):
    """06 と同じ位相マスク。倍音ループは (h+1) 倍して足すだけなので、
    係数 sum_h (h+1)*freq[h] の1本に潰れる(ここでは等価な形で計算する)。"""

    def __init__(self):
        super().__init__()
        self.fx = nn.Parameter(torch.randn(HARM, NUM_FREQS) * 0.1)
        self.fy = nn.Parameter(torch.randn(HARM, NUM_FREQS) * 0.1)
        c = torch.linspace(-G // 2, G // 2 - 1, G) * DX
        Y, X = torch.meshgrid(c, c, indexing="ij")
        w = torch.arange(1, NUM_FREQS + 1, dtype=torch.float32).view(-1, 1, 1) * math.pi / (G * DX)
        self.register_buffer("cos_wx", torch.cos(w * X.unsqueeze(0)))
        self.register_buffer("sin_wy", torch.sin(w * Y.unsqueeze(0)))
        k = 2 * math.pi / WAVELENGTH
        self.register_buffer("lens", -k * (X ** 2 + Y ** 2) / (2 * 0.025))

    def forward(self, U):
        scale = torch.arange(1, HARM + 1, device=U.device, dtype=torch.float32).view(-1, 1)
        cx = (self.fx * scale).sum(0)
        cy = (self.fy * scale).sum(0)
        phase = (
            torch.einsum("f,fhw->hw", cx, self.cos_wx)
            + torch.einsum("f,fhw->hw", cy, self.sin_wy)
            + self.lens
        )
        pf = torch.complex(torch.cos(phase), torch.sin(phase)).reshape(1, 1, -1)
        return U * pf


class Prop(nn.Module):
    """角スペクトル伝搬。shift_fix=True で fft の周波数並びに合わせて ifftshift する。"""

    def __init__(self, shift_fix):
        super().__init__()
        df = 1.0 / (G * DX)
        f = torch.linspace(-G // 2, G // 2 - 1, G, dtype=torch.float64) * df
        FY, FX = torch.meshgrid(f, f, indexing="ij")
        k = 2 * math.pi / WAVELENGTH
        kz_sq = k ** 2 - (2 * math.pi * FX) ** 2 - (2 * math.pi * FY) ** 2
        kz = torch.sqrt(kz_sq.clamp(min=0))  # このピッチでは全成分が伝搬成分
        H = torch.exp(1j * kz * PROP_DIST)
        if shift_fix:
            H = torch.fft.ifftshift(H, dim=(-2, -1))
        self.register_buffer("H", H.to(torch.complex64))

    def forward(self, U):
        B, T, _ = U.shape
        u = U.reshape(B, T, G, G)
        return torch.fft.ifft2(torch.fft.fft2(u) * self.H).reshape(B, T, -1)


class Attn(nn.Module):
    """06 と同じ(強度から Q/K を作り、因果マスク付き softmax で複素波を重み付き和)。"""

    def __init__(self):
        super().__init__()
        self.q = nn.Linear(G * G, 64)
        self.k = nn.Linear(G * G, 64)

    def forward(self, U):
        I = U.real ** 2 + U.imag ** 2
        sc = torch.bmm(self.q(I), self.k(I).transpose(1, 2)) / 8.0
        T = U.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=U.device), diagonal=1)
        w = F.softmax(sc.masked_fill(mask, float("-inf")), dim=-1)
        return torch.bmm(w.to(U.dtype), U)


class Net(nn.Module):
    def __init__(self, V, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(V, G * G)
        self.pos = nn.Parameter(torch.randn(CTX) * 0.5)
        self.pl = nn.ModuleList([PhaseLayer() for _ in range(NUM_LAYERS)])
        self.pr = nn.ModuleList([Prop(cfg["shift"]) for _ in range(NUM_LAYERS)])
        self.at = nn.ModuleList([Attn() for _ in range(NUM_LAYERS)]) if cfg["attn"] else None
        self.thr = nn.Parameter(torch.full((NUM_LAYERS,), 0.02))
        self.ln = nn.LayerNorm(G * G) if cfg["head"] == "ln" else nn.Identity()
        self.head = nn.Linear(G * G, V)

    def post(self, U, i):
        if self.cfg["quench"]:  # modReLU (振幅を縮めるだけ。位相は保つ)
            amp = U.abs()
            U = U * (torch.relu(amp - self.thr[i]) / (amp + 1e-8))
        if self.cfg["norm"] == "amp06":  # 06 の正規化: 平均「振幅」の平方根で割る
            U = U / torch.sqrt(U.abs().mean(-1, keepdim=True) + 1e-8)
        elif self.cfg["norm"] == "power":  # 05 型: 平均「強度」が1になるよう割る
            p = (U.real ** 2 + U.imag ** 2).mean(-1, keepdim=True)
            U = U / torch.sqrt(p + 1e-8)
        return U

    def forward(self, x):
        B, T = x.shape
        e = self.emb(x)
        ph = self.pos[:T].view(1, T, 1)
        U = torch.complex(e * torch.cos(ph), e * torch.sin(ph))
        for i in range(NUM_LAYERS):
            U = self.pr[i](self.pl[i](U))
            if self.at is not None:
                U = self.at[i](U)
            U = self.post(U, i)
        I = U.real ** 2 + U.imag ** 2
        return self.head(self.ln(I))


# ------------------------------------------------------------ train / eval
@torch.no_grad()
def evaluate(model, data, V):
    model.eval()
    x, y = data
    tl, correct, n = 0.0, 0, 0
    for i in range(0, len(x), 256):
        xb, yb = x[i : i + 256].to(device), y[i : i + 256].to(device)
        lg = model(xb)
        tl += F.cross_entropy(lg.reshape(-1, V), yb.reshape(-1), reduction="sum").item()
        correct += (lg[:, -1].argmax(-1) == yb[:, -1]).sum().item()
        n += xb.size(0)
    return tl / (n * CTX), 100.0 * correct / n


def run(name, cfg, V, tr, va):
    torch.manual_seed(0)
    model = Net(V, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    dl = DataLoader(TensorDataset(*tr), batch_size=BATCH, shuffle=True)
    print(f"\n=== {name} === {cfg}")
    first_loss, last_loss, t0 = None, None, time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        tot, n, bad = 0.0, 0, False
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb).reshape(-1, V), yb.reshape(-1))
            if not math.isfinite(loss.item()):
                bad = True
                break
            loss.backward()
            if cfg["clip"] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
            tot += loss.item() * xb.size(0)
            n += xb.size(0)
        if bad:
            print(f"  ep{ep}: loss が NaN/Inf になったため中断")
            return dict(name=name, first=float("nan"), last=float("nan"), vl=float("nan"), acc=float("nan"))
        tl = tot / max(n, 1)
        vl, acc = evaluate(model, va, V)
        first_loss = tl if first_loss is None else first_loss
        last_loss = tl
        print(f"  ep{ep}: train {tl:.3f} | val {vl:.3f} | val最終位置acc {acc:.1f}% | {time.time() - t0:.0f}s")
    return dict(name=name, first=first_loss, last=last_loss, vl=vl, acc=acc)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_corpus.txt"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip().replace("\n", "")
    vocab = sorted(set(text))
    c2i = {c: i for i, c in enumerate(vocab)}
    ids = [c2i[c] for c in text]
    V = len(vocab)
    cut = int(len(ids) * 0.8)
    tr, va = make_windows(ids[:cut], CTX), make_windows(ids[cut:], CTX)

    counts = torch.bincount(tr[1].reshape(-1), minlength=V).float()
    p = counts / counts.sum()
    h1 = -(p[p > 0] * p[p > 0].log()).sum().item()
    print(f"device={device} | 語彙 V={V} | 学習窓 {len(tr[0])} / 検証窓 {len(va[0])}")
    print(f"参考: ln(V)={math.log(V):.3f} | ユニグラム(頻度のみ)のクロスエントロピー={h1:.3f}")
    print("      train損失がユニグラム値より高い/上がる -> 頻度すら学べていない(最適化の失敗)")

    base = dict(attn=True, quench=True, norm="amp06", head="raw", shift=False, lr=2e-3, clip=0.0)
    configs = [
        ("A  06そのまま", dict(base)),
        ("B  A + lr5e-4 + 勾配クリップ", dict(base, lr=5e-4, clip=1.0)),
        ("C  A + 出力にLayerNorm", dict(base, head="ln")),
        ("D  A + 強度正規化(05型)", dict(base, norm="power")),
        ("E  D + LayerNorm", dict(base, norm="power", head="ln")),
        ("F  E でアテンション無し", dict(base, norm="power", head="ln", attn=False)),
        ("G  F でクエンチ無し(05+因果)", dict(base, norm="power", head="ln", attn=False, quench=False)),
        ("H  全部入り+ifftshift", dict(base, norm="power", head="ln", shift=True, lr=1e-3, clip=1.0)),
    ]
    results = [run(n, c, V, tr, va) for n, c in configs]

    print("\n" + "=" * 78)
    print(f"ln(V)={math.log(V):.3f}  ユニグラム={h1:.3f}   (損失の単位: nat/文字)")
    print(f"{'設定':<30}{'train ep1':>10}{'train 最終':>11}{'val 損失':>10}{'val acc%':>10}")
    for r in results:
        print(f"{r['name']:<30}{r['first']:>10.3f}{r['last']:>11.3f}{r['vl']:>10.3f}{r['acc']:>10.1f}")
