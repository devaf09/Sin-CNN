import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import time
import json

# 再現性のための乱数シード固定
torch.manual_seed(42)
np.random.seed(42)

# ========================================================
# 0. ハイパーパラメータ設定
# ========================================================
IMAGE_SIZE = 48      # 光空間解像度 (48x48 = 2,304ピクセル)
NUM_FREQS = 256      # 周波数成分
NUM_LAYERS = 8       # 位相レイヤー数
WAVELENGTH = 633e-9 # 赤色レーザー波長 (633nm)
DX = 20e-6          # ピクセルピッチ (20µm)
PROP_DIST = 0.002   # 層間伝播距離 (2.0mm)

CONTEXT_LEN = 8     # 過去8文字の文脈を考慮
BATCH_SIZE = 64     # ミニバッチサイズ
LEARNING_RATE = 0.005

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- 実行デバイス: {device} ---")

# ========================================================
# 1. .txt ファイル読み込み & 辞書自動構築
# ========================================================
def load_text_and_build_vocab(filepath="sample_corpus.txt"):
    if not os.path.exists(filepath):
        # テスト用ダミーファイルの生成
        print(f"'{filepath}' が存在しないため、サンプルを自動生成します。")
        sample_data = "光計算技術を用いた次世代AIアーキテクチャの実験を行っています。光の干渉と回折を利用したニューラルネットワークです。" * 3
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(sample_data)
            
    print(f"テキストファイル '{filepath}' を読み込んでいます...")
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read().strip().replace("\n", "")
        
    vocab = sorted(list(set(text)))
    char2id = {c: i for i, c in enumerate(vocab)}
    id2char = {i: c for i, c in enumerate(vocab)}
    
    print(f"辞書(Vocab)構築完了 → 総文字数: {len(text)}文字 | ユニーク単語/文字数: {len(vocab)}個")
    return text, vocab, char2id, id2char

# ========================================================
# 2. 直交位相割り当てモジュール & 光ニューラルネットワーク
# ========================================================
class OrthogonalHolographicMultiplexer(nn.Module):
    """
    【概念2: ホログラム加算 & 直交位相割り当て】
    8文字のコンテキスト位置に対して 2π/8刻みの直交位相 ϕ_m を割り当て、
    自乗検波時のクロストーク項を最小化しながら複素電場へ多重化します。
    """
    def __init__(self, context_len=8, grid_size=48):
        super().__init__()
        self.context_len = context_len
        self.grid_size = grid_size
        
        # 2πを等分する直交位相角度: [0, π/4, π/2, 3π/4, π, 5π/4, 3π/2, 7π/4]
        angles = np.linspace(0, 2 * np.pi, context_len, endpoint=False)
        self.register_buffer('phase_angles', torch.tensor(angles, dtype=torch.float32))
        
        # 各位置の寄与度を微調整する学習可能重み
        self.pos_weights = nn.Parameter(torch.linspace(0.3, 1.0, context_len))

    def forward(self, token_embeddings):
        """
        token_embeddings: [Batch, context_len, 2304]
        """
        batch_size = token_embeddings.shape[0]
        A = token_embeddings.view(batch_size, self.context_len, self.grid_size, self.grid_size)
        
        phi = self.phase_angles.view(1, self.context_len, 1, 1)
        w = self.pos_weights.view(1, self.context_len, 1, 1)
        
        # 複素オイラー変換 U_m = w_m * A_m * (cos(ϕ_m) + i*sin(ϕ_m)) による重ね合わせ
        u_real = torch.sum(w * A * torch.cos(phi), dim=1)
        u_imag = torch.sum(w * A * torch.sin(phi), dim=1)
        
        return torch.complex(u_real, u_imag)

class OpticalPhaseLayer2D_Continuous(nn.Module):
    """
    【概念1: 空間周波数格子によるパラメータ圧縮 O(N^2) -> O(N)】
    """
    def __init__(self, grid_size=48, num_freqs=256):
        super().__init__()
        self.grid_size = grid_size
        self.num_freqs = num_freqs
        
        self.freq_x = nn.Parameter(torch.randn(num_freqs) * 0.2)
        self.freq_y = nn.Parameter(torch.randn(num_freqs) * 0.2)
        
        x = torch.linspace(-grid_size // 2, grid_size // 2 - 1, grid_size) * DX
        y = torch.linspace(-grid_size // 2, grid_size // 2 - 1, grid_size) * DX
        Y, X = torch.meshgrid(y, x, indexing='ij')
        self.register_buffer('X', X)
        self.register_buffer('Y', Y)
        
        freq_idx = torch.arange(1, num_freqs + 1, dtype=torch.float32).view(-1, 1, 1)
        w = freq_idx * np.pi / (grid_size * DX)
        
        self.register_buffer('cos_wx', torch.cos(w * X.unsqueeze(0)))
        self.register_buffer('sin_wy', torch.sin(w * Y.unsqueeze(0)))
        
        k = 2 * np.pi / WAVELENGTH
        lens_phase = -k * (self.X**2 + self.Y**2) / (2 * 0.025)
        self.register_buffer('lens_phase', lens_phase)

    def get_phase_map(self):
        phase_map = torch.einsum('f, fhw -> hw', self.freq_x, self.cos_wx) + \
                    torch.einsum('f, fhw -> hw', self.freq_y, self.sin_wy)
        return (phase_map + self.lens_phase) % (2 * np.pi)

    def forward(self, input_wave):
        phase_map = torch.einsum('f, fhw -> hw', self.freq_x, self.cos_wx) + \
                    torch.einsum('f, fhw -> hw', self.freq_y, self.sin_wy)
        
        final_phase = phase_map + self.lens_phase
        phase_factor = torch.complex(torch.cos(final_phase), torch.sin(final_phase))
        return input_wave * phase_factor

class AngularSpectrumPropagation(nn.Module):
    """
    【概念3: 計算量の削減 (角スペクトル自由空間伝播)】
    """
    def __init__(self, grid_size=48, distance=0.002):
        super().__init__()
        df = 1.0 / (grid_size * DX)
        fx = torch.linspace(-grid_size // 2, grid_size // 2 - 1, grid_size) * df
        fy = torch.linspace(-grid_size // 2, grid_size // 2 - 1, grid_size) * df
        FX, FY = torch.meshgrid(fy, fx, indexing='ij')
        
        k = 2 * np.pi / WAVELENGTH
        kx = 2 * np.pi * FX
        ky = 2 * np.pi * FY
        
        kz_sq = k**2 - kx**2 - ky**2
        kz = torch.where(kz_sq >= 0, torch.sqrt(torch.relu(kz_sq)), 1j * torch.sqrt(torch.relu(-kz_sq)))
        
        transfer_func = torch.exp(1j * kz * distance)
        self.register_buffer('H', transfer_func)

    def forward(self, U_in):
        return torch.fft.ifft2(torch.fft.fft2(U_in, dim=(-2, -1)) * self.H, dim=(-2, -1))

class Context8PhotonicLLM(nn.Module):
    def __init__(self, vocab_size, grid_size=48, num_freqs=256, num_layers=8, context_len=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.grid_size = grid_size
        self.num_layers = num_layers
        self.context_len = context_len
        
        self.token_embedding = nn.Embedding(vocab_size, grid_size * grid_size)
        
        # 直交位相ホログラム多重化モジュール
        self.multiplexer = OrthogonalHolographicMultiplexer(context_len=context_len, grid_size=grid_size)
        
        self.phase_layers = nn.ModuleList([
            OpticalPhaseLayer2D_Continuous(grid_size, num_freqs) for _ in range(num_layers)
        ])
        self.propagations = nn.ModuleList([
            AngularSpectrumPropagation(grid_size, distance=PROP_DIST) for _ in range(num_layers)
        ])
        
        self.classifier_head = nn.Linear(grid_size * grid_size, vocab_size)

    def forward(self, context_tokens):
        # 1. 埋め込みベクトルの取得: [Batch, 8, 2304]
        embeds = self.token_embedding(context_tokens)
        
        # 2. 直交位相割り当てによるホログラム加算: [Batch, 48, 48] (complex64)
        U = self.multiplexer(embeds)

        # 3. 光位相変調 & 自由空間伝播 (8層)
        for i in range(self.num_layers):
            U = self.phase_layers[i](U)
            U = self.propagations[i](U)
            if i < self.num_layers - 1:
                intensity = torch.abs(U)**2
                norm_factor = torch.sqrt(intensity.mean(dim=(-2, -1), keepdim=True) + 1e-8)
                U = U / norm_factor
                
        # 4. 受光素子での強度検波 |U|^2 ＆ 分類器
        final_intensity = torch.abs(U)**2
        flattened_intensity = final_intensity.view(final_intensity.size(0), -1)
        logits = self.classifier_head(flattened_intensity)
        return logits, final_intensity

# ========================================================
# 3. 圧縮モデル＆メタデータのエクスポート機能 (最適化版)
# ========================================================
def save_model_and_datasets(model, vocab, char2id, id2char, output_dir="optics_export"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. バッファ(展開用大容量テンソル)を除外した真の軽量チェックポイントの保存
    raw_state = model.state_dict()
    clean_state_dict = {}
    ignored_keys = ["cos_wx", "sin_wy", "X", "Y", "lens_phase", "H"]
    
    for k, v in raw_state.items():
        if not any(buf in k for buf in ignored_keys):
            clean_state_dict[k] = v.cpu()

    checkpoint_path = os.path.join(output_dir, "optical_llm.pth")
    torch.save({
        'model_state_dict': clean_state_dict,
        'vocab': vocab,
        'char2id': char2id,
        'id2char': id2char,
        'config': {
            'grid_size': IMAGE_SIZE,
            'num_freqs': NUM_FREQS,
            'num_layers': NUM_LAYERS,
            'context_len': CONTEXT_LEN
        }
    }, checkpoint_path)
    
    file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"[保存] 軽量PyTorchチェックポイント: {checkpoint_path} ({file_size_mb:.2f} MB)")
    
    # 2. 語彙メタデータのJSON保存
    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({'vocab': vocab, 'char2id': char2id}, f, ensure_ascii=False, indent=2)
    print(f"[保存] 語彙メタデータ: {vocab_path}")

    # 3. 実空間2D位相マスクデータのNumpy出力
    phase_maps = [layer.get_phase_map().detach().cpu().numpy() for layer in model.phase_layers]
    phase_maps_np = np.stack(phase_maps, axis=0) # [8, 48, 48]
    phase_path = os.path.join(output_dir, "phase_masks_2d.npy")
    np.save(phase_path, phase_maps_np)
    print(f"[保存] 8層分の光学位相板マスクデータ: {phase_path} (Shape: {phase_maps_np.shape})")

    # 4. 純粋パラメータのみのRawバイナリ出力 (設計の核)
    raw_params_path = os.path.join(output_dir, "continuous_freq_weights.bin")
    freq_weights = []
    for layer in model.phase_layers:
        freq_weights.append(layer.freq_x.detach().cpu().numpy())
        freq_weights.append(layer.freq_y.detach().cpu().numpy())
    
    freq_array = np.concatenate(freq_weights).astype(np.float32)
    freq_array.tofile(raw_params_path)
    print(f"[保存] 周波数パラメータRawバイナリ: {raw_params_path} ({os.path.getsize(raw_params_path)/1024:.1f} KB)")

# ========================================================
# 4. メイン実行パイプライン
# ========================================================
if __name__ == "__main__":
    FILE_PATH = "sample_corpus.txt" 
    raw_text, vocab, char2id, id2char = load_text_and_build_vocab(FILE_PATH)
    vocab_size = len(vocab)
    
    token_ids = [char2id[c] for c in raw_text]
    
    inputs, targets = [], []
    for i in range(len(token_ids) - CONTEXT_LEN):
        inputs.append(token_ids[i : i + CONTEXT_LEN])
        targets.append(token_ids[i + CONTEXT_LEN])
        
    input_tokens = torch.tensor(inputs, dtype=torch.long)
    target_tokens = torch.tensor(targets, dtype=torch.long)
    
    dataloader = DataLoader(TensorDataset(input_tokens, target_tokens), batch_size=BATCH_SIZE, shuffle=True)
    
    model = Context8PhotonicLLM(
        vocab_size=vocab_size, 
        grid_size=IMAGE_SIZE, 
        num_freqs=NUM_FREQS, 
        num_layers=NUM_LAYERS,
        context_len=CONTEXT_LEN
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 100
    print(f"\n--- CONTEXT_LEN={CONTEXT_LEN} (直交位相多重化・光AI) 学習開始 (全 {epochs} エポック) ---")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct_preds, total_samples = 0.0, 0, 0
        
        for batch_inputs, batch_targets in dataloader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch_inputs)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_inputs.size(0)
            preds = logits.argmax(dim=1)
            correct_preds += (preds == batch_targets).sum().item()
            total_samples += batch_inputs.size(0)
            
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch [{epoch:03d}/{epochs:03d}] Loss: {total_loss/total_samples:.4f} | 記憶精度: {(correct_preds/total_samples)*100:.1f}%")
            
    # 学習済みデータの出力実行
    print("\n--- 学習済みデータの出力・保存開始 ---")
    save_model_and_datasets(model, vocab, char2id, id2char)
