import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import time

# 再現性のための乱数シード固定
torch.manual_seed(42)
np.random.seed(42)

# ========================================================
# 0. ハイパーパラメータ設定
# ========================================================
IMAGE_SIZE = 48      # 光空間解像度 (48x48 = 2,304ピクセル)
NUM_FREQS = 256      # 【表現力UP】8文字文脈に対応するため周波数成分を256に拡張
NUM_LAYERS = 8       # 位相レイヤー数
WAVELENGTH = 633e-9 # 赤色レーザー波長 (633nm)
DX = 20e-6          # ピクセルピッチ (20µm)
PROP_DIST = 0.002   # 層間伝播距離 (2.0mm)

# 【今回の核】コンテキスト長を 3 → 8 文字に拡張
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
        print(f"エラー: '{filepath}' が見つかりません。")
        exit()
            
    print(f"テキストファイル '{filepath}' を読み込んでいます...")
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read().strip().replace("\n", "")
        
    vocab = sorted(list(set(text)))
    char2id = {c: i for i, c in enumerate(vocab)}
    id2char = {i: c for i, c in enumerate(vocab)}
    
    print(f"辞書(Vocab)構築完了 → 総文字数: {len(text)}文字 | ユニーク単語/文字数: {len(vocab)}個")
    return text, vocab, char2id, id2char

# ========================================================
# 2. 8文字文脈対応 光ニューラルネットワーク
# ========================================================
class OpticalPhaseLayer2D_Continuous(nn.Module):
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

    def forward(self, input_wave):
        phase_map = torch.einsum('f, fhw -> hw', self.freq_x, self.cos_wx) + \
                    torch.einsum('f, fhw -> hw', self.freq_y, self.sin_wy)
        
        final_phase = phase_map + self.lens_phase
        phase_factor = torch.complex(torch.cos(final_phase), torch.sin(final_phase))
        return input_wave * phase_factor

class AngularSpectrumPropagation(nn.Module):
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
    """ 8文字文脈光位相エンコーディング対応 光LLM """
    def __init__(self, vocab_size, grid_size=48, num_freqs=256, num_layers=8, context_len=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.grid_size = grid_size
        self.num_layers = num_layers
        self.context_len = context_len
        
        self.token_embedding = nn.Embedding(vocab_size, grid_size * grid_size)
        
        # 8文字分の位置位相シフトおよび重み付けパラメータ
        self.pos_phase_shift = nn.Parameter(torch.randn(context_len) * 0.5)
        # 位置ごとの相対的な強度重み（学習可能）
        self.pos_weights = nn.Parameter(torch.linspace(0.3, 1.0, context_len))
        
        self.phase_layers = nn.ModuleList([
            OpticalPhaseLayer2D_Continuous(grid_size, num_freqs) for _ in range(num_layers)
        ])
        self.propagations = nn.ModuleList([
            AngularSpectrumPropagation(grid_size, distance=PROP_DIST) for _ in range(num_layers)
        ])
        
        self.classifier_head = nn.Linear(grid_size * grid_size, vocab_size)

    def forward(self, context_tokens):
        # context_tokens: [Batch, Context_Len (8)]
        batch_size = context_tokens.shape[0]
        
        U_combined = torch.zeros((batch_size, self.grid_size, self.grid_size), dtype=torch.complex64, device=context_tokens.device)
        
        # 8文字の多重光位相エンコーディング
        for pos_idx in range(self.context_len):
            tokens = context_tokens[:, pos_idx]
            embed = self.token_embedding(tokens).view(batch_size, self.grid_size, self.grid_size)
            
            phase_offset = self.pos_phase_shift[pos_idx]
            weight = self.pos_weights[pos_idx] # 位置ごとの重み
            
            # 位相回転と重み付けを行った光波の加算（光重畳）
            wave = weight * torch.complex(embed * torch.cos(phase_offset), embed * torch.sin(phase_offset))
            U_combined += wave

        U = U_combined
        
        # 8層 光回路伝播
        for i in range(self.num_layers):
            U = self.phase_layers[i](U)
            U = self.propagations[i](U)
            if i < self.num_layers - 1:
                intensity = torch.abs(U)**2
                norm_factor = torch.sqrt(intensity.mean(dim=(-2, -1), keepdim=True) + 1e-8)
                U = U / norm_factor
                
        final_intensity = torch.abs(U)**2
        flattened_intensity = final_intensity.view(batch_size, -1)
        logits = self.classifier_head(flattened_intensity)
        return logits, final_intensity

# ========================================================
# 3. サンプリング用補助関数
# ========================================================
def sample_next_token(logits, temperature=0.7, top_k=5):
    logits = logits / temperature
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item()

# ========================================================
# 4. メイン実行パイプライン
# ========================================================
if __name__ == "__main__":
    FILE_PATH = "sample_corpus.txt" 
    raw_text, vocab, char2id, id2char = load_text_and_build_vocab(FILE_PATH)
    vocab_size = len(vocab)
    
    token_ids = [char2id[c] for c in raw_text]
    
    # CONTEXT_LEN = 8 に応じたデータセット作成
    inputs = []
    targets = []
    for i in range(len(token_ids) - CONTEXT_LEN):
        inputs.append(token_ids[i : i + CONTEXT_LEN])
        targets.append(token_ids[i + CONTEXT_LEN])
        
    input_tokens = torch.tensor(inputs, dtype=torch.long)
    target_tokens = torch.tensor(targets, dtype=torch.long)
    
    dataset = TensorDataset(input_tokens, target_tokens)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = Context8PhotonicLLM(
        vocab_size=vocab_size, 
        grid_size=IMAGE_SIZE, 
        num_freqs=NUM_FREQS, 
        num_layers=NUM_LAYERS,
        context_len=CONTEXT_LEN
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 300
    print(f"\n--- CONTEXT_LEN={CONTEXT_LEN} (8文字文脈) 光AI学習開始(全 {epochs} エポック) ---")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_preds = 0
        total_samples = 0
        
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(batch_inputs)
            loss = criterion(logits, batch_targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_inputs.size(0)
            preds = logits.argmax(dim=1)
            correct_preds += (preds == batch_targets).sum().item()
            total_samples += batch_inputs.size(0)
            
        epoch_loss = total_loss / total_samples
        epoch_acc = (correct_preds / total_samples) * 100
        
        if epoch % 50 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch:03d}/{epochs:03d}] Loss: {epoch_loss:.4f} | 記憶精度: {epoch_acc:.1f}% | 経過時間: {elapsed:.1f}s")
            
    total_elapsed = time.time() - start_time
    print(f"\n学習完了！ 全計算時間: {total_elapsed/60:.2f} 分")
    
    # テキスト自動生成
    model.eval()
    print("\n" + "="*50)
    print(" ★ 8文字文脈対応 光AI テキスト自動生成 ★ ")
    print("="*50)
    
    # プロンプトも8文字に拡張
    prompt_chars = list(raw_text[:CONTEXT_LEN])
    print(f"開始プロンプト8文字: 『 {''.join(prompt_chars)} 』")
    
    gen_length = 100
    
    for mode in ["argmax (厳密)", "sampling (Temperature=0.7, Top-k=5)"]:
        current_context = [char2id[c] for c in prompt_chars]
        generated_text = list(prompt_chars)
        
        with torch.no_grad():
            for _ in range(gen_length):
                inp_tensor = torch.tensor([current_context], dtype=torch.long).to(device)
                logits, _ = model(inp_tensor)
                
                if "argmax" in mode:
                    next_id = logits.argmax(dim=1).item()
                else:
                    next_id = sample_next_token(logits, temperature=0.7, top_k=5)
                    
                next_char = id2char[next_id]
                generated_text.append(next_char)
                current_context = current_context[1:] + [next_id]
                
        print(f"\n【生成モード: {mode}】")
        print("".join(generated_text))