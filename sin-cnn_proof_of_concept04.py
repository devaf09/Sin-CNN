import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# 再現性のための乱数シード固定
torch.manual_seed(42)
np.random.seed(42)

# ========================================================
# 0. 高スペック大型ハイパーパラメータ設定
# ========================================================
IMAGE_SIZE = 48      # 光空間解像度 (48x48 = 2,304マス)
NUM_FREQS = 512     # 空間周波数軸数 (512軸: 表現力を極限拡張)
NUM_LAYERS = 16     # 位相レイヤー数 (16層 Deep D²NN)
WAVELENGTH = 633e-9 # 赤色レーザー波長 (633nm)
DX = 8e-6           # ピクセルピッチ (8µm)
PROP_DIST = 0.010   # 層間伝播距離 (1.0cm)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- 実行デバイス: {device} ---")

# ========================================================
# 1. .txt ファイル読み込み & 辞書自動構築
# ========================================================
def load_text_and_build_vocab(filepath="sample_corpus.txt"):
    """ .txt ファイルを読み込み、自動で辞書 (Vocab) を構築する関数 """
    if not os.path.exists(filepath):
        print(f"提示されたファイル '{filepath}' が見つからないため、デモ用テキストファイルを作成します...")
        sample_text = (
            "光AIは光の回折と干渉を利用した超高速な計算技術です。\n"
            "空間周波数パラメータと多層位相プレートを用いることで、\n"
            "電気代をほぼゼロに抑えた行列計算とテキスト予測を実現します。\n"
            "このモデルはテキストファイルから自動的に辞書を構築し学習を行います。"
        )
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(sample_text)
            
    print(f"テキストファイル '{filepath}' を読み込んでいます...")
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read().strip().replace("\n", "")
        
    # ユニークな文字を取り出して辞書を作成
    vocab = sorted(list(set(text)))
    char2id = {c: i for i, c in enumerate(vocab)}
    id2char = {i: c for i, c in enumerate(vocab)}
    
    print(f"辞書(Vocab)構築完了 → ユニーク単語/文字数: {len(vocab)}個")
    return text, vocab, char2id, id2char

# ========================================================
# 2. 16層・512軸 大型光ニューラルネットワーク定義
# ========================================================
def gaussian_blur_simple(tensor):
    kernel = torch.tensor([[1., 2., 1.],
                           [2., 4., 2.],
                           [1., 2., 1.]], device=tensor.device) / 16.0
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    tensor_4d = tensor.unsqueeze(0).unsqueeze(0)
    padded = F.pad(tensor_4d, (1, 1, 1, 1), mode='reflect')
    blurred = F.conv2d(padded, kernel)
    return blurred.squeeze()

class OpticalPhaseLayer2D_Continuous(nn.Module):
    """ 512軸の高解像度連続位相モジュール """
    def __init__(self, grid_size=48, num_freqs=512):
        super().__init__()
        self.grid_size = grid_size
        self.num_freqs = num_freqs
        
        self.freq_x = nn.Parameter(torch.randn(num_freqs) * 0.08)
        self.freq_y = nn.Parameter(torch.randn(num_freqs) * 0.08)
        
        x = torch.linspace(-grid_size // 2, grid_size // 2 - 1, grid_size) * DX
        y = torch.linspace(-grid_size // 2, grid_size // 2 - 1, grid_size) * DX
        Y, X = torch.meshgrid(y, x, indexing='ij')
        self.register_buffer('X', X)
        self.register_buffer('Y', Y)
        
        k = 2 * np.pi / WAVELENGTH
        lens_phase = -k * (self.X**2 + self.Y**2) / (2 * 0.025)
        self.register_buffer('lens_phase', lens_phase)

    def forward(self, input_wave):
        phase_map = torch.zeros((self.grid_size, self.grid_size), device=self.freq_x.device)
        for i in range(self.num_freqs):
            wx = (i + 1) * np.pi / (self.grid_size * DX)
            wy = (i + 1) * np.pi / (self.grid_size * DX)
            phase_map += self.freq_x[i] * torch.cos(wx * self.X) + self.freq_y[i] * torch.sin(wy * self.Y)
            
        final_phase = gaussian_blur_simple(phase_map) + self.lens_phase
        phase_factor = torch.complex(torch.cos(final_phase), torch.sin(final_phase))
        return input_wave * phase_factor

class AngularSpectrumPropagation(nn.Module):
    def __init__(self, grid_size=48, distance=0.010):
        super().__init__()
        df = 1.0 / (grid_size * DX)
        fx = torch.linspace(-grid_size // 2, grid_size // 2 - 1, grid_size) * df
        fy = torch.linspace(-grid_size // 2, grid_size // 2 - 1, grid_size) * df
        FX, FY = torch.meshgrid(fy, fx, indexing='ij')
        
        k = 2 * np.pi / WAVELENGTH
        arg = torch.clamp(k**2 - (2 * np.pi * FX)**2 - (2 * np.pi * FY)**2, min=0.0)
        kz = torch.sqrt(arg)
        
        transfer_func = torch.complex(torch.cos(kz * distance), torch.sin(kz * distance))
        self.register_buffer('H', transfer_func)

    def forward(self, U_in):
        return torch.fft.ifft2(torch.fft.fft2(U_in) * self.H)

class LargePhotonicLLM(nn.Module):
    """ 16層 Deep 光LLM (Photonic LLM) """
    def __init__(self, vocab_size, grid_size=48, num_freqs=512, num_layers=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.grid_size = grid_size
        self.num_layers = num_layers
        
        self.token_embedding = nn.Embedding(vocab_size, grid_size * grid_size)
        self.phase_layers = nn.ModuleList([
            OpticalPhaseLayer2D_Continuous(grid_size, num_freqs) for _ in range(num_layers)
        ])
        self.propagations = nn.ModuleList([
            AngularSpectrumPropagation(grid_size, distance=PROP_DIST) for _ in range(num_layers)
        ])

    def forward(self, token_id):
        embed = self.token_embedding(token_id).view(-1, self.grid_size, self.grid_size)
        U = torch.complex(torch.sort(embed).values, torch.zeros_like(embed)) + 1e-8
        
        for i in range(self.num_layers):
            U = self.phase_layers[i](U)
            U = self.propagations[i](U)
            if i < self.num_layers - 1:
                intensity = torch.abs(U)**2
                U = U * torch.sigmoid(intensity * 0.05)
                
        final_intensity = torch.abs(U)**2
        
        # 動的な語彙サイズに応じて集光スポット位置を割り当て
        step = self.grid_size // int(np.ceil(np.sqrt(self.vocab_size)))
        logits = []
        count = 0
        for r in range(0, self.grid_size, step):
            for c in range(0, self.grid_size, step):
                if count < self.vocab_size:
                    spot_power = final_intensity[:, r:r+step, c:c+step].sum(dim=(1, 2))
                    logits.append(spot_power)
                    count += 1
                    
        logits = torch.stack(logits, dim=1)
        scaled_logits = logits * 10000.0
        return scaled_logits, final_intensity

# ========================================================
# 3. メイン実行パイプライン (学習 & 生成)
# ========================================================
if __name__ == "__main__":
    # A. テキストファイル読み込み (ファイル名を適時変更可能)
    FILE_PATH = "sample_corpus.txt" 
    raw_text, vocab, char2id, id2char = load_text_and_build_vocab(FILE_PATH)
    vocab_size = len(vocab)
    
    # B. 入力データ (入力文字 → 次の文字) のシーケンス生成
    token_ids = [char2id[c] for c in raw_text]
    input_tokens = torch.tensor(token_ids[:-1], dtype=torch.long).to(device)
    target_tokens = torch.tensor(token_ids[1:], dtype=torch.long).to(device)
    
    # C. 16層・512軸大型光LLMモデル構築
    model = LargePhotonicLLM(vocab_size=vocab_size, grid_size=IMAGE_SIZE, num_freqs=NUM_FREQS, num_layers=NUM_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # D. 16層モデルでの学習 (120 エポック)
    epochs = 120
    print(f"\n--- 16層・512軸光AIによる辞書学習開始(全 {epochs} エポック) ---")
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        logits, _ = model(input_tokens)
        loss = criterion(logits, target_tokens)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0 or epoch == 1:
            preds = logits.argmax(dim=1)
            acc = (preds == target_tokens).float().mean() * 100
            print(f"Epoch [{epoch:03d}/{epochs:03d}] Loss: {loss.item():.4f} | 辞書記憶精度: {acc.item():.1f}%")
            
    # E. 自己回帰 (Auto-regressive) によるテキスト生成インターフェース
    model.eval()
    print("\n" + "="*50)
    print(" ★ 16層 光AI テキスト自動生成インターフェース ★ ")
    print("="*50)
    
    # プロンプト(開始の文字)を指定して文章を生成
    start_char = raw_text[0] # 先頭の1文字をデフォルト指定
    print(f"開始プロンプト文字: 『 {start_char} 』")
    
    gen_length = 30 # 生成する文字数
    current_token = torch.tensor([char2id[start_char]], dtype=torch.long).to(device)
    generated_text = [start_char]
    
    with torch.no_grad():
        for _ in range(gen_length):
            logits, _ = model(current_token)
            next_id = logits.argmax(dim=1).item()
            next_char = id2char[next_id]
            generated_text.append(next_char)
            current_token = torch.tensor([next_id], dtype=torch.long).to(device)
            
    print("\n【光AIによる生成文章結果】")
    print("".join(generated_text))
