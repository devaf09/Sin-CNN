import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# 再現性のための乱数シード固定
torch.manual_seed(42)
np.random.seed(42)

# ========================================================
# 1. ハイパーパラメータ設定 (128軸・8層モデル)
# ========================================================
IMAGE_SIZE = 32      # 解像度 (32x32ピクセル)
NUM_FREQS = 128     # 周波数軸数 (128軸)
NUM_LAYERS = 8      # 位相レイヤー数 (8層)
WAVELENGTH = 633e-9 # 光の波長 (633nm: 赤色レーザー)
DX = 10e-6          # ピクセルピッチ (10µm)
PROP_DIST = 0.015   # 層間伝播距離 (1.5cm)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- 実行デバイス: {device} ---")

# ========================================================
# 2. 光学AI (D²NN) モデルの定義
# ========================================================
def gaussian_blur_simple(tensor):
    """位相マスクをなめらかにする平滑化フィルタ"""
    kernel = torch.tensor([[1., 2., 1.],
                           [2., 4., 2.],
                           [1., 2., 1.]], device=tensor.device) / 16.0
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    tensor_4d = tensor.unsqueeze(0).unsqueeze(0)
    padded = F.pad(tensor_4d, (1, 1, 1, 1), mode='reflect')
    blurred = F.conv2d(padded, kernel)
    return blurred.squeeze()

class OpticalPhaseLayer2D_Continuous(nn.Module):
    """128軸周波数パラメータからなめらかな連続位相マスクを合成するレイヤー"""
    def __init__(self, grid_size=32, num_freqs=128):
        super().__init__()
        self.grid_size = grid_size
        self.num_freqs = num_freqs
        
        # X軸・Y軸の周波数パラメータ (学習対象)
        self.freq_x = nn.Parameter(torch.randn(num_freqs) * 0.08)
        self.freq_y = nn.Parameter(torch.randn(num_freqs) * 0.08)
        
        x = torch.linspace(-grid_size // 2, grid_size // 2 - 1, grid_size) * DX
        y = torch.linspace(-grid_size // 2, grid_size // 2 - 1, grid_size) * DX
        Y, X = torch.meshgrid(y, x, indexing='ij')
        self.register_buffer('X', X)
        self.register_buffer('Y', Y)
        
        # フレネルレンズ位相 (光の集束)
        k = 2 * np.pi / WAVELENGTH
        f_dist = 0.03
        lens_phase = -k * (self.X**2 + self.Y**2) / (2 * f_dist)
        self.register_buffer('lens_phase', lens_phase)

    def get_phase_map(self):
        phase_map = torch.zeros((self.grid_size, self.grid_size), device=self.freq_x.device)
        for i in range(self.num_freqs):
            wx = (i + 1) * np.pi / (self.grid_size * DX)
            wy = (i + 1) * np.pi / (self.grid_size * DX)
            phase_map += self.freq_x[i] * torch.cos(wx * self.X) + self.freq_y[i] * torch.sin(wy * self.Y)
            
        phase_map = gaussian_blur_simple(phase_map)
        return phase_map + self.lens_phase

    def forward(self, input_wave):
        final_phase = self.get_phase_map()
        phase_factor = torch.complex(torch.cos(final_phase), torch.sin(final_phase))
        return input_wave * phase_factor

class AngularSpectrumPropagation(nn.Module):
    """角スペクトル法 (ASM) による光波の空間伝播シミュレータ"""
    def __init__(self, grid_size=32, distance=0.015):
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
        U_fft = torch.fft.fft2(U_in)
        U_propagated = torch.fft.ifft2(U_fft * self.H)
        return U_propagated

class AdvancedDeepOpticalAI_Continuous(nn.Module):
    """8層構造 Deep 光ニューラルネットワーク"""
    def __init__(self, num_classes=10, grid_size=32, num_freqs=128, num_layers=8):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.num_layers = num_layers
        
        self.phase_layers = nn.ModuleList([
            OpticalPhaseLayer2D_Continuous(grid_size, num_freqs) for _ in range(num_layers)
        ])
        self.propagations = nn.ModuleList([
            AngularSpectrumPropagation(grid_size, distance=PROP_DIST) for _ in range(num_layers)
        ])

    def calc_tv_loss(self):
        """位相マスクのなめらかさ(全変動)損失"""
        tv_loss = 0.0
        for layer in self.phase_layers:
            pm = layer.get_phase_map()
            diff_h = torch.mean(torch.abs(pm[:, 1:] - pm[:, :-1]))
            diff_v = torch.mean(torch.abs(pm[1:, :] - pm[:-1, :]))
            tv_loss += diff_h + diff_v
        return tv_loss

    def forward(self, img):
        # 複素光波へのエンコーディング
        U = torch.complex(torch.sqrt(img), torch.zeros_like(img))
        
        for i in range(self.num_layers):
            U = self.phase_layers[i](U)
            U = self.propagations[i](U)
            if i < self.num_layers - 1:
                intensity = torch.abs(U)**2
                U = U * torch.sigmoid(intensity * 0.05) # 層間非線形処理
                
        final_intensity = torch.abs(U)**2
        
        # 10箇所の集光ターゲット位置での光パワー測定
        logits = []
        step = self.grid_size // int(np.ceil(np.sqrt(self.num_classes)))
        count = 0
        for r in range(0, self.grid_size, step):
            for c in range(0, self.grid_size, step):
                if count < self.num_classes:
                    spot_power = final_intensity[:, r:r+step, c:c+step].sum(dim=(1, 2))
                    logits.append(spot_power)
                    count += 1
                    
        logits = torch.stack(logits, dim=1)
        scaled_logits = logits * 10000.0 # 勾配最適化用スケール調整
        return scaled_logits, final_intensity

# ========================================================
# 3. データ読み込みと前処理 (Train 80% / Test 20%)
# ========================================================
print("手書き数字データセット(sklearn digits: 1,797枚)をロード中...")
digits = load_digits()
data, y = digits.images, digits.target

# 学習用(1,437枚)と未学習テスト用(360枚)に分割
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    data, y, test_size=0.20, random_state=42, stratify=y
)

def preprocess_dataset(X_raw):
    tensor_raw = torch.tensor(X_raw, dtype=torch.float32).unsqueeze(1)
    unsampled = F.interpolate(tensor_raw, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
    unsampled_min = unsampled.min()
    unsampled_max = unsampled.max()
    normalized = (unsampled - unsampled_min) / (unsampled_max - unsampled_min + 1e-8)
    return normalized.squeeze(1)

X_train = preprocess_dataset(X_train_raw).to(device)
y_train = torch.tensor(y_train_raw, dtype=torch.long).to(device)
X_test = preprocess_dataset(X_test_raw).to(device)
y_test = torch.tensor(y_test_raw, dtype=torch.long).to(device)

print(f"データ準備完了 → 学習用: {len(X_train)}枚 | 完全未学習テスト用: {len(X_test)}枚")

# ========================================================
# 4. モデルの学習パイプライン
# ========================================================
model = AdvancedDeepOpticalAI_Continuous(num_classes=10, grid_size=IMAGE_SIZE, num_freqs=NUM_FREQS, num_layers=NUM_LAYERS).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

batch_size = 128
epochs = 25

print("\n--- 8層光学AIモデルの学習開始 ---")
for epoch in range(1, epochs + 1):
    model.train()
    permutation = torch.randperm(X_train.size(0))
    epoch_loss = 0.0
    correct_train = 0
    
    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]
        
        optimizer.zero_grad()
        scaled_logits, _ = model(batch_x)
        
        ce_loss = criterion(scaled_logits, batch_y)
        tv_loss = model.calc_tv_loss() * 0.0001
        loss = ce_loss + tv_loss
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * batch_x.size(0)
        correct_train += (scaled_logits.argmax(dim=1) == batch_y).sum().item()
        
    scheduler.step()
    train_acc = (correct_train / len(X_train)) * 100
    
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Loss: {epoch_loss/len(X_train):.4f} | 学習データ精度: {train_acc:.1f}%")

# ========================================================
# 5. 未学習テストデータでの総合検証
# ========================================================
model.eval()
with torch.no_grad():
    scaled_logits_test, final_intensities = model(X_test)
    test_preds = scaled_logits_test.argmax(dim=1)
    test_acc = (test_preds == y_test).float().mean().item() * 100

print("\n" + "="*50)
print(f"★ 完全未学習テストデータ (360枚) の総合識別精度: {test_acc:.2f}% ★")
print("="*50 + "\n")

# ========================================================
# 6. 可視化レポートの生成&画像保存 (`optical_ai_report.png`)
# ========================================================
sample_indices = torch.randint(0, len(X_test), (10,))
fig, axes = plt.subplots(5, 4, figsize=(14, 15))
fig.suptitle(f"Real Handwritten Digits (Unseen Test Set) Optical AI Recognition Report\n[ Test Accuracy: {test_acc:.1f}% | 128-Freq / 8-Layer D2NN Model ]", fontsize=15, fontweight='bold', y=0.98)

for i in range(10):
    test_idx = sample_indices[i]
    img_in = X_test[test_idx].cpu().numpy()
    img_out = final_intensities[test_idx].cpu().numpy()
    
    true_c = y_test[test_idx].item()
    pred_c = test_preds[test_idx].item()
    
    row_in = (i // 2) * 2
    col_in = (i % 2) * 2
    row_out = row_in + 1
    col_out = col_in
    
    # 入力画像
    axes[row_in, col_in].imshow(img_in, cmap='gray')
    axes[row_in, col_in].set_title(f"Unseen Test '{true_c}': Input", fontsize=11, fontweight='bold')
    axes[row_in, col_in].axis('off')
    
