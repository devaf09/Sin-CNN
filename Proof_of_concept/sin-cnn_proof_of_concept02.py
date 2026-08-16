import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- CPUリソース割り当て ---
torch.set_num_threads(2)

class WaveCoreLayer(nn.Module):
	"""
	【ステップ1：位相変調器アレイ（Phase Shifter Array）】
	
	イメージ：28x28個の独立した「移相器（Phase Shifter）」が並んだパネルです。
	入力信号に対して、各ピクセルが個別に「位相遅れ」を与えます。
	"""
	def __init__(self, size):
		super().__init__()
		
		# self.phase_weight（可変位相パラメータ）
		# 各素子の「移相量 φ」を保持するレジスタです。
		# 初期状態では各素子がバラバラな位相を返しますが、
		# 学習（Tuning）によって、特定の信号に同期するように最適化されます。
		self.phase_weight = nn.Parameter(torch.rand(size, size) * 2 * torch.pi)

	def forward(self, u_in):
		"""
		信号通過処理：入力波形に複素平面上の回転を与えます。
		"""
		# u_in * exp(j * φ)
		# 各タップの信号を複素数として扱い、ベクトル回転（Phase Shift）を行います。
		# 電気回路でいうところの「インピーダンスの位相成分」を制御している状態です。
		u_out = u_in * torch.exp(1j * self.phase_weight)
		return u_out

class WaveSpace(nn.Module):
	"""
	【ステップ2：空間伝搬の伝達関数（Transfer Function）】
	
	イメージ：信号が空中（伝送路）を伝わる際の「位相特性」と「遅延特性」をシミュレートします。
	自由空間を一つの「巨大なフィルタ回路」として定義します。
	"""
	def __init__(self, size):
		super().__init__()
		
		# 周波数ドメインでの座標系を作成
		f = torch.fft.fftfreq(size)
		fy, fx = torch.meshgrid(f, f, indexing='ij')
		
		# h_kernel（伝達関数 H(f)）
		# 空間を伝搬する際の「群遅延」や「位相の回り込み」を数式化したものです。
		# 2次元のFIRフィルタの係数のような役割を果たします。
		h_kernel = torch.exp(-1j * torch.pi * (fx**2 + fy**2) * 4.0)
		self.register_buffer("h_kernel", h_kernel)

	def forward(self, u_in):
		"""
		空間フィルタリング処理：高速フーリエ変換(FFT)を用いて畳み込み演算を高速化します。
		"""
		# 1. FFT: 空間ドメインの信号を、スペクトル（周波数）成分に変換します。
		u_f = torch.fft.fftn(u_in, dim=(-2, -1))
		
		# 2. 伝達関数の乗算: 入力 X(f) にフィルタ H(f) を掛け、出力 Y(f) を得ます。
		# 信号処理の基本、Y(f) = X(f) * H(f) です。
		u_f_prop = u_f * self.h_kernel
		
		# 3. IFFT: 周波数成分から、再び空間的な信号（波形）に戻します。
		return torch.fft.ifftn(u_f_prop, dim=(-2, -1))

class SinCNN_Origin(nn.Module):
	"""
	【ステップ3：適応型フェーズドアレイ・システム（System Architecture）】
	
	各移相器を調整し、特定の入力パターンが来た時にだけ、
	特定の「受信用アンテナ（Focus Point）」で合成出力が最大化（同相合成）されるようにします。
	"""
	def __init__(self):
		super().__init__()
		self.layer = WaveCoreLayer(28) # 1段目の移相器アレイ
		self.space = WaveSpace(28)	 # 伝送路（空間）の特性
		
		# focus_points（受信タップの座標）
		# スクリーン上に配置した10個の「受信用プローブ」です。
		self.focus_points = [
			(4,4), (4,14), (4,24), (14,4), (14,14), 
			(14,24), (24,4), (24,14), (24,24), (12,12)
		]

	def forward(self, x):
		# 1. 輝度信号（実数）を、信号の振幅（複素数）に変換
		u = x.to(torch.complex64)
		
		# 2. 移相器を通り、空間フィルタ（伝送路）を通過
		u = self.layer(u)
		u = self.space(u)
		
		# 3. 信号電力（Power）の測定
		# P = |V|^2。複素電圧ベクトルの絶対値の2乗をとり、信号強度を算出します。
		intensity = torch.abs(u)**2
		
		# 4. 各プローブでの受信電力をリスト化
		results = []
		for (y, x_pos) in self.focus_points:
			# 特定座標のタップにおける電力値をサンプリング
			val = intensity[:, 0, y:y+1, x_pos:x_pos+1].mean(dim=(-1, -2))
			results.append(val)
		
		return torch.stack(results, dim=1)

def main():
	# --- データストリームの供給 ---
	loader = DataLoader(
		datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()),
		batch_size=16, shuffle=True
	)

	model = SinCNN_Origin()
	# optimizer: 最小二乗法のように、位相エラーを最小化するように移相器をチューニングします。
	optimizer = optim.Adam(model.parameters(), lr=0.01)
	# criterion: 受信電力の分布が正解のチャンネルに集中しているかを評価する評価関数です。
	criterion = nn.CrossEntropyLoss()

	print("SIN-CNN 001: 適応型フィルタの自動チューニングを開始します。")
	for epoch in range(1, 3):
		for i, (data, target) in enumerate(loader):
			if i > 100: break 
			
			optimizer.zero_grad()   # 蓄積された勾配（エラー成分）をリセット
			output = model(data)	# 信号を流し、プローブ出力を確認
			loss = criterion(output, target) # 目標値との位相・電力のズレを計算
			loss.backward()		 # 各移相器の調整方向（感度）を逆算
			optimizer.step()		# 移相量をアップデート！
			
			if i % 20 == 0:
				print(f"Epoch {epoch} Step {i}: Total Error = {loss.item():.4f}")

	print("チューニング完了。特定パターンに対する位相同期が確立されました。")


import torch
import torch.nn as nn
import torch.fft as fft

class WavePropagationLayer(nn.Module):
    """
    1層分の波の伝搬と干渉を行うレイヤー。
    学習対象は「行列」ではなく「物理パラメータ（周波数、位相、距離）」のみ。
    """
    def __init__(self, size=240, num_channels=8):
        super(WavePropagationLayer, self).__init__()
        self.size = size
        self.num_channels = num_channels
        
        # 学習パラメータ: 各チャンネルごとの特性
        # 自由な行列ではなく、数個の定数のみを学習させることで計算量とメモリを節約
        self.center_freq = nn.Parameter(torch.rand(num_channels) * 0.1)
        self.phase_shift = nn.Parameter(torch.rand(num_channels) * 2 * 3.1415)
        self.propagation_dist = nn.Parameter(torch.ones(num_channels) * 1.0)

    def forward(self, x):
        # x: [Batch, Channels, Height, Width] (複素数または実数)
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))

        # 1. 空間領域から周波数領域へ (2D FFT)
        X = fft.fft2(x)
        
        # 周波数グリッドの作成
        freq_y = fft.fftfreq(self.size).to(x.device)
        freq_x = fft.fftfreq(self.size).to(x.device)
        fy, fx = torch.meshgrid(freq_y, freq_x, indexing='ij')
        rho = torch.sqrt(fx**2 + fy**2) # 中心からの距離

        # 2. 関数ベースの伝達関数 (Transfer Function) の適用
        # 行列全体を学習する代わりに、Sinc関数やガウス関数でフィルタを生成
        # これにより「特定の周波数帯域」を抽出する物理的な挙動を模倣
        output_list = []
        for i in range(self.num_channels):
            # 位相遅延と帯域通過を組み合わせたフィルタ
            # exp(j * (位相 + 距離*周波数))
            h = torch.exp(1j * (self.phase_shift[i] + self.propagation_dist[i] * rho))
            # 窓関数（特定の周波数帯を通過させる）
            window = torch.exp(-((rho - self.center_freq[i])**2) / 0.01)
            
            # 周波数領域での要素ごとの積（これが畳み込みに相当するが非常に高速）
            output_list.append(X[:, i:i+1] * h * window)
            
        X_out = torch.cat(output_list, dim=1)

        # 3. 周波数領域から空間領域へ (2D IFFT)
        out = fft.ifft2(X_out)
        
        # 4. 非線形性の導入（検波）
        # 強度（絶対値の二乗）をとることで、次の層へ渡す「実数」の情報を作る
        return torch.abs(out)**2

class DeepWaveCNN(nn.Module):
    def __init__(self, size=240, num_layers=3, num_channels=16):
        super(DeepWaveCNN, self).__init__()
        # 多段化（レイヤーを重ねる）
        self.layers = nn.ModuleList([
            WavePropagationLayer(size, num_channels) for _ in range(num_layers)
        ])
        
        # 最後に特定の座標へ「集光（分類）」するための全結合層
        self.fc = nn.Linear(num_channels * size * size, 10) # 10クラス分類の例

    def forward(self, x):
        # 1チャンネルの入力をマルチチャンネルに拡張
        if x.shape[1] == 1:
            x = x.repeat(1, len(self.layers[0].center_freq), 1, 1)
            
        for layer in self.layers:
            x = layer(x)
            
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --- 使用例 ---
# model = DeepWaveCNN(size=240, num_layers=5, num_channels=16)
# 100層重ねても、学習パラメータは (3つ × 16ch × 100層) 程度で非常に軽量。







if __name__ == "__main__":
	main()

