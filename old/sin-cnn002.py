import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalConv2d(nn.Module):
    """
    Sinusoidal Convolution Layer
    
    このレイヤーは、フィルタを「空間的な重み」としてではなく、
    「特定の周波数(omega)と位相(phi)を持つ正弦波の集合」として定義します。
    
    各フィルタのパラメータは以下の複素振幅形式を想定しています:
    V = V_m * exp(j * phi)
    （V_m: 振幅, phi: 位相）
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SinusoidalConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # 1. Amplitude (V_m): 信号の強さ
        self.amplitude = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1) * 0.1)
        
        # 2. Phase (phi): 信号の初期位相
        self.phase = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1) * 0.1)
        
        # 3. Angular Frequency (omega): 空間周波数 (Spatial Frequency)
        # 空間的な波の細かさを決定する
        self.omega = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1) * 0.5)

    def forward(self, x):
        """
        Input x: [Batch, In_Channels, Height, Width]
        Output: [Batch, Out_Channels, Height, Width]
        """
        batch_size, _, h, w = x.shape
        
        # 座標グリッドの生成 (空間的な軸 t 相当)
        # y, x 座標を 0 から Kernel_Size までの時間軸(t)に見立てる
        yy, xx = torch.meshgrid(
            torch.linspace(0, self.kernel_size-1, h), 
            torch.linspace(0, self.kernel_size-1, w), 
            indexing='ij'
        )
        # 座標を [1, 1, H, W] の形状に拡張
        yy = yy.unsqueeze(0).unsqueeze(0).to(x.device)
        xx = xx.unsqueeze(0).unsqueeze(0).to(x.device)

        # --- フィルタの生成 (Signal Synthesis) ---
        # 信号モデル: f(t) = V_m * sin(omega * t + phi)
        # ここでは 2次元空間における波を生成
        # phase_shift = omega * (dist_from_center) + phi
        # 簡易化のため、中心からの距離に基づく波形を生成
        dist = torch.sqrt(yy**2 + xx**2) 
        
        # 複素的な振る舞いを実数部として抽出
        # Kernel_sin = V_m * sin(omega * dist + phi)
        kernel_sin = self.amplitude * torch.sin(self.omega * dist + self.phase)
        
        # フィルタの形状を [Out_C, In_C, H, W] に整形
        # (元の入力 x の空間解像度に合わせて生成)
        kernel = kernel_sin.expand(self.out_channels, self.in_channels, h, w)

        # 畳み込み演算 (Spatial Filtering)
        # ここでは、生成した正弦波カーネルを「重み」として使用
        # 入力信号 x に対して、特定の周波数成分を抽出するプロセス
        # 実際には 1x1 convolution 的な要素として作用
        out = F.conv2d(x, kernel.reshape(self.out_channels, self.in_channels, 1, 1), 
                       padding=0, groups=self.in_channels)
        
        return out

class ComplexImpedanceNet(nn.Module):
    """
    Impedance-based Feature Extractor
    
    入力画像から、特定の周波数成分（エッジやテクスチャ）の
    応答（振幅と位相）を抽出するネットワーク
    """
    def __init__(self, in_channels=1, out_channels=16):
        super(ComplexImpedanceNet, self).__init__()
        # 空間周波数フィルタ層
        self.conv1 = SinusoidalConv2d(in_channels, out_channels, kernel_size=3)
        # 特徴マップの整理
        self.fc = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(out_channels, 10) # 10クラス分類 (MNIST想定)

    def forward(self, x):
        # 入力 x: [Batch, 1, 28, 28]
        x = self.conv1(x)
        x = self.fc(x).view(x.size(0), -1)
        x = self.classifier(x)
        return x

# --- 動作検証 ---
if __name__ == "__main__":
    # ダミー入力 (Batch=4, Channel=1, 28x28)
    input_data = torch.randn(4, 1, 28, 28)
    model = ComplexImpedanceNet(in_channels=1, out_channels=16)
    
    output = model(input_data)
    
    print(f"Input Shape:  {input_data.shape}")
    print(f"Output Shape: {output.shape}") # [4, 10]
    print("Success: Signal extraction completed.")

