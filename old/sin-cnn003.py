import torch
import torch.nn as nn

# --- ここが重要！ ---
# matplotlibをインポートする「前」に、描画モードを 'Agg' (非表示モード) に設定します。
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
# ------------------

import numpy as np

# (前述のモデル定義などは省略)
class SinConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # 簡略化した例
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
    
    def forward(self, x):
        return torch.nn.functional.conv2d(x, self.weight, padding='same')

# 可視化用の関数
def save_visual_weight(weight_tensor, filename="output_pattern.png"):
    """
    重みテンソルを画像として保存する
    """
    # テンソルをnumpyに変換し、1チャンネルの画像データとして扱う
    # weight_tensor: (out_channels, in_channels, k, k) -> 取る1つ分
    data = weight_tensor[0, 0].detach().cpu().numpy()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap='viridis') # 色合いは科学計算でよく使われるviridis
    plt.colorbar(label='Intensity')
    plt.title("Visualizing Kernel Weight (First Channel)")
    
    # 画面表示ではなく、ファイルとして保存！
    plt.savefig(filename)
    print(f"✅ 画像を保存しました: {filename}")
    plt.close()

# 実行テスト
if __name__ == "__main__":
    # ダミーの重み（本来はモデルの重み）
    dummy_weight = torch.randn(1, 1, 5, 5)
    
    # 保存実行
    save_visual_weight(dummy_weight, "kernel_visualization.png")

