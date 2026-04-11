import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- 1. 基本設定（CPUで動かすための設定） ---
torch.set_num_threads(4) # CPUの計算スレッド数を指定
IMG_SIZE = 28            # MNISTは28x28ピクセル

class SIN_Layer(nn.Module):
    """
    【干渉層】
    『ガラス板』に相当します。各ピクセルが「どれだけ光を遅らせるか（位相）」を学習します。
    """
    def __init__(self, size):
        super().__init__()
        # 学習パラメータ：0〜2π（一周期分）の乱数で初期化
        self.phase = nn.Parameter(torch.rand(size, size) * 2 * torch.pi)

    def forward(self, u_in):
        # 入力された波に「位相のズレ」を与える
        # 数式：u_out = u_in * e^(i * phase)
        return u_in * torch.exp(1j * self.phase)

class Propagation(nn.Module):
    """
    【回折層】
    光が空間を伝わって「拡散」する物理現象を計算します。
    """
    def __init__(self, size):
        super().__init__()
        # 伝搬の特性（Transfer Function）を事前に計算して固定
        f = torch.fft.fftfreq(size)
        fy, fx = torch.meshgrid(f, f, indexing='ij')
        # 簡易的な自由空間の伝搬式（位相の広がりを定義）
        dist_factor = torch.exp(-1j * torch.pi * (fx**2 + fy**2) * 5.0)
        self.register_buffer("h", dist_factor)

    def forward(self, u_in):
        # FFT（高速フーリエ変換）を使って周波数空間へ飛ばし、伝搬させて戻す
        u_f = torch.fft.fftn(u_in, dim=(-2, -1))
        u_f_propagated = u_f * self.h
        return torch.fft.ifftn(u_f_propagated, dim=(-2, -1))

class WaveModel(nn.Module):
    """
    【全体のネットワーク構造】
    [入力画像] -> [干渉層1] -> [拡散] -> [干渉層2] -> [拡散] -> [出力面（集光）]
    """
    def __init__(self):
        super().__init__()
        self.layer1 = SIN_Layer(IMG_SIZE)
        self.prop1 = Propagation(IMG_SIZE)
        self.layer2 = SIN_Layer(IMG_SIZE)
        self.prop2 = Propagation(IMG_SIZE)
        
        # 0〜9の数字に対応する「集光座標」を10箇所決める（画面上の固定位置）
        self.detect_pos = [
            (4,4), (4,14), (4,24), 
            (14,4), (14,14), (14,24),
            (24,4), (24,14), (24,24), (10,10)
        ]

    def forward(self, x):
        # 1. 入力画像を「波」の形にする (振幅=画像の値、位相=0)
        u = x.to(torch.complex64)
        
        # 2. 層を通過して光が干渉し、広がる
        u = self.prop1(self.layer1(u))
        u = self.prop2(self.layer2(u))
        
        # 3. 出力面での「光の強さ（振幅の2乗）」を計算
        intensity = torch.abs(u)**2
        
        # 4. 決めた10箇所の「明るさ」を取り出し、分類のスコアにする
        results = []
        for (y, x) in self.detect_pos:
            # 座標(y, x)周辺の明るさを集計
            val = intensity[:, 0, y:y+2, x:x+2].mean(dim=(-1, -2))
            results.append(val)
        return torch.stack(results, dim=1)

# --- 2. 実行用メインプログラム ---
def main():
    # データの読み込み（MNISTをダウンロードして準備）
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = WaveModel()
    optimizer = optim.Adam(model.parameters(), lr=0.05) # 学習率を少し高めに設定
    criterion = nn.CrossEntropyLoss()

    print("学習を開始します。しばらくお待ちください...")
    for epoch in range(1, 4): # まずは3周（エポック）だけ回してみる
        total_loss = 0
        for i, (data, target) in enumerate(loader):
            if i > 200: break # 時間短縮のため、1周の中で200バッチ分だけ学習
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if i % 50 == 0:
                print(f"エポック {epoch} [{i*32}/{len(dataset)}] 誤差: {loss.item():.4f}")

    print("\n--- テスト実行 ---")
    test_loader = DataLoader(dataset, batch_size=5, shuffle=True)
    images, labels = next(iter(test_loader))
    with torch.no_grad():
        results = model(images)
        predictions = results.argmax(dim=1)
        
        for i in range(5):
            print(f"画像 {i+1}: 予測した数字 = {predictions[i].item()}, 正解 = {labels[i].item()}")

if __name__ == "__main__":
    main()

