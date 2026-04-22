import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- サーバーへの配慮 ---
# このサーバーには頭脳（CPU）が2つあるので、両方をしっかり使うように指示します。
torch.set_num_threads(2)

class WaveCoreLayer(nn.Module):
    """
    【ステップ1：魔法のガラス板（干渉層）】
    
    この層は、すりガラスのような「学習できるガラス板」です。
    光（データ）がこの板を通ると、場所によって「光が届くタイミング」がズレます。
    このズレが重なり合うことで、特定の場所に光が集まる「波紋」が生まれます。
    """
    def __init__(self, size):
        super().__init__()
        
        # self.phase_weight（位相の重み）
        # これが「ガラスの厚みのムラ」に相当します。
        # 0から2π（円一周分）のランダムな数字で埋められた28x28の板を作ります。
        # 学習が進むと、このムラが「数字を認識するための絶妙な形」に変化していきます。
        self.phase_weight = nn.Parameter(torch.rand(size, size) * 2 * torch.pi)

    def forward(self, u_in):
        """
        光がガラス板を通り抜ける瞬間の計算です。
        """
        # torch.exp(1j * self.phase_weight)
        # ここが最大のポイントです！
        # 「1j」は数学でいう虚数ですが、ここでは「波の回転（タイミング）」だと思ってください。
        # 光に「ガラスの厚み分の遅れ（位相）」を掛け合わせて、光を少しだけ曲げます。
        u_out = u_in * torch.exp(1j * self.phase_weight)
        return u_out

class WaveSpace(nn.Module):
    """
    【ステップ2：光が広がる空間（回折層）】
    
    ガラス板を通った光が、次の場所へ届くまでに「ボワーッ」と広がる現象を再現します。
    池に石を投げた時の波紋が広がっていくのと同じです。
    """
    def __init__(self, size):
        super().__init__()
        
        # 光がどう広がるかのルール（フィルタ）を事前に作っておきます。
        # f, fy, fxなどは、画面の端から端までの「距離」を測るための定規です。
        f = torch.fft.fftfreq(size)
        fy, fx = torch.meshgrid(f, f, indexing='ij')
        
        # h_kernel（伝搬関数）
        # 「中心から離れるほど、波がどう遅れて届くか」という物理法則を数式にしています。
        # これがあるおかげで、バラバラだった光が遠くで「干渉（合流）」できます。
        h_kernel = torch.exp(-1j * torch.pi * (fx**2 + fy**2) * 4.0)
        self.register_buffer("h_kernel", h_kernel)

    def forward(self, u_in):
        """
        光が空間を伝わっていく計算です。
        """
        # 1. torch.fft.fftn（高速フーリエ変換）
        # 光を「空間的な模様」から「波の細かさ（周波数）」の世界に翻訳します。
        # これを使うと、光の広がりを計算するのがめちゃくちゃ速くなります（ワープ航法のようなものです）。
        u_f = torch.fft.fftn(u_in, dim=(-2, -1))
        
        # 2. 伝搬ルールを掛け算
        u_f_prop = u_f * self.h_kernel
        
        # 3. torch.fft.ifftn（逆変換）
        # 周波数の世界から、もう一度「目に見える光の形」に戻します。
        # この時点で、光は空間を伝わってボヤけたり重なったりしています。
        return torch.fft.ifftn(u_f_prop, dim=(-2, -1))

class SinCNN_Origin(nn.Module):
    """
    【ステップ3：光の計算機（全体構造）】
    
    「ガラス板」と「空間」を組み合わせて、最終的にどこに光が集まるかを見守ります。
    """
    def __init__(self):
        super().__init__()
        self.layer = WaveCoreLayer(28) # 魔法のガラスを1枚セット
        self.space = WaveSpace(28)     # その後ろに広がる空間をセット
        
        # focus_points（集光ターゲット）
        # 出口のスクリーンに「10個のセンサー（座標）」を置きます。
        # 「0」という数字が入ってきたら、(4,4)のセンサーが一番光るように学習させます。
        self.focus_points = [
            (4,4), (4,14), (4,24), (14,4), (14,14), 
            (14,24), (24,4), (24,14), (24,24), (12,12)
        ]

    def forward(self, x):
        # 1. 入力された「数字の画像」を「光の波」に変えます。
        # 複素数（complex64）にすることで、波としての計算ができるようになります。
        u = x.to(torch.complex64)
        
        # 2. ガラスを通し、空間を伝わらせます。
        u = self.layer(u)
        u = self.space(u)
        
        # 3. intensity（光の強さ）を測ります。
        # 波はプラスやマイナスに揺れていますが、2乗（abs**2）することで
        # 私たちの目に見える「明るさ」というエネルギーに変換されます。
        intensity = torch.abs(u)**2
        
        # 4. 10個のセンサーがどれくらい光っているかをチェックします。
        results = []
        for (y, x_pos) in self.focus_points:
            # センサーが置かれた1ピクセル分の明るさを取り出します。
            val = intensity[:, 0, y:y+1, x_pos:x_pos+1].mean(dim=(-1, -2))
            results.append(val)
        
        # 10個のセンサーの「明るさリスト」を返します。
        return torch.stack(results, dim=1)

def main():
    # --- データの蛇口をひねる ---
    # サーバーのメモリ（2GB）が苦しくならないよう、
    # 16枚ずつゆっくり画像を読み込んで学習させます。
    loader = DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=16, shuffle=True
    )

    model = SinCNN_Origin()
    # optimizer（ネジ回し役）：誤差を見て、ガラス板のムラを少しずつ調整します。
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # criterion（答え合わせ役）：正解のセンサーが光っているか、厳しくチェックします。
    criterion = nn.CrossEntropyLoss()

    print("SIN-CNN 001: 物理シミュレーションによる学習を開始します。")
    for epoch in range(1, 3):
        for i, (data, target) in enumerate(loader):
            # 最初は時間がかかるので、100回計算したら一回区切ります。
            if i > 100: break 
            
            optimizer.zero_grad()   # 前回の計算の記憶をリセット
            output = model(data)    # 光を走らせて、センサーの明るさを測る
            loss = criterion(output, target) # 理想の明るさとどれだけズレているか計算
            loss.backward()         # ズレを解消するために、ガラスの厚みをどう変えるべきか逆算
            optimizer.step()        # 実際にガラスの厚み（位相）を調整！
            
            if i % 20 == 0:
                print(f"エポック {epoch} ステップ {i}: ズレ具合(Loss) = {loss.item():.4f}")

    print("実験完了。ガラス板に最初の『波紋』が刻まれました。")

if __name__ == "__main__":
    main()

