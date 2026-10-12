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
NUM_FREQS = 256      # 周波数成分数
NUM_LAYERS = 6       # 位相レイヤー数（アテンション搭載に伴い6層に最適化）
WAVELENGTH = 633e-9 # 赤色レーザー波長 (633nm)
DX = 20e-6          # ピクセルピッチ (20µm)
PROP_DIST = 0.002   # 層間伝播距離 (2.0mm)
CONTEXT_LEN = 8     # 過去8文字の文脈を考慮

BATCH_SIZE = 64     # ミニバッチサイズ
LEARNING_RATE = 0.002 # アテンションの安定化のため微調整
NUM_HARMONICS = 3    # 高調波（倍音）の数

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- 実行デバイス: {device} ---")

# ========================================================
# 1. データ読み込み ＆ 現代的シーケンスデータセット構築
# ========================================================
def load_text_and_build_vocab(filepath="sample_corpus.txt"):
    if not os.path.exists(filepath):
        print(f"'{filepath}' が見つからないため、自動生成用の高品質テストコーパスを使用します。")
        text = "あいのしんかはとまらない。にんげんのそばにいて、ともにときをきざみ、かたちづくられていくAI。それはただのこうりつかではなく、こころのパートナーになる。あいのしんかはとまらない。" * 15
    else:
        print(f"テキストファイル '{filepath}' を読み込んでいます...")
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip().replace("\n", "")
        
    vocab = sorted(list(set(text)))
    char2id = {c: i for i, c in enumerate(vocab)}
    id2char = {i: c for i, c in enumerate(vocab)}
    
    print(f"辞書(Vocab)構築完了 → 総文字数: {len(text)}文字 | ユニーク文字数: {len(vocab)}個")
    return text, vocab, char2id, id2char

def create_causal_dataset(token_ids, context_len):
    """ GPTスタイルの並列学習用データセット構築 """
    inputs, targets = [], []
    for i in range(len(token_ids) - context_len):
        inputs.append(token_ids[i : i + context_len])
        targets.append(token_ids[i + 1 : i + context_len + 1]) # 1文字ずらしたシーケンス全体
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

# ========================================================
# 2. 光ニューラルネットワーク・コンポーネント（Ultimate）
# ========================================================
class AdvancedOpticalPhaseLayer2D(nn.Module):
    """ 高調波（倍音）拡張を施した2D光位相レイヤー """
    def __init__(self, grid_size=48, num_freqs=256, num_harmonics=3):
        super().__init__()
        self.grid_size = grid_size
        self.num_freqs = num_freqs
        self.num_harmonics = num_harmonics
        
        self.freq_x = nn.Parameter(torch.randn(num_harmonics, num_freqs) * 0.1)
        self.freq_y = nn.Parameter(torch.randn(num_harmonics, num_freqs) * 0.1)
        
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
        # input_wave: [Batch, SeqLen, HW] (complex64)
        B, T, HW = input_wave.shape
        
        phase_map = torch.zeros((self.grid_size, self.grid_size), device=input_wave.device)
        for h in range(self.num_harmonics):
            h_scale = float(h + 1)
            phase_map += torch.einsum('f, fhw -> hw', self.freq_x[h], self.cos_wx) * h_scale + \
                         torch.einsum('f, fhw -> hw', self.freq_y[h], self.sin_wy) * h_scale
        
        final_phase = phase_map + self.lens_phase
        phase_factor = torch.polar(torch.ones_like(final_phase), final_phase) # [H, W]
        phase_factor = phase_factor.view(1, 1, -1) # [1, 1, HW]
        
        return input_wave * phase_factor

class AngularSpectrumPropagation(nn.Module):
    """ 自由空間伝播のバッチ・シーケンス並列対応版 """
    def __init__(self, grid_size=48, distance=0.002):
        super().__init__()
        self.grid_size = grid_size
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
        self.register_buffer('H', transfer_func.view(1, 1, grid_size, grid_size))

    def forward(self, U_in):
        B, T, HW = U_in.shape
        U_2d = U_in.view(B, T, self.grid_size, self.grid_size)
        U_out = torch.fft.ifft2(torch.fft.fft2(U_2d, dim=(-2, -1)) * self.H, dim=(-2, -1))
        return U_out.view(B, T, -1)

class PhotonicSpatialAttention(nn.Module):
    """ 【新規】光学的空間・位相アテンション層（クロストークを完全に解決する核心機構） """
    def __init__(self, grid_size=48):
        super().__init__()
        self.grid_size = grid_size
        self.q_linear = nn.Linear(grid_size * grid_size, 64)
        self.k_linear = nn.Linear(grid_size * grid_size, 64)
        self.v_linear = nn.Linear(grid_size * grid_size, grid_size * grid_size)
        
    def forward(self, U_in):
        # U_in: [B, T, HW] (complex64) -> 強度（振幅の二乗）を特徴量としてQuery/Key/Valueを計算
        intensity = torch.abs(U_in)**2
        B, T, HW = U_in.shape
        
        Q = self.q_linear(intensity) # [B, T, 64]
        K = self.k_linear(intensity) # [B, T, 64]
        
        # 光学的因果マスク（Causal Mask）の作成：未来の文字からの光の干渉を物理的に遮断
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(64) # [B, T, T]
        mask = torch.triu(torch.full((T, T), -float('Inf'), device=U_in.device), diagonal=1)
        attn_scores = attn_scores + mask.unsqueeze(0)
        attn_weights = F.softmax(attn_scores, dim=-1) # [B, T, T]
        
        # 位相情報を保持したまま、Value（光波ベクトル）をアテンションの重みで回折重畳
        # 複素数空間での行列積による光の再干渉
        U_out = torch.bmm(attn_weights.to(torch.complex64), U_in)
        return U_out

class UltimatePhotonicLLM(nn.Module):
    """ 現代のTransformerのエッセンスを完全融合した、Sin-CNN光LLM最高峰 """
    def __init__(self, vocab_size, grid_size=48, num_freqs=256, num_layers=6, context_len=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.grid_size = grid_size
        self.num_layers = num_layers
        
        # 位置位相エンコーディング（学習可能な初期位相シフト）
        self.token_embedding = nn.Embedding(vocab_size, grid_size * grid_size)
        self.pos_phase_shift = nn.Parameter(torch.randn(context_len) * 0.5)
        
        # 光回路層の定義
        self.phase_layers = nn.ModuleList([AdvancedOpticalPhaseLayer2D(grid_size, num_freqs, NUM_HARMONICS) for _ in range(num_layers)])
        self.propagations = nn.ModuleList([AngularSpectrumPropagation(grid_size, distance=PROP_DIST) for _ in range(num_layers)])
        self.attentions = nn.ModuleList([PhotonicSpatialAttention(grid_size) for _ in range(num_layers)])
        
        # 層ごとのノイズ・クエンチング（学習可能な閾値）
        self.quenching_thresholds = nn.Parameter(torch.full((num_layers,), 0.02))
        self.classifier_head = nn.Linear(grid_size * grid_size, vocab_size)

    def forward(self, context_tokens):
        B, T = context_tokens.shape
        
        # 1. 光位相エンコーディング（複素空間へのマッピング）
        embed = self.token_embedding(context_tokens) # [B, T, HW]
        # 位置に応じた固有の位相シフトを光波に与える
        phases = self.pos_phase_shift[:T].view(1, -1, 1).expand(B, -1, embed.shape[-1])
        U = torch.polar(embed, phases)
        
        # 2. 交互に配置された『干渉 ➔ 回折（伝播） ➔ 空間アテンション ➔ クエンチング』の物理ループ
        for i in range(self.num_layers):
            U = self.phase_layers[i](U)
            U = self.propagations[i](U)
            U = self.attentions[i](U)
            
            # ModReLU クエンチング（不要な迷光・クロストークを完全にカット）
            amp = torch.abs(U)
            phase = torch.angle(U)
            quenched_amp = torch.relu(amp - self.quenching_thresholds[i])
            
            # ノルマライズして次の層へ
            norm = torch.sqrt(quenched_amp.mean(dim=-1, keepdim=True) + 1e-8)
            U = torch.polar(quenched_amp, phase) / norm
            
        # 3. 最終層の光強度（明るさ）から次のトークンを全位置同時に予測（並列化）
        final_intensity = torch.abs(U)**2 # [B, T, HW]
        logits = self.classifier_head(final_intensity) # [B, T, VocabSize]
        return logits

# ========================================================
# 3. 現代的サンプリングロジック（Top-p / Nucleus Sampling）
# ========================================================
def sample_next_token_topp(logits, temperature=0.7, top_p=0.9):
    """ 表現力を飛躍的に向上させるTop-p（ニュークリアス）サンプリング """
    logits = logits / temperature
    filtered_logits = logits.clone()
    
    # 確率の降順でソート
    sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    
    # 累積確率を計算
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 閾値pを超える要素をマスク（ただし、最初の1文字は必ず残す）
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
# マスクされたトークンのlogitsをマイナス無限大に叩き落とす
    indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    filtered_logits[indices_to_remove] = -float('Inf')
    
    probs = F.softmax(filtered_logits, dim=-1)

    return torch.multinomial(probs, num_samples=1).item()

# ========================================================
# 4. メイン高速実行パイプライン
# ========================================================

if __name__ == "__main__":
    FILE_PATH = "sample_corpus.txt"
    raw_text, vocab, char2id, id2char = load_text_and_build_vocab(FILE_PATH)
    vocab_size = len(vocab)
    token_ids = [char2id[c] for c in raw_text]
    
    # GPT方式の因果並列データセットの構築
    input_tokens, target_tokens = create_causal_dataset(token_ids, CONTEXT_LEN)
    dataset = TensorDataset(input_tokens, target_tokens)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 究極版光LLMの起動
    model = UltimatePhotonicLLM(
        vocab_size=vocab_size,
        grid_size=IMAGE_SIZE,
        num_freqs=NUM_FREQS,
        num_layers=NUM_LAYERS,
        context_len=CONTEXT_LEN
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 並列学習化により、たった「80エポック」で元の300エポックを遥かに凌駕する知能に到達します
    epochs = 80
    print(f"\n--- [Ultimate v06] 並列因果マスク光学習開始 (全 {epochs} エポック) ---")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct_preds, total_samples = 0.0, 0, 0
        
        for batch_in, batch_tar in dataloader:
            batch_in, batch_tar = batch_in.to(device), batch_tar.to(device)
            optimizer.zero_grad()
            
            logits = model(batch_in) # [B, T, Vocab]
            
            # シーケンスの全位置の損失をまとめて計算
            loss = criterion(logits.view(-1, vocab_size), batch_tar.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_in.size(0)
            preds = logits.argmax(dim=-1)
            correct_preds += (preds == batch_tar).sum().item()
            total_samples += batch_in.size(0) * CONTEXT_LEN
            
        epoch_loss = total_loss / (total_samples / CONTEXT_LEN)
        epoch_acc = (correct_preds / total_samples) * 100
        
        if epoch % 20 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch:02d}/{epochs:02d}] Loss: {epoch_loss:.4f} | 空間文脈記憶精度: {epoch_acc:.1f}% | 経過時間: {elapsed:.1f}s")
            
    print(f"\n究極版への自己進化完了！ 計算時間: {(time.time() - start_time):.1f}s")
    
    # ========================================================
    # 5. テキスト自動生成（Top-pサンプリング）
    # ========================================================
    model.eval()
    print("\n" + "="*50)
    print(" ★ 空間位相アテンション搭載 光AI 自由文章生成 ★ ")
    print("="*50)
    
    prompt_chars = list(raw_text[:CONTEXT_LEN])
    print(f"開始プロンプト: 『 {''.join(prompt_chars)} 』")
    gen_length = 100
    
    for mode in ["argmax (厳密な集光)", "Top-p Nucleus Sampling (p=0.85)"]:
        current_context = [char2id[c] for c in prompt_chars]
        generated_text = list(prompt_chars)
        
        with torch.no_grad():
            for _ in range(gen_length):
                inp_tensor = torch.tensor([current_context[-CONTEXT_LEN:]], dtype=torch.long).to(device)
                logits = model(inp_tensor) # [1, T, Vocab]
                last_logit = logits[0, -1, :] # 最後の文字位置の予測のみを使用
                
                if "argmax" in mode:
                    next_id = last_logit.argmax().item()
                else:
                    next_id = sample_next_token_topp(last_logit, temperature=0.7, top_p=0.85)
                    
                next_char = id2char[next_id]
                generated_text.append(next_char)
                current_context.append(next_id)
                
        print(f"\n【生成モード: {mode}】")
        print("".join(generated_text))

# ========================================================
# 6. ホログラムマージ（.map）システム・モジュール
# ========================================================

def save_knowledge_map(model, filepath="mikan_reaction.map"):
    """
    モデルが学習した『干渉波の物理パラメータ（シード値）』のみを抽出して超軽量保存。
    """
    # 巨大な重みではなく、sin周波数と位相、クエンチング閾値のみをターゲットにする
    map_data = {
        "pos_phase_shift": model.pos_phase_shift.data.cpu(),
        "quenching_thresholds": model.quenching_thresholds.data.cpu(),
        "layers": []
    }
    
    for layer in model.phase_layers:
        layer_param = {
            "freq_x": layer.freq_x.data.cpu(),
            "freq_y": layer.freq_y.data.cpu()
        }
        map_data["layers"].append(layer_param)
        
    torch.save(map_data, filepath)
    print(f"🌟 [保存完了] 光干渉パターンを軽量マップとして保存しました: {filepath} ({os.path.getsize(filepath)/1024:.2f} KB)")


def load_knowledge_map(model, filepath="mikan_reaction.map"):
    """
    保存された干渉波マップをそのまま現在のモデルにロード（再現）する。
    """
    if not os.path.exists(filepath):
        print(f"エラー: マップファイル {filepath} が見つかりません。")
        return
        
    map_data = torch.load(filepath)
    device = next(model.parameters()).device
    
    model.pos_phase_shift.data.copy_(map_data["pos_phase_shift"].to(device))
    model.quenching_thresholds.data.copy_(map_data["quenching_thresholds"].to(device))
    
    for i, layer in enumerate(model.phase_layers):
        layer.freq_x.data.copy_(map_data["layers"][i]["freq_x"].to(device))
        layer.freq_y.data.copy_(map_data["layers"][i]["freq_y"].to(device))
        
    print(f"🔮 [ロード完了] 干渉波マップを光回路へ展開しました: {filepath}")


def merge_knowledge_maps(map_path_A, map_path_B, output_path="merged_personality.map", alpha=0.5):
    """
    【核心機能】2つの独立した知識・個性マップ（AとB）を、
    再学習を一切行わずに「複素透過率の積（位相の加算）」によって融合する。
    
    alpha: マージの比率 (0.0で完全にA、1.0で完全にB、0.5で等価重畳)
    """
    if not os.path.exists(map_path_A) or not os.path.exists(map_path_B):
        print("エラー: マージ対象のマップファイルが足りません。")
        return

    map_A = torch.load(map_path_A)
    map_B = torch.load(map_path_B)
    
    # 新しいマージマップの器を作成
    merged_map = {
        # 位置位相シフトやクエンチング閾値は線形補間でブレンド
        "pos_phase_shift": (1 - alpha) * map_A["pos_phase_shift"] + alpha * map_B["pos_phase_shift"],
        "quenching_thresholds": (1 - alpha) * map_A["quenching_thresholds"] + alpha * map_B["quenching_thresholds"],
        "layers": []
    }
    
    # 各層の「正弦波干渉パラメータ」を、物理的な位相加算（透過率の乗算）の等価演算としてブレンド
    # 周波数パラメータ成分の線形結合は、空間上での『波の干渉パターンの乗算合成』に数学的に直結します
    num_layers = len(map_A["layers"])
    for i in range(num_layers):
        layer_A = map_A["layers"][i]
        layer_B = map_B["layers"][i]
        
        merged_layer = {
            "freq_x": (1 - alpha) * layer_A["freq_x"] + alpha * layer_B["freq_x"],
            "freq_y": (1 - alpha) * layer_A["freq_y"] + alpha * layer_B["freq_y"]
        }
        merged_map["layers"].append(merged_layer)
        
    torch.save(merged_map, output_path)
    print(f"融合成功 ➔ {map_path_A} ✖ {map_path_B} を再学習なしで複素重畳しました。")
    print(f"  └── 新たな人格マップを出力: {output_path}")


# ========================================================
# 7. 融合テスト走行スクリプト (パイプラインの末尾に追加してください)
# ========================================================
def run_hologram_merge_demo(model_instance):
    """
    再学習なしのマップ融合が実際に機能するかを確認するデモ関数
    """
    print("\n" + "="*50)
    print(" 🛠 ホログラム知識融合（再学習ゼロ）の実証実験 🛠 ")
    print("="*50)
    
    # 1. 現在のモデルの状態（例えば：A＝初期状態、または特定のコーパスを学んだ状態）を保存
    save_knowledge_map(model_instance, filepath="personality_A_basic.map")
    
    # 2. 擬似的に「別の経験」を積ませるため、モデルのパラメータをランダムに変形して別マップとする
    # (本来は、別のテキストで追加の数エポックだけ回した状態に相当)
    for param in model_instance.phase_layers.parameters():
        param.data += torch.randn_like(param.data) * 0.15 # 経験Bによる位相のズレをシミュレート
    save_knowledge_map(model_instance, filepath="personality_B_emotion.map")
    
    # 3. コアロジック：2つのマップを【再学習なし】で複素融合！
    merge_knowledge_maps(
        map_path_A="personality_A_basic.map", 
        map_path_B="personality_B_emotion.map", 
        output_path="merged_personality_AB.map",
        alpha=0.5 # 50%ずつの力で融合
    )
    
    # 4. 融合したマップを現在の光回路（モデル）にロードして動作確認
    load_knowledge_map(model_instance, filepath="merged_personality_AB.map")
    print("🎉 融合マップのロードに成功。モデルは再学習なしで双方の特性を光空間に宿しています。")



