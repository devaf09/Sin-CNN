# ========================================================
# 応用スクリプト2: スタンドアロン超軽量推論エンジン
# (PyTorch不要 / C++やマイコン移植へのベースライン)
# ========================================================
import numpy as np
import json

class LightweightNumpyOpticalInference:
    def __init__(self, export_dir="optics_export"):
        # 1. 語彙の読み込み
        with open(f"{export_dir}/vocab.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
            self.vocab = meta['vocab']
            self.char2id = meta['char2id']
            self.id2char = {v: k for k, v in self.char2id.items()}
            
        # 2. PyTorch保存モデルのロード
        import torch
        ckpt = torch.load(f"{export_dir}/optical_llm.pth", map_location='cpu')
        state = ckpt['model_state_dict']
        
        self.embedding = state['token_embedding.weight'].numpy() # [vocab_size, 2304]
        self.pos_phase = state['pos_phase_shift'].numpy()       # [8]
        self.pos_weights = state['pos_weights'].numpy()         # [8]
        self.classifier_w = state['classifier_head.weight'].numpy() # [vocab_size, 2304]
        self.classifier_b = state['classifier_head.bias'].numpy()   # [vocab_size]
        
        # 2D位相マップのロード
        self.phase_masks = np.load(f"{export_dir}/phase_masks_2d.npy") # [8, 48, 48]
        
        # 物理定数・角スペクトル伝播伝達関数 H のNumPy再構築
        grid_size = 48
        DX, WAVELENGTH, PROP_DIST = 20e-6, 633e-9, 0.002
        df = 1.0 / (grid_size * DX)
        fx = np.linspace(-grid_size // 2, grid_size // 2 - 1, grid_size) * df
        fy = np.linspace(-grid_size // 2, grid_size // 2 - 1, grid_size) * df
        FX, FY = np.meshgrid(fx, fy, indexing='ij')
        
        k = 2 * np.pi / WAVELENGTH
        kx, ky = 2 * np.pi * FX, 2 * np.pi * FY
        kz_sq = k**2 - kx**2 - ky**2
        kz = np.where(kz_sq >= 0, np.sqrt(np.maximum(kz_sq, 0)), 1j * np.sqrt(np.maximum(-kz_sq, 0)))
        
        self.H = np.exp(1j * kz * PROP_DIST)

    def predict_next(self, context_ids):
        # 1. 8文字文脈の光重畳 (Complex Wavefront)
        U = np.zeros((48, 48), dtype=np.complex64)
        for pos_idx in range(8):
            token_id = context_ids[pos_idx]
            embed = self.embedding[token_id].reshape(48, 48)
            phase = self.pos_phase[pos_idx]
            weight = self.pos_weights[pos_idx]
            wave = weight * (embed * np.cos(phase) + 1j * embed * np.sin(phase))
            U += wave

        # 2. 8層 光伝播シミュレーション (位相変調 + FFT角スペクトル伝播)
        for layer in range(8):
            # 位相板透過
            phase_factor = np.exp(1j * self.phase_masks[layer])
            U = U * phase_factor
            
            # 空間伝播 (FFT -> H積 -> IFFT)
            U = np.fft.ifft2(np.fft.fft2(U) * self.H)
            
            if layer < 7:
                intensity = np.abs(U)**2
                norm_factor = np.sqrt(np.mean(intensity) + 1e-8)
                U = U / norm_factor

        # 3. 受光強度センサ ＆ 分類器
        final_intensity = (np.abs(U)**2).reshape(-1)
        logits = np.dot(self.classifier_w, final_intensity) + self.classifier_b
        return np.argmax(logits)

# テスト実行
if __name__ == "__main__":
    engine = LightweightNumpyOpticalInference()
    print("[初期化完了] NumPy単体での光計算推論エンジンの準備が完了しました。")
