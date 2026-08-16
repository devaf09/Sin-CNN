import numpy as np
import json
import torch

class LightweightNumpyOpticalInference:
    def __init__(self, export_dir="optics_export"):
        # 1. 語彙辞書のロード
        with open(f"{export_dir}/vocab.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
            self.vocab = meta['vocab']
            self.char2id = meta['char2id']
            self.id2char = {v: k for k, v in self.char2id.items()}
            
        # 2. モデルパラメータのロード
        ckpt = torch.load(f"{export_dir}/optical_llm.pth", map_location='cpu')
        state = ckpt['model_state_dict']
        
        self.embedding = state['token_embedding.weight'].numpy()      # [vocab_size, 2304]
        self.pos_phase = state['pos_phase_shift'].numpy()            # [8]
        self.pos_weights = state['pos_weights'].numpy()              # [8]
        self.classifier_w = state['classifier_head.weight'].numpy() # [vocab_size, 2304]
        self.classifier_b = state['classifier_head.bias'].numpy()   # [vocab_size]
        
        # 3. 事前計算された2D位相変調マップのロード
        self.phase_masks = np.load(f"{export_dir}/phase_masks_2d.npy") # [8, 48, 48]
        
        # 4. 角スペクトル伝播関数（H）の初期化
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
        # 8文字の光重畳
        U = np.zeros((48, 48), dtype=np.complex64)
        for pos_idx in range(8):
            token_id = context_ids[pos_idx]
            embed = self.embedding[token_id].reshape(48, 48)
            phase = self.pos_phase[pos_idx]
            weight = self.pos_weights[pos_idx]
            U += weight * (embed * np.cos(phase) + 1j * embed * np.sin(phase))

        # 8層の光学伝播
        for layer in range(8):
            U = U * np.exp(1j * self.phase_masks[layer])
            U = np.fft.ifft2(np.fft.fft2(U) * self.H)
            if layer < 7:
                U = U / np.sqrt(np.mean(np.abs(U)**2) + 1e-8)

        # 受光素子強度 ＆ 線形分類
        final_intensity = (np.abs(U)**2).reshape(-1)
        logits = np.dot(self.classifier_w, final_intensity) + self.classifier_b
        return np.argmax(logits)

    def generate_text(self, prompt, length=50):
        if len(prompt) < 8:
            prompt = prompt.rjust(8, self.vocab[0]) # 8文字に満たない場合は埋める
        
        current_context = [self.char2id[c] for c in prompt[-8:]]
        generated = list(prompt[-8:])
        
        for _ in range(length):
            next_id = self.predict_next(current_context)
            next_char = self.id2char[next_id]
            generated.append(next_char)
            current_context = current_context[1:] + [next_id]
            
        return "".join(generated)

if __name__ == "__main__":
    engine = LightweightNumpyOpticalInference()
    
    # 辞書に含まれる最初の8文字をプロンプトにする例
    sample_prompt = "".join(engine.vocab[:8])
    print(f"--- 推論開始 (プロンプト: {sample_prompt}) ---")
    result = engine.generate_text(sample_prompt, length=40)
    print("生成結果:", result)
