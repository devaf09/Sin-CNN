# ========================================================
# 応用スクリプト1: 位相マスクの可視化と高解像度ビットマップ変換
# (エッチング装置や空間光位相変調器 SLM への入力データ作成)
# ========================================================
import numpy as np
import matplotlib.pyplot as plt

def generate_slm_patterns():
    # 保存した位相マップをロード
    phase_masks = np.load("optics_export/phase_masks_2d.npy") # [8, 48, 48]
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle("8-Layer Diffractive Optical Elements (Phase Masks 0 to 2π)", fontsize=14)
    
    for layer in range(8):
        ax = axes[layer // 4, layer % 4]
        # 0 ~ 2π の位相分布を 8bit (0~255) グレースケール画像に変換
        # これは空間光変調器 (SLM) に直接転送できるパターンになります
        slm_image = (phase_masks[layer] / (2 * np.pi) * 255).astype(np.uint8)
        
        im = ax.imshow(slm_image, cmap='twilight', vmin=0, vmax=255)
        ax.set_title(f"Layer {layer + 1}")
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig("optics_export/phase_layers_vis.png")
    print("[完了] 全8層の位相分布画像を保存しました: optics_export/phase_layers_vis.png")

if __name__ == "__main__":
    generate_slm_patterns()

