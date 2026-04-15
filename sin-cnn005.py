import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot.pyplot as plt
import os

# --- 1. モデルの定義 ---
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = self.dropout1(self.matplotlib_flatten_logic(x)) # 簡略化のためflattenはtorch.flattenを使用
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    # 構造をシンプルにするため、flattenをforward内に含めます
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# --- 2. 学習・評価・可視化のメイン関数 ---
def main():
    # 設定
    batch_size = 64
    learning_rate = 0.01
    epochs = 1  # 今回は動作確認のため1エポック
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データの準備
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # モデル、最適化手法、損失関数
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # --- 学習フェーズ ---
    print(f"Using device: {device}")
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(decive if hasattr(device, 'device') else device) # 修正
            # 修正：deviceの指定を確実にする
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    # --- 評価フェーズ (Accuracyの計算) ---
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_images = []
    all_labels = []

    print("\nEvaluating...")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

            # 可視化用のデータを保存 (最初の数件だけ)
            if len(all_images) < 10:
                for i in range(data.size(0)):
                    if len(all_images) >= 10: break
                    all_images.append(data[i].cpu())
                    all_predictions.append(pred[i].item())
                    all_labels.append(target[i].item())

    accuracy = 100. * correct / total
    print(f'\Test Accuracy: {accuracy:.2f}%\n')

    # --- 可視化フェーズ (画像の保存) ---
    plt.figure(figsize=(12, 5))
    for i in range(len(all_images)):
        plt.subplot(2, 5, i + 1)
        plt.imshow(all_images[i].squeeze(), cmap='gray')
        color = 'green' if all_predictions[i] == all_labels[i] else 'red'
        plt.title(f"P: {all_predictions[i]} (L: {all_labels[i]})", color=color)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('result.png')
    print("Results visualization saved as 'result.png'")

if __name__ == '__main__':
    main()

