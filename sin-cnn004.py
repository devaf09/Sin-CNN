import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def main():
    # 1. ハイパーパラメータの設定
    batch_size = 64
    learning_rate = 0.001
    epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. データの準備 (MNIST)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # torchvisionのtransformsを使うために、ここでimportを補完
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 3. モデルの定義
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.dropout1 = torch.nn.Dropout(0.25)
            self.dropout2 = torch.nn.Dropout(0.5)
            self.fc1 = torch.nn.Linear(64 * 28 * 28, 128)
            self.fc2 = torch.nn.Linear(128, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = torch.relu(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            return x

    model = Net().to(device)

    # 4. 損失関数と最適化手法
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # 5. 学習ループ
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                      f"Loss: {loss.item():.6f}")
        
        print(f"Epoch {epoch} Summary: Average Loss: {running_loss/len(train_loader):.4f}")

    # 6. テスト（評価）
    print("\nTesting on test set...")
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f"\nTest Set: Average loss: {test_loss:.4f}, "
          f"Accuracy: {100. * correct / len(test_loader.dataset):.2f}%\n")

# 実行に必要なライブラリのインポート
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

if __name__ == '__main__':
    # 実行
    import torch.nn as nn
    # main関数の定義内で使うために、必要なものを使えるように整理
    from torch.utils.data import DataLoader
    
    # 再定義（コードの独立性を高めるため）
    def run_full_script():
        # 再度、必要なものを全て内部に含める形に整理
        import torch
        import torchvision
        import torchvision.transforms as transforms
        import torchvision.datasets as datasets
        from torch.utils.data import DataLoader
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ネットワーク定義
        class SimpleNet(torch.nn.Module):
            def __init__(self):
                super(SimpleNet, self).__init__()
                self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
                self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
                self.dropout1 = torch.nn.Dropout(0.25)
                self.fc1 = torch.nn.Linear(9216, 128)
                self.fc2 = torch.nn.Linear(128, 10)

            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = torch.max_pool2d(x, 2)
                x = self.dropout1(x)
                x = torch.flatten(x, 1)
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        model = SimpleNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        
        print(f"Starting training on {device}...")
        for epoch in range(1, 3):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch} [{batch_idx*64}/{len(train_set)}] Loss: {loss.item():.4f}")
        print("Training complete.")

    run_full_script()

