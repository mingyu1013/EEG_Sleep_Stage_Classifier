import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ——— Dataset ———
class MergedEEGDataset(Dataset):
    def __init__(self, npz_path, indices=None):
        data = np.load(npz_path)
        X = data['X_window'].astype(np.float32)
        y = data['y_stage'].astype(np.int64)
        if indices is not None:
            X = X[indices]
            y = y[indices]
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        x = (x - x.mean()) / (x.std() + 1e-6)
        x = torch.from_numpy(x)[None, :]
        y = torch.tensor(self.y[idx])
        return x, y

# ——— Model with BatchNorm & additional CNN layers ———
class CNN_LSTM(nn.Module):
    def __init__(self, hidden_size=256, num_classes=5, dropout_p=0.5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=2,
                            batch_first=True, dropout=dropout_p, bidirectional=True)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)            # (B,128,L/8)
        x = x.permute(0, 2, 1)     # (B,L/8,128)
        _, (h_n, _) = self.lstm(x)
        h = torch.cat((h_n[-2], h_n[-1]), dim=1)  # bidirectional
        h = self.dropout(h)
        return self.fc(h)

# ——— Training & Evaluation ———
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_true, all_pred = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_pred.extend(preds)
        all_true.extend(y.cpu().numpy())
    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average='macro')
    return total_loss / len(loader), acc, f1

def eval_model(model, loader, device):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_pred.extend(preds)
            all_true.extend(y.cpu().numpy())
    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average='macro')
    report = classification_report(all_true, all_pred, digits=4)
    cm = confusion_matrix(all_true, all_pred)
    return acc, f1, report, cm

# ——— Main ———
if __name__ == '__main__':
    npz_file = '/home/cymg0001/sleep/preprocessing_lstm/preprocessed_gpu.npz'
    save_dir = '/home/cymg0001/sleep/models'
    os.makedirs(save_dir, exist_ok=True)
    chkpt_path = os.path.join(save_dir, 'cnn_lstm_checkpoint.pth')
    best_path = os.path.join(save_dir, 'cnn_lstm_best.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Split dataset
    full = MergedEEGDataset(npz_file)
    N = len(full)
    idx = np.arange(N)
    np.random.shuffle(idx)
    n_train = int(0.8 * N)
    train_idx, val_idx = idx[:n_train], idx[n_train:]
    train_ds = MergedEEGDataset(npz_file, indices=train_idx)
    val_ds = MergedEEGDataset(npz_file, indices=val_idx)

    # Weighted sampler for train
    classes, counts = np.unique(train_ds.y, return_counts=True)
    cw = {c: len(train_ds) / count for c, count in zip(classes, counts)}
    sw = np.array([cw[int(y)] for y in train_ds.y])
    sampler = WeightedRandomSampler(sw, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=128, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4)

    # Model, loss, optimizer, scheduler
    model = CNN_LSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)

    # Early stopping
    best_f1 = 0.0
    patience = 5
    wait = 0

    for epoch in range(1, 51):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_f1, report, cm = eval_model(model, val_loader, device)
        scheduler.step(val_f1)
        print(f"Epoch {epoch:02d}: TLoss {train_loss:.4f} TAcc {train_acc:.4f} TF1 {train_f1:.4f} | "
              f"VAcc {val_acc:.4f} VF1 {val_f1:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, chkpt_path)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_path)
            wait = 0
            print("  > New best model saved")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break

    # Final evaluation
    print("\nBest Evaluation:")
    model.load_state_dict(torch.load(best_path))
    _, _, report, cm = eval_model(model, val_loader, device)
    print(report)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='Blues')
    plt.title(f"Confusion Matrix (F1={best_f1:.4f})")
    plt.colorbar()
    plt.xlabel('Pred')
    plt.ylabel('True')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', color='white')
    plt.tight_layout()
    plt.show()

