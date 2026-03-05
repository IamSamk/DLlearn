"""
RNN and LSTM Models for Cat vs Dog Classification
===================================================
(a) Custom RNN and LSTM architectures
(b) Variants: SimpleRNN, GRU, LSTM, Bidirectional LSTM
(c) Performance comparison (accuracy, loss)
(d) Visualization of training metrics
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import kagglehub

# ──────────────────────────────────────────────
# 1. DOWNLOAD & PREPARE DATASET
# ──────────────────────────────────────────────

print("Downloading dataset...")
path = kagglehub.dataset_download("bhavikjikadara/dog-and-cat-classification-dataset")
print("Path to dataset files:", path)

# --- walk directory to find images -----------------------------------------
cat_images, dog_images = [], []
for root, dirs, files in os.walk(path):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            full = os.path.join(root, f)
            lower_path = full.lower()
            if 'cat' in lower_path:
                cat_images.append(full)
            elif 'dog' in lower_path:
                dog_images.append(full)

print(f"Found {len(cat_images)} cat images, {len(dog_images)} dog images")

# --- take a SMALL sample for fast experimentation -------------------------
SAMPLE_PER_CLASS = 500          # adjust as needed
np.random.seed(42)
cat_sample = list(np.random.choice(cat_images, min(SAMPLE_PER_CLASS, len(cat_images)), replace=False))
dog_sample = list(np.random.choice(dog_images, min(SAMPLE_PER_CLASS, len(dog_images)), replace=False))

all_paths  = cat_sample + dog_sample
all_labels = [0] * len(cat_sample) + [1] * len(dog_sample)   # 0 = cat, 1 = dog
print(f"Using {len(all_paths)} images ({len(cat_sample)} cats, {len(dog_sample)} dogs)")

# ──────────────────────────────────────────────
# 2. DATASET & DATALOADER
# ──────────────────────────────────────────────

IMG_SIZE = 64   # resize to 64×64 → treat 64 rows as 64 time‑steps, 64*3 features

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),            # → (3, 64, 64) float [0,1]
])

class CatDogDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # Reshape for RNN: (C, H, W) → (H, C*W)  i.e. (seq_len, input_size)
        # Each row of the image becomes one time‑step
        c, h, w = img.shape
        img_seq = img.permute(1, 0, 2).reshape(h, c * w)   # (64, 192)
        label = self.labels[idx]
        return img_seq, label

dataset = CatDogDataset(all_paths, all_labels, transform=transform)

# 80 / 20 train‑test split
train_size = int(0.8 * len(dataset))
test_size  = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size],
                                  generator=torch.Generator().manual_seed(42))

BATCH_SIZE = 32
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

# ──────────────────────────────────────────────
# 3. MODEL DEFINITIONS  (a) Custom Architectures
# ──────────────────────────────────────────────

INPUT_SIZE  = IMG_SIZE * 3   # 192  (each row: 64 pixels × 3 channels)
SEQ_LEN     = IMG_SIZE       # 64   (number of rows)
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
NUM_CLASSES = 2
DROPOUT     = 0.3

# ---- (i) Simple RNN -------------------------------------------------------
class SimpleRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                          num_layers=NUM_LAYERS, batch_first=True,
                          dropout=DROPOUT, nonlinearity='tanh')
        self.fc = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.rnn(x)          # (batch, seq, hidden)
        out = out[:, -1, :]           # last time‑step
        return self.fc(out)

# ---- (ii) GRU (RNN variant) -----------------------------------------------
class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                          num_layers=NUM_LAYERS, batch_first=True,
                          dropout=DROPOUT)
        self.fc = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

# ---- (iii) LSTM ------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                            num_layers=NUM_LAYERS, batch_first=True,
                            dropout=DROPOUT)
        self.fc = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# ---- (iv) Bidirectional LSTM (LSTM variant) --------------------------------
class BiLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                            num_layers=NUM_LAYERS, batch_first=True,
                            dropout=DROPOUT, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE * 2, 64),   # *2 for bidirectional
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]          # (batch, hidden*2)
        return self.fc(out)

# ──────────────────────────────────────────────
# 4. TRAINING & EVALUATION HELPERS
# ──────────────────────────────────────────────

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def train_model(model, train_loader, test_loader, epochs=15, lr=1e-3):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(1, epochs + 1):
        # --- training ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X.size(0)
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total

        # --- evaluation ---
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                running_loss += loss.item() * X.size(0)
                correct += (outputs.argmax(1) == y).sum().item()
                total += y.size(0)

        test_loss = running_loss / total
        test_acc  = correct / total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f"  Epoch {epoch:02d}/{epochs}  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  |  "
              f"Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.4f}")

    return history


def evaluate_model(model, loader):
    """Return predictions and ground truths for classification report."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            preds = model(X).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    return np.array(all_preds), np.array(all_labels)

# ──────────────────────────────────────────────
# 5. TRAIN ALL MODELS   (b) Experiment with variants
# ──────────────────────────────────────────────

EPOCHS = 15

models_dict = {
    'SimpleRNN': SimpleRNNModel(),
    'GRU':       GRUModel(),
    'LSTM':      LSTMModel(),
    'BiLSTM':    BiLSTMModel(),
}

all_histories = {}

for name, model in models_dict.items():
    print(f"\n{'='*60}")
    print(f" Training {name}")
    print(f"{'='*60}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f" Total parameters: {total_params:,}")
    history = train_model(model, train_loader, test_loader, epochs=EPOCHS)
    all_histories[name] = history

# ──────────────────────────────────────────────
# 6. ANALYSIS & COMPARISON   (c)
# ──────────────────────────────────────────────

print("\n" + "=" * 70)
print(" FINAL RESULTS COMPARISON")
print("=" * 70)

summary_rows = []
for name, model in models_dict.items():
    model.to(device)
    preds, labels = evaluate_model(model, test_loader)
    acc = (preds == labels).mean()
    summary_rows.append((name, acc, all_histories[name]['test_loss'][-1]))
    print(f"\n--- {name} ---")
    print(f"  Test Accuracy : {acc:.4f}")
    print(classification_report(labels, preds, target_names=['Cat', 'Dog']))

# comparison table
print("\n" + "-" * 50)
print(f"{'Model':<15} {'Test Acc':>10} {'Test Loss':>10}")
print("-" * 50)
for name, acc, loss in summary_rows:
    print(f"{name:<15} {acc:>10.4f} {loss:>10.4f}")
print("-" * 50)

# ──────────────────────────────────────────────
# 7. PLOTS   (d) Visualize loss & accuracy
# ──────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# (i) Training Loss
ax = axes[0][0]
for name, h in all_histories.items():
    ax.plot(range(1, EPOCHS+1), h['train_loss'], label=name)
ax.set_title('Training Loss', fontsize=14)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.legend(); ax.grid(True, alpha=0.3)

# (ii) Test Loss
ax = axes[0][1]
for name, h in all_histories.items():
    ax.plot(range(1, EPOCHS+1), h['test_loss'], label=name)
ax.set_title('Test Loss', fontsize=14)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.legend(); ax.grid(True, alpha=0.3)

# (iii) Training Accuracy
ax = axes[1][0]
for name, h in all_histories.items():
    ax.plot(range(1, EPOCHS+1), h['train_acc'], label=name)
ax.set_title('Training Accuracy', fontsize=14)
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
ax.legend(); ax.grid(True, alpha=0.3)

# (iv) Test Accuracy
ax = axes[1][1]
for name, h in all_histories.items():
    ax.plot(range(1, EPOCHS+1), h['test_acc'], label=name)
ax.set_title('Test Accuracy', fontsize=14)
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle('RNN & LSTM Variants – Cat vs Dog Classification', fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__) if '__file__' in dir() else '.', 'training_curves.png'), dpi=150, bbox_inches='tight')
plt.show()
print("\n[Saved] training_curves.png")

# ── Bar chart: final test accuracy comparison ──
fig2, ax2 = plt.subplots(figsize=(8, 5))
names   = [r[0] for r in summary_rows]
accs    = [r[1] for r in summary_rows]
colors  = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
bars = ax2.bar(names, accs, color=colors, edgecolor='black')
for bar, a in zip(bars, accs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{a:.2%}', ha='center', va='bottom', fontweight='bold')
ax2.set_ylim(0, 1.05)
ax2.set_ylabel('Test Accuracy')
ax2.set_title('Model Comparison – Test Accuracy', fontsize=14)
ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__) if '__file__' in dir() else '.', 'accuracy_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] accuracy_comparison.png")

# ── Confusion matrices ──
fig3, axes3 = plt.subplots(1, 4, figsize=(20, 4))
for idx, (name, model) in enumerate(models_dict.items()):
    model.to(device)
    preds, labels = evaluate_model(model, test_loader)
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'],
                ax=axes3[idx])
    axes3[idx].set_title(name)
    axes3[idx].set_xlabel('Predicted')
    axes3[idx].set_ylabel('Actual')
plt.suptitle('Confusion Matrices', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__) if '__file__' in dir() else '.', 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.show()
print("[Saved] confusion_matrices.png")

print("\n✅ All done!")
