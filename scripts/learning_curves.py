import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from braindecode.models import EEGNetv4, ShallowFBCSPNet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

DATA_DIR = "processed_data_2b"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

d = np.load(f"{DATA_DIR}/within_subject_4.npz")
X_train = d["X_train"] * 1e6
y_train = d["y_train"]
X_test = d["X_test"] * 1e6
y_test = d["y_test"]
n_times = X_train.shape[2]

def make_loader(X, y, batch_size=32, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X_train, y_train)
test_loader = make_loader(X_test, y_test, shuffle=False)

class TinyEEGTransformer(nn.Module):
    def __init__(self, n_chans, n_times, d_model=32, n_heads=4, n_layers=2, n_classes=2):
        super().__init__()
        self.input_proj = nn.Linear(n_chans, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, n_times, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=64,
            dropout=0.2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_proj(x) + self.pos_embedding
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)

def train_with_curve(model, epochs=30):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    train_losses, test_accs = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(DEVICE)
                out = model(xb)
                preds.extend(out.argmax(dim=1).cpu().numpy())
                trues.extend(yb.numpy())
        test_accs.append(accuracy_score(trues, preds))
        print(f"  {epoch+1}/30 done", flush=True)

    return train_losses, test_accs

models = {
    "EEGNet": EEGNetv4(n_chans=3, n_outputs=2, n_times=n_times).to(DEVICE),
    "ShallowConvNet": ShallowFBCSPNet(n_chans=3, n_outputs=2, n_times=n_times).to(DEVICE),
    "Transformer": TinyEEGTransformer(3, n_times).to(DEVICE),
}

curves = {}
for name, model in models.items():
    print(f"Training {name}...", flush=True)
    losses, accs = train_with_curve(model)
    curves[name] = {"loss": losses, "acc": accs}
    print(f"{name} done. Final test acc: {accs[-1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = {"EEGNet": "#2196F3", "ShallowConvNet": "#4CAF50", "Transformer": "#FF5722"}

for name, c in curves.items():
    axes[0].plot(c["loss"], label=name, color=colors[name], linewidth=2)
    axes[1].plot(c["acc"], label=name, color=colors[name], linewidth=2)

axes[0].set_title("Training Loss — Subject 4 (within-subject)", fontsize=13)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title("Test Accuracy — Subject 4 (within-subject)", fontsize=13)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].axhline(0.857, color='gray', linestyle='--', linewidth=1, label="CSP+LDA baseline")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("learning_curves.png", dpi=150)
print("Saved learning_curves.png")
