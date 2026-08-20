import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import time

DATA_DIR = "processed_data_2b"
SUBJECTS = list(range(1, 10))
N_CHANS = 3
N_CLASSES = 2
DEVICE = torch.device("cpu")  # forced CPU — MPS is unreliable for Transformer attention ops

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

def load_split(path):
    d = np.load(path)
    return d["X_train"], d["y_train"], d["X_test"], d["y_test"]

def make_loader(X, y, batch_size=128, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)

def train_and_eval(X_train, y_train, X_test, y_test, n_times, epochs=15):
    X_train = X_train * 1e6
    X_test = X_test * 1e6

    model = TinyEEGTransformer(N_CHANS, n_times, n_classes=N_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_loader = make_loader(X_train, y_train, batch_size=128)
    test_loader = make_loader(X_test, y_test, batch_size=128, shuffle=False)

    model.train()
    for epoch in range(epochs):
        t0 = time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
        print(f"    epoch {epoch+1}/{epochs} done in {time.time()-t0:.1f}s", flush=True)

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            out = model(xb)
            preds.extend(out.argmax(dim=1).cpu().numpy())
            trues.extend(yb.numpy())

    return accuracy_score(trues, preds), f1_score(trues, preds)

results = []

print("=== Transformer: Within-subject ===", flush=True)
for subj in SUBJECTS:
    print(f"  Subject {subj} starting...", flush=True)
    X_train, y_train, X_test, y_test = load_split(f"{DATA_DIR}/within_subject_{subj}.npz")
    acc, f1 = train_and_eval(X_train, y_train, X_test, y_test, X_train.shape[2])
    print(f"Subject {subj}: acc={acc:.3f}, f1={f1:.3f}", flush=True)
    results.append({"model": "Transformer", "split": "within", "subject": subj, "accuracy": acc, "f1": f1})

print("\n=== Transformer: Cross-subject ===", flush=True)
for subj in SUBJECTS:
    print(f"  Holdout {subj} starting...", flush=True)
    X_train, y_train, X_test, y_test = load_split(f"{DATA_DIR}/cross_subject_holdout_{subj}.npz")
    acc, f1 = train_and_eval(X_train, y_train, X_test, y_test, X_train.shape[2])
    print(f"Holdout {subj}: acc={acc:.3f}, f1={f1:.3f}", flush=True)
    results.append({"model": "Transformer", "split": "cross", "subject": subj, "accuracy": acc, "f1": f1})

df = pd.DataFrame(results)
df.to_csv("transformer_results.csv", index=False)
print("\n=== Summary ===")
print(df.groupby("split")[["accuracy", "f1"]].mean())
print("\nSaved to transformer_results.csv")
