import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from braindecode.models import EEGNetv4, ShallowFBCSPNet
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import os

DATA_DIR = "processed_data_2b"
SUBJECTS = list(range(1, 10))
N_CHANS = 3
N_CLASSES = 2
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_split(path):
    d = np.load(path)
    return d["X_train"], d["y_train"], d["X_test"], d["y_test"]

def make_loader(X, y, batch_size=32, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)

def train_and_eval(model_class, X_train, y_train, X_test, y_test, n_times, epochs=30):
    X_train = X_train * 1e6
    X_test = X_test * 1e6

    model = model_class(n_chans=N_CHANS, n_outputs=N_CLASSES, n_times=n_times).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_loader = make_loader(X_train, y_train)
    test_loader = make_loader(X_test, y_test, shuffle=False)

    model.train()
    for epoch in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            out = model(xb)
            preds.extend(out.argmax(dim=1).cpu().numpy())
            trues.extend(yb.numpy())

    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds)
    return acc, f1

results = []
n_times = None

for model_name, model_class in [("EEGNet", EEGNetv4), ("ShallowConvNet", ShallowFBCSPNet)]:
    print(f"\n=== {model_name}: Within-subject ===")
    for subj in SUBJECTS:
        X_train, y_train, X_test, y_test = load_split(f"{DATA_DIR}/within_subject_{subj}.npz")
        n_times = X_train.shape[2]
        acc, f1 = train_and_eval(model_class, X_train, y_train, X_test, y_test, n_times)
        print(f"Subject {subj}: acc={acc:.3f}, f1={f1:.3f}")
        results.append({"model": model_name, "split": "within", "subject": subj, "accuracy": acc, "f1": f1})

    print(f"\n=== {model_name}: Cross-subject ===")
    for subj in SUBJECTS:
        X_train, y_train, X_test, y_test = load_split(f"{DATA_DIR}/cross_subject_holdout_{subj}.npz")
        n_times = X_train.shape[2]
        acc, f1 = train_and_eval(model_class, X_train, y_train, X_test, y_test, n_times)
        print(f"Holdout {subj}: acc={acc:.3f}, f1={f1:.3f}")
        results.append({"model": model_name, "split": "cross", "subject": subj, "accuracy": acc, "f1": f1})

df = pd.DataFrame(results)
df.to_csv("deep_baseline_results.csv", index=False)
print("\n=== Summary ===")
print(df.groupby(["model", "split"])[["accuracy", "f1"]].mean())
print("\nSaved to deep_baseline_results.csv")

