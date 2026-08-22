"""
Temperature scaling calibration for ShallowConvNet.
Fits a single temperature parameter T on held-out calibration
data (20% of training set) to correct overconfident probabilities.
Reports ECE before and after scaling per subject.
"""

import sys
import os
sys.path.insert(0, '/Users/zarakhursheed/bci_paper/configs')
from config import CONFIG

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from braindecode.models import ShallowFBCSPNet
from sklearn.metrics import accuracy_score
import pandas as pd

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE   = torch.device("mps" if torch.backends.mps.is_available()
                         else "cpu")
DATA_DIR    = CONFIG["data_dir"]
RESULTS_DIR = CONFIG["results_dir"]
os.makedirs(RESULTS_DIR, exist_ok=True)

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits, labels, lr=0.01, max_iter=200):
        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=lr, max_iter=max_iter
        )
        criterion = nn.CrossEntropyLoss()
        logits_t  = torch.tensor(logits, dtype=torch.float32)
        labels_t  = torch.tensor(labels, dtype=torch.long)

        def eval_step():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits_t), labels_t)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        return self

def make_loader(X, y, shuffle=True):
    return DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        ), batch_size=128, shuffle=shuffle
    )

def train_model(X_train, y_train, n_times):
    torch.manual_seed(SEED)
    model = ShallowFBCSPNet(
        n_chans=3, n_outputs=2, n_times=n_times
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss()
    loader    = make_loader(X_train, y_train)
    model.train()
    for _ in range(CONFIG["epochs"]):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
    return model

def get_logits(model, X):
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        return model(xb).cpu().numpy()

def compute_ece(y_true, probs, n_bins=10):
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    corr  = (preds == y_true).astype(float)
    bins  = np.linspace(0, 1, n_bins + 1)
    ece   = 0.0
    for i in range(n_bins):
        mask = (confs >= bins[i]) & (confs < bins[i+1])
        if mask.sum() > 0:
            ece += mask.mean() * abs(
                corr[mask].mean() - confs[mask].mean()
            )
    return ece

SUBJECTS = list(range(1, 10))
rows = []

print("=== Temperature Scaling — ShallowConvNet ===")
print("Calibration split: 20% of training data (held out before training)")
print()
print(f"{'Subject':<10} {'Temp T':>8} {'ECE before':>12} "
      f"{'ECE after':>11} {'Improvement':>13} {'Accuracy':>10}")
print("-" * 68)

for subj in SUBJECTS:
    d       = np.load(f"{DATA_DIR}/within_subject_{subj}.npz")
    X_train = d["X_train"] * CONFIG["scale"]
    y_train = d["y_train"]
    X_test  = d["X_test"]  * CONFIG["scale"]
    y_test  = d["y_test"]

    # Split: 80% train, 20% calibration
    # Calibration data is held out BEFORE training
    n_cal = int(len(X_train) * 0.2)
    X_cal = X_train[-n_cal:]
    y_cal = y_train[-n_cal:]
    X_tr  = X_train[:-n_cal]
    y_tr  = y_train[:-n_cal]

    model = train_model(X_tr, y_tr, X_tr.shape[2])

    # Fit temperature on calibration set only
    cal_logits = get_logits(model, X_cal)
    scaler     = TemperatureScaler()
    scaler.fit(cal_logits, y_cal)
    T = scaler.temperature.item()

    # Evaluate on test set before and after
    test_logits = get_logits(model, X_test)
    probs_raw   = torch.softmax(
        torch.tensor(test_logits), dim=1
    ).numpy()
    probs_cal   = torch.softmax(
        torch.tensor(test_logits) / T, dim=1
    ).numpy()

    ece_before  = compute_ece(y_test, probs_raw)
    ece_after   = compute_ece(y_test, probs_cal)
    improvement = ece_before - ece_after
    acc         = accuracy_score(y_test,
                                  probs_cal.argmax(axis=1))

    print(f"{subj:<10} {T:>8.3f} {ece_before:>12.4f} "
          f"{ece_after:>11.4f} {improvement:>+13.4f} {acc:>10.3f}")

    rows.append({
        "subject":      subj,
        "temperature":  round(T, 4),
        "ece_before":   round(ece_before, 4),
        "ece_after":    round(ece_after, 4),
        "improvement":  round(improvement, 4),
        "accuracy":     round(acc, 4),
    })

df  = pd.DataFrame(rows)
out = os.path.join(RESULTS_DIR, "temperature_scaling.csv")
df.to_csv(out, index=False)

print()
print(f"Mean ECE before temperature scaling: {df['ece_before'].mean():.4f}")
print(f"Mean ECE after  temperature scaling: {df['ece_after'].mean():.4f}")
print(f"Mean improvement:                    {df['improvement'].mean():+.4f}")
print(f"Mean temperature T:                  {df['temperature'].mean():.3f}")
print(f"Mean accuracy (post-scaling):        {df['accuracy'].mean():.3f}")
print()
print(f"Saved to {out}")
print()
print("Note: temperature scaling fitted on held-out calibration")
print("data (20% of training set) per subject. Never fitted on")
print("test data. Accuracy is unchanged by temperature scaling")
print("(only probabilities are adjusted, not predictions).")
