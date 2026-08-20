import sys
import os
sys.path.insert(0, '/Users/zarakhursheed/bci_paper/configs')
from config import CONFIG

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from braindecode.models import EEGNetv4, ShallowFBCSPNet
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

# ── Get seed from command line argument ───────────────────────────────────────
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42

# ── Fix all random seeds ──────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DATA_DIR = CONFIG["data_dir"]
RESULTS_DIR = os.path.join(CONFIG["results_dir"], f"seed_{SEED}")
CHECKPOINT_DIR = CONFIG["checkpoint_dir"]
SUBJECTS = list(range(1, CONFIG["n_subjects"] + 1))
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"Running with seed={SEED}, device={DEVICE}")

def load_split(path):
    d = np.load(path)
    return d["X_train"], d["y_train"], d["X_test"], d["y_test"]

def make_loader(X, y, batch_size=128, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t),
                      batch_size=batch_size, shuffle=shuffle)

def train_and_eval(model_class, X_train, y_train, X_test, y_test,
                   n_times, model_name, split_name, subject):
    # Re-fix seed before each model init so results are
    # independent of run order
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_train = X_train * CONFIG["scale"]
    X_test  = X_test  * CONFIG["scale"]

    model = model_class(
        n_chans=CONFIG["n_channels"],
        n_outputs=2,
        n_times=n_times
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=CONFIG["lr"]
    )
    criterion = nn.CrossEntropyLoss()

    train_loader = make_loader(X_train, y_train)
    test_loader  = make_loader(X_test,  y_test, shuffle=False)

    model.train()
    for epoch in range(CONFIG["epochs"]):
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

    # Save checkpoint
    ckpt_name = (f"{model_name}_{split_name}_subj{subject}"
                 f"_seed{SEED}.pt")
    torch.save(model.state_dict(),
               os.path.join(CHECKPOINT_DIR, ckpt_name))

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            preds.extend(model(xb).argmax(dim=1).cpu().numpy())
            trues.extend(yb.numpy())

    acc = accuracy_score(trues, preds)
    f1  = f1_score(trues, preds, zero_division=0)
    return acc, f1

results = []

for model_name, model_class in [
    ("EEGNet", EEGNetv4),
    ("ShallowConvNet", ShallowFBCSPNet)
]:
    print(f"\n=== {model_name} — Within-subject (seed={SEED}) ===")
    for subj in SUBJECTS:
        X_train, y_train, X_test, y_test = load_split(
            f"{DATA_DIR}/within_subject_{subj}.npz"
        )
        acc, f1 = train_and_eval(
            model_class, X_train, y_train,
            X_test, y_test, X_train.shape[2],
            model_name, "within", subj
        )
        print(f"  Subject {subj}: acc={acc:.3f}, f1={f1:.3f}")
        results.append({
            "seed": SEED, "model": model_name,
            "split": "within", "subject": subj,
            "accuracy": acc, "f1": f1
        })

    print(f"\n=== {model_name} — Cross-subject (seed={SEED}) ===")
    for subj in SUBJECTS:
        X_train, y_train, X_test, y_test = load_split(
            f"{DATA_DIR}/cross_subject_holdout_{subj}.npz"
        )
        acc, f1 = train_and_eval(
            model_class, X_train, y_train,
            X_test, y_test, X_train.shape[2],
            model_name, "cross", subj
        )
        print(f"  Holdout {subj}: acc={acc:.3f}, f1={f1:.3f}")
        results.append({
            "seed": SEED, "model": model_name,
            "split": "cross", "subject": subj,
            "accuracy": acc, "f1": f1
        })

df = pd.DataFrame(results)
out_path = os.path.join(RESULTS_DIR, f"deep_baselines_seed{SEED}.csv")
df.to_csv(out_path, index=False)

print(f"\n=== Summary (seed={SEED}) ===")
print(df.groupby(["model", "split"])[["accuracy", "f1"]].mean().round(3))
print(f"\nSaved to {out_path}")
