"""
Confusion matrices for EEGNet and ShallowConvNet.
Runs across all 9 subjects, 5 seeds. Saves per-subject
and aggregate confusion matrices to CSV and prints them.
"""

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
from sklearn.metrics import confusion_matrix, f1_score
import pandas as pd

SEEDS    = CONFIG["seeds"]
SUBJECTS = list(range(1, CONFIG["n_subjects"] + 1))
DATA_DIR = CONFIG["data_dir"]
RESULTS_DIR = CONFIG["results_dir"]
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def make_loader(X, y, batch_size=128, shuffle=True):
    return DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        ), batch_size=batch_size, shuffle=shuffle
    )

def train_and_predict(model_class, X_train, y_train,
                      X_test, y_test, n_times, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model_class(
        n_chans=3, n_outputs=2, n_times=n_times
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss()
    loader = make_loader(X_train, y_train)

    model.train()
    for _ in range(CONFIG["epochs"]):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        xb   = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        pred = model(xb).argmax(dim=1).cpu().numpy()
    return pred

rows = []

for model_name, model_class in [
    ("EEGNet", EEGNetv4),
    ("ShallowConvNet", ShallowFBCSPNet)
]:
    print(f"\n=== Confusion Matrices: {model_name} ===")

    # Aggregate across all subjects and seeds
    all_true, all_pred = [], []

    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        for subj in SUBJECTS:
            d = np.load(
                f"{DATA_DIR}/within_subject_{subj}.npz"
            )
            X_train = d["X_train"] * CONFIG["scale"]
            y_train = d["y_train"]
            X_test  = d["X_test"]  * CONFIG["scale"]
            y_test  = d["y_test"]

            pred = train_and_predict(
                model_class, X_train, y_train,
                X_test, y_test, X_train.shape[2], seed
            )

            # Per-subject confusion matrix
            cm  = confusion_matrix(y_test, pred, labels=[0, 1])
            f1  = f1_score(y_test, pred, zero_division=0)
            acc = (pred == y_test).mean()

            tn, fp, fn, tp = cm.ravel()
            rows.append({
                "model":   model_name,
                "seed":    seed,
                "subject": subj,
                "split":   "within",
                "TP": int(tp), "TN": int(tn),
                "FP": int(fp), "FN": int(fn),
                "accuracy": round(float(acc), 4),
                "macro_f1": round(float(f1), 4),
                "sensitivity": round(float(tp/(tp+fn)) if (tp+fn)>0 else 0, 4),
                "specificity": round(float(tn/(tn+fp)) if (tn+fp)>0 else 0, 4),
            })

            all_true.extend(y_test.tolist())
            all_pred.extend(pred.tolist())

    # Aggregate confusion matrix across all subjects and seeds
    cm_agg = confusion_matrix(all_true, all_pred, labels=[0, 1])
    tn, fp, fn, tp = cm_agg.ravel()
    total = len(all_true)

    print(f"\nAggregate confusion matrix ({len(SEEDS)} seeds × {len(SUBJECTS)} subjects):")
    print(f"                  Predicted Left   Predicted Right")
    print(f"  True Left       {tp:>6} ({tp/total*100:.1f}%)    {fn:>6} ({fn/total*100:.1f}%)")
    print(f"  True Right      {fp:>6} ({fp/total*100:.1f}%)    {tn:>6} ({tn/total*100:.1f}%)")
    print(f"  Sensitivity (True Left rate):  {tp/(tp+fn):.3f}")
    print(f"  Specificity (True Right rate): {tn/(tn+fp):.3f}")
    print(f"  Macro-F1: {f1_score(all_true, all_pred, average='macro'):.3f}")

df = pd.DataFrame(rows)
out = os.path.join(RESULTS_DIR, "confusion_matrices.csv")
df.to_csv(out, index=False)

# Summary: mean per model/split
print("\n=== Summary — Mean Across Subjects and Seeds ===")
summary = df.groupby(["model", "split"])[
    ["accuracy", "macro_f1", "sensitivity", "specificity"]
].mean().round(3)
print(summary)

summary_out = os.path.join(RESULTS_DIR, "confusion_summary.csv")
summary.reset_index().to_csv(summary_out, index=False)

print(f"\nSaved to:")
print(f"  {out}")
print(f"  {summary_out}")
