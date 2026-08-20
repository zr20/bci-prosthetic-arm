"""
Confidence threshold selection for Policies B and C.

Professor Basit's requirement:
  "Choose the abstention threshold before the final evaluation
   using development data."

Method:
  Use a held-out development set (subjects 7-9 from BCI IV-2b,
  separate from the final evaluation subjects 1-6) to sweep
  thresholds and pick the one that minimises error rate while
  maintaining at least 60% coverage.

This is done ONCE before any final evaluation, then frozen.
"""

import sys
import os
sys.path.insert(0, '/Users/zarakhursheed/bci_paper/configs')
sys.path.insert(0, '/Users/zarakhursheed/bci_paper/scripts')
from config import CONFIG

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from braindecode.models import EEGNetv4
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE     = torch.device("mps" if torch.backends.mps.is_available()
                           else "cpu")
DATA_DIR   = CONFIG["data_dir"]
RESULTS_DIR = CONFIG["results_dir"]
os.makedirs(RESULTS_DIR, exist_ok=True)

# Development subjects (held out from final evaluation)
DEV_SUBJECTS   = [7, 8, 9]
EVAL_SUBJECTS  = [1, 2, 3, 4, 5, 6]
MIN_COVERAGE   = 0.60   # must act on at least 60% of trials

print("=== Confidence Threshold Selection ===")
print(f"Development subjects: {DEV_SUBJECTS}")
print(f"Evaluation subjects:  {EVAL_SUBJECTS}")
print(f"Minimum coverage:     {MIN_COVERAGE:.0%}")
print()

def make_loader(X, y, batch_size=128, shuffle=True):
    return DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        ), batch_size=batch_size, shuffle=shuffle
    )

def train_model(X_train, y_train, n_times):
    torch.manual_seed(SEED)
    model = EEGNetv4(n_chans=3, n_outputs=2,
                     n_times=n_times).to(DEVICE)
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

def get_probs(model, X):
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        return torch.softmax(model(xb), dim=1).cpu().numpy()

# Train on all non-dev subjects, collect probs on dev subjects
print("Training EEGNet on evaluation subjects...")
X_train_list, y_train_list = [], []
for subj in EVAL_SUBJECTS:
    d = np.load(f"{DATA_DIR}/within_subject_{subj}.npz")
    X_train_list.append(d["X_train"] * CONFIG["scale"])
    y_train_list.append(d["y_train"])

X_train_all = np.concatenate(X_train_list)
y_train_all = np.concatenate(y_train_list)
n_times     = X_train_all.shape[2]

model = train_model(X_train_all, y_train_all, n_times)
print("Done.\n")

# Collect dev set probabilities
print("Collecting dev set probabilities...")
dev_probs, dev_labels = [], []
for subj in DEV_SUBJECTS:
    d = np.load(f"{DATA_DIR}/within_subject_{subj}.npz")
    X_test = d["X_test"] * CONFIG["scale"]
    y_test = d["y_test"]
    probs  = get_probs(model, X_test)
    dev_probs.append(probs)
    dev_labels.append(y_test)
    print(f"  Subject {subj}: {len(y_test)} trials")

dev_probs  = np.concatenate(dev_probs)
dev_labels = np.concatenate(dev_labels)
confidences = dev_probs.max(axis=1)
predictions = dev_probs.argmax(axis=1)
correct     = (predictions == dev_labels).astype(float)
print(f"\nDev set: {len(dev_labels)} trials total")
print(f"Baseline accuracy (no abstention): "
      f"{correct.mean():.3f}")

# Sweep thresholds
thresholds = np.arange(0.50, 0.96, 0.01)
results    = []

for t in thresholds:
    mask     = confidences >= t
    coverage = mask.mean()
    if mask.sum() == 0:
        continue
    error    = 1 - correct[mask].mean()
    results.append({
        "threshold": round(float(t), 2),
        "coverage":  round(float(coverage), 4),
        "error_rate": round(float(error), 4),
        "n_acted":   int(mask.sum()),
        "n_abstained": int((~mask).sum())
    })

df = pd.DataFrame(results)

# Select threshold: lowest error rate with coverage >= MIN_COVERAGE
eligible = df[df["coverage"] >= MIN_COVERAGE]
if len(eligible) == 0:
    print("WARNING: No threshold meets minimum coverage. Using 0.60.")
    best_threshold = 0.60
else:
    best_idx       = eligible["error_rate"].idxmin()
    best_threshold = eligible.loc[best_idx, "threshold"]
    best_coverage  = eligible.loc[best_idx, "coverage"]
    best_error     = eligible.loc[best_idx, "error_rate"]

print(f"\n=== Selected Threshold: {best_threshold} ===")
print(f"  Error rate at threshold: {best_error:.3f}")
print(f"  Coverage at threshold:   {best_coverage:.3f}")
print(f"  Abstention rate:         {1-best_coverage:.3f}")
print(f"\nThis threshold is FROZEN for all final evaluations.")

# Save threshold and sweep results
df.to_csv(os.path.join(RESULTS_DIR, "threshold_sweep.csv"),
          index=False)

with open(os.path.join(RESULTS_DIR, "selected_threshold.txt"), "w") as f:
    f.write(f"selected_threshold={best_threshold}\n")
    f.write(f"dev_subjects={DEV_SUBJECTS}\n")
    f.write(f"eval_subjects={EVAL_SUBJECTS}\n")
    f.write(f"min_coverage={MIN_COVERAGE}\n")
    f.write(f"error_rate_at_threshold={best_error}\n")
    f.write(f"coverage_at_threshold={best_coverage}\n")
    f.write(f"seed={SEED}\n")

print(f"\nThreshold saved to "
      f"{RESULTS_DIR}/selected_threshold.txt")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(df["threshold"], df["error_rate"],
             'b-', linewidth=2, label="Error rate")
axes[0].plot(df["threshold"], df["coverage"],
             'g--', linewidth=2, label="Coverage")
axes[0].axvline(best_threshold, color='red', linestyle=':',
                linewidth=2, label=f"Selected: {best_threshold}")
axes[0].axhline(MIN_COVERAGE, color='gray', linestyle=':',
                alpha=0.5, label=f"Min coverage: {MIN_COVERAGE}")
axes[0].set_xlabel("Confidence Threshold")
axes[0].set_ylabel("Rate")
axes[0].set_title("Threshold Selection (Dev Set)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(df["coverage"], df["error_rate"],
             'b-o', linewidth=2, markersize=3)
axes[1].axvline(best_coverage, color='red', linestyle=':',
                linewidth=2,
                label=f"Selected (cov={best_coverage:.2f}, "
                      f"err={best_error:.3f})")
axes[1].set_xlabel("Coverage (fraction of trials acted on)")
axes[1].set_ylabel("Error Rate")
axes[1].set_title("Risk-Coverage Trade-off (Dev Set)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].invert_xaxis()

plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, "threshold_selection.png")
plt.savefig(fig_path, dpi=150)
print(f"Plot saved to {fig_path}")
