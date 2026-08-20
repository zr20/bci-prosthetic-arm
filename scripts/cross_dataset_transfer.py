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
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery
import pandas as pd

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DATA_DIR = CONFIG["data_dir"]
RESULTS_DIR = os.path.join(CONFIG["results_dir"], f"seed_{SEED}")
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Cross-dataset transfer: BCI IV-2b to BCI IV-2a")
print(f"Seed={SEED}, device={DEVICE}")

# Load BCI IV-2b source data (raw .gdf — needs x1e6 scaling)
print("\nLoading BCI IV-2b source data...")
SUBJECTS_2B = list(range(1, 10))
X_source_list, y_source_list = [], []
for subj in SUBJECTS_2B:
    d = np.load(f"{DATA_DIR}/within_subject_{subj}.npz")
    X_all = np.concatenate([d["X_train"], d["X_test"]], axis=0)
    y_all = np.concatenate([d["y_train"], d["y_test"]], axis=0)
    X_source_list.append(X_all)
    y_source_list.append(y_all)
    print(f"  2b Subject {subj}: {len(y_all)} trials")

# Apply x1e6 scaling to 2b data (raw Volts from .gdf files)
X_source = np.concatenate(X_source_list, axis=0) * CONFIG["scale"]
y_source = np.concatenate(y_source_list, axis=0)
n_times_source = X_source.shape[2]
print(f"Source total: {X_source.shape}")
print(f"Source amplitude std: {X_source.std():.2f} (should be ~2-10)")

# Load and harmonise BCI IV-2a target data
# NOTE: MOABB LeftRightImagery returns data already in microvolts
# DO NOT apply x1e6 scaling — amplitude mismatch would cause model collapse
print("\nLoading and harmonising BCI IV-2a target data...")
print("  Channel subset: C3 (idx 7), Cz (idx 9), C4 (idx 11)")
print("  Epoch trim: 1001 -> 875 samples")
print("  Sampling rate: 250Hz (identical)")
print("  Labels: left_hand->0, right_hand->1")
print("  Scaling: NONE — MOABB already returns microvolts")

C3_IDX, CZ_IDX, C4_IDX = 7, 9, 11
dataset_2a = BNCI2014_001()
paradigm_2a = LeftRightImagery(fmin=4, fmax=40)

SUBJECTS_2A = list(range(1, 10))
X_target_per_subject = {}
y_target_per_subject = {}

for subj in SUBJECTS_2A:
    X, labels, meta = paradigm_2a.get_data(
        dataset=dataset_2a, subjects=[subj]
    )
    X_subset = X[:, [C3_IDX, CZ_IDX, C4_IDX], :n_times_source]
    y = np.array([0 if l == 'left_hand' else 1 for l in labels])
    # No x1e6 scaling — already in microvolts from MOABB
    X_target_per_subject[subj] = X_subset
    y_target_per_subject[subj] = y
    print(f"  2a Subject {subj}: {len(y)} trials, "
          f"std={X_subset.std():.2f}")

def make_loader(X, y, batch_size=128, shuffle=True):
    return DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        ),
        batch_size=batch_size, shuffle=shuffle
    )

def train_model(model_class, X_train, y_train, n_times):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = model_class(
        n_chans=3, n_outputs=2, n_times=n_times
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss()
    loader = make_loader(X_train, y_train)
    model.train()
    for epoch in range(CONFIG["epochs"]):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{CONFIG['epochs']} done")
    return model

def evaluate(model, X_test, y_test):
    model.eval()
    loader = make_loader(X_test, y_test, shuffle=False)
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            preds.extend(model(xb).argmax(dim=1).cpu().numpy())
            trues.extend(yb.numpy())
    return accuracy_score(trues, preds), f1_score(trues, preds,
                                                   zero_division=0)

results = []

for model_name, model_class in [
    ("EEGNet", EEGNetv4),
    ("ShallowConvNet", ShallowFBCSPNet)
]:
    print(f"\n=== Training {model_name} on BCI IV-2b ===")
    model = train_model(model_class, X_source, y_source, n_times_source)

    print(f"\n=== Testing {model_name} on BCI IV-2a subjects ===")
    for subj in SUBJECTS_2A:
        X_test = X_target_per_subject[subj]
        y_test = y_target_per_subject[subj]
        acc, f1 = evaluate(model, X_test, y_test)
        print(f"  2a Subject {subj}: acc={acc:.3f}, f1={f1:.3f}")
        results.append({
            "seed": SEED,
            "model": model_name,
            "source": "BCI-IV-2b",
            "target": "BCI-IV-2a",
            "target_subject": subj,
            "accuracy": acc,
            "f1": f1
        })

df = pd.DataFrame(results)
out = os.path.join(RESULTS_DIR, f"cross_dataset_seed{SEED}.csv")
df.to_csv(out, index=False)

print(f"\n=== Summary (seed={SEED}) ===")
print(df.groupby("model")[["accuracy", "f1"]].mean().round(3))

print("\n=== Domain Robustness Gap ===")
ci_path = "/Users/zarakhursheed/bci_paper/results/results_with_ci.csv"
if os.path.exists(ci_path):
    ci_results = pd.read_csv(ci_path)
    for model_name in ["EEGNet", "ShallowConvNet"]:
        within_ds = ci_results[
            (ci_results["model"] == model_name) &
            (ci_results["split"] == "cross")
        ]["mean_accuracy"].values[0]
        cross_ds = df[df["model"] == model_name]["accuracy"].mean()
        gap = within_ds - cross_ds
        print(f"  {model_name}: within-dataset={within_ds:.3f}, "
              f"cross-dataset={cross_ds:.3f}, "
              f"domain gap={gap:+.3f}")

print(f"\nSaved to {out}")
