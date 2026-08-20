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
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

def make_loader(X, y, batch_size=128, shuffle=True):
    return DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        ), batch_size=batch_size, shuffle=shuffle
    )

def train_model(model_class, X_train, y_train, n_times):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = model_class(n_chans=3, n_outputs=2,
                        n_times=n_times).to(DEVICE)
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
    return model

def get_probs(model, X_test):
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        return torch.softmax(model(xb), dim=1).cpu().numpy()

def brier_score(y_true, probs):
    """
    Multi-class Brier score.
    Range: 0 (perfect) to 2 (worst).
    Chance level for binary classification = 0.50.
    Lower is better.
    """
    y_onehot = np.zeros((len(y_true), 2))
    y_onehot[np.arange(len(y_true)), y_true] = 1
    return np.mean(np.sum((probs - y_onehot) ** 2, axis=1))

def expected_calibration_error(y_true, probs, n_bins=10):
    """
    Expected Calibration Error (ECE).
    Measures mean absolute difference between confidence and accuracy.
    Range: 0 (perfect calibration) to 1 (worst).
    Lower is better.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i+1])
        if mask.sum() > 0:
            bin_acc  = correct[mask].mean()
            bin_conf = confidences[mask].mean()
            bin_frac = mask.mean()
            ece += bin_frac * abs(bin_acc - bin_conf)
            bin_data.append({
                "bin_centre": (bins[i] + bins[i+1]) / 2,
                "accuracy":   bin_acc,
                "confidence": bin_conf,
                "fraction":   bin_frac
            })
    return ece, bin_data

def risk_coverage_curve(y_true, probs, n_thresholds=50):
    """
    As confidence threshold rises, coverage drops and error rate drops.
    Shows the trade-off between how often the system acts and how often it errs.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y_true).astype(float)
    thresholds = np.linspace(0.5, 1.0, n_thresholds)
    curve = []
    for t in thresholds:
        mask = confidences >= t
        if mask.sum() > 0:
            curve.append({
                "threshold":  t,
                "coverage":   mask.mean(),
                "error_rate": 1 - correct[mask].mean()
            })
    return curve

SUBJECTS = list(range(1, CONFIG["n_subjects"] + 1))
all_results = []
all_curves  = []
plot_data   = {}

for model_name, model_class in [
    ("EEGNet", EEGNetv4),
    ("ShallowConvNet", ShallowFBCSPNet)
]:
    print(f"\n=== Calibration: {model_name} (seed={SEED}) ===")
    all_probs, all_labels = [], []

    for subj in SUBJECTS:
        d = np.load(f"{DATA_DIR}/within_subject_{subj}.npz")
        X_train = d["X_train"] * CONFIG["scale"]
        y_train = d["y_train"]
        X_test  = d["X_test"]  * CONFIG["scale"]
        y_test  = d["y_test"]
        model  = train_model(model_class, X_train, y_train, X_train.shape[2])
        probs  = get_probs(model, X_test)
        all_probs.append(probs)
        all_labels.append(y_test)

    all_probs  = np.concatenate(all_probs,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    bs          = brier_score(all_labels, all_probs)
    ece, bin_data = expected_calibration_error(all_labels, all_probs)
    rc_curve    = risk_coverage_curve(all_labels, all_probs)
    acc         = accuracy_score(all_labels, all_probs.argmax(axis=1))

    # Correct interpretations
    brier_interp = (
        "better than chance" if bs < 0.50
        else "at or worse than chance"
    )
    ece_interp = (
        "well calibrated" if ece < 0.05
        else ("moderately overconfident" if ece < 0.15
              else "poorly calibrated")
    )

    print(f"  Accuracy:    {acc:.3f}")
    print(f"  Brier Score: {bs:.4f}  "
          f"[chance=0.50 for binary; lower=better] → {brier_interp}")
    print(f"  ECE:         {ece:.4f}  "
          f"[0=perfect calibration; lower=better] → {ece_interp}")

    all_results.append({
        "seed": SEED,
        "model": model_name,
        "accuracy": acc,
        "brier_score": bs,
        "brier_chance_level": 0.50,
        "brier_interpretation": brier_interp,
        "ece": ece,
        "ece_interpretation": ece_interp
    })

    for row in bin_data:
        all_curves.append({
            "seed": SEED, "model": model_name,
            "type": "calibration", **row
        })
    for row in rc_curve:
        all_curves.append({
            "seed": SEED, "model": model_name,
            "type": "risk_coverage", **row
        })

    plot_data[model_name] = {
        "bin_data": bin_data,
        "rc_curve": rc_curve,
        "bs": bs, "ece": ece
    }

pd.DataFrame(all_results).to_csv(
    os.path.join(RESULTS_DIR, f"calibration_seed{SEED}.csv"), index=False
)
pd.DataFrame(all_curves).to_csv(
    os.path.join(RESULTS_DIR, f"calibration_curves_seed{SEED}.csv"), index=False
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = {"EEGNet": "#2196F3", "ShallowConvNet": "#4CAF50"}

for model_name, data in plot_data.items():
    bd   = data["bin_data"]
    confs = [b["confidence"] for b in bd]
    accs  = [b["accuracy"]   for b in bd]
    axes[0].plot(confs, accs, 'o-',
                 label=f"{model_name} (ECE={data['ece']:.3f})",
                 color=colors[model_name], linewidth=2)
    rc    = data["rc_curve"]
    covs  = [r["coverage"]   for r in rc]
    risks = [r["error_rate"] for r in rc]
    axes[1].plot(covs, risks, '-',
                 label=f"{model_name} (BS={data['bs']:.3f})",
                 color=colors[model_name], linewidth=2)

axes[0].plot([0,1],[0,1],'k--', label='Perfect calibration', alpha=0.5)
axes[0].set_xlabel('Mean Confidence')
axes[0].set_ylabel('Fraction Correct')
axes[0].set_title(f'Calibration Curves (seed={SEED})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0.5, 1.0)
axes[0].set_ylim(0, 1)

axes[1].set_xlabel('Coverage (fraction of trials acted on)')
axes[1].set_ylabel('Error Rate (risk)')
axes[1].set_title(f'Risk-Coverage Curves (seed={SEED})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].invert_xaxis()

plt.tight_layout()
out_fig = os.path.join(RESULTS_DIR, f"calibration_plots_seed{SEED}.png")
plt.savefig(out_fig, dpi=150)
print(f"\nPlots saved to {out_fig}")
