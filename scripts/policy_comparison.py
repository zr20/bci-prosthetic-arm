"""
Section 4: Three-policy replay comparison.

Decoder: ShallowConvNet (within-subject, seed=42)
Frozen threshold: 0.83 (selected on dev subjects 7-9)

Runs Policy A, B, and C on identical pre-recorded EEG trial streams
from all 9 BCI IV-2b subjects. Reports:
  - Task success rate
  - Incorrect/unsafe actions
  - Abstention rate
  - Coverage
  - Mean latency (ms)

Per Professor Basit: labelled as OFFLINE REPLAY SIMULATION.
"""

import sys
import os
import time
sys.path.insert(0, '/Users/zarakhursheed/bci_paper/configs')
sys.path.insert(0, '/Users/zarakhursheed/bci_paper/scripts')
from config import CONFIG
from state_machine import ArmStateMachine
from control_policies import (PolicyA_DirectControl,
                               PolicyB_ConfidenceGated,
                               PolicyC_SharedControl)

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from braindecode.models import ShallowFBCSPNet
import pandas as pd

SEED      = 42
THRESHOLD = 0.83   # frozen from threshold_selection_shallow.py
SUBJECTS  = list(range(1, CONFIG["n_subjects"] + 1))
TRIAL_DURATION_S = 4.5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)

DEVICE      = torch.device("mps" if torch.backends.mps.is_available()
                            else "cpu")
DATA_DIR    = CONFIG["data_dir"]
RESULTS_DIR = CONFIG["results_dir"]
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=== Three-Policy Replay Comparison ===")
print(f"Decoder:          ShallowConvNet (within-subject)")
print(f"Seed:             {SEED}")
print(f"Frozen threshold: {THRESHOLD}")
print(f"Subjects:         {SUBJECTS}")
print(f"Label:            OFFLINE REPLAY SIMULATION")
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
    model = ShallowFBCSPNet(n_chans=3, n_outputs=2,
                             n_times=n_times).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
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

def get_probs_with_latency(model, X_single):
    t_start  = time.perf_counter()
    X_scaled = X_single * CONFIG["scale"]
    x_tensor = torch.tensor(
        X_scaled[np.newaxis], dtype=torch.float32
    ).to(DEVICE)
    model.eval()
    with torch.no_grad():
        logits = model(x_tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    latency_ms = (time.perf_counter() - t_start) * 1000
    return probs, latency_ms

def evaluate_policy(policy, model, X_test, y_test,
                    policy_name, subject):
    results  = []
    n_trials = len(y_test)

    for i in range(n_trials):
        probs, latency_ms = get_probs_with_latency(model, X_test[i])
        true_label = int(y_test[i])
        t_decide   = time.perf_counter()

        if isinstance(policy, PolicyA_DirectControl):
            decoded, confidence, abstained = policy.decide(probs)
            intent  = f"CLASS_{decoded}"
            correct = (decoded == true_label)
            unsafe  = not correct and not abstained

        elif isinstance(policy, PolicyB_ConfidenceGated):
            decoded, confidence, abstained = policy.decide(probs)
            if abstained:
                intent  = "ABSTAIN"
                correct = False
                unsafe  = False
            else:
                intent  = f"CLASS_{decoded}"
                correct = (decoded == true_label)
                unsafe  = not correct

        elif isinstance(policy, PolicyC_SharedControl):
            intent, confidence, abstained = policy.decide(probs)
            decoded = int(probs.argmax())
            if abstained:
                correct = False
                unsafe  = False
            else:
                correct = (decoded == true_label)
                unsafe  = not correct and not abstained

        decision_ms      = (time.perf_counter() - t_decide) * 1000
        total_latency_ms = latency_ms + decision_ms

        results.append({
            "policy":           policy_name,
            "subject":          subject,
            "trial":            i + 1,
            "true_label":       true_label,
            "decoded":          int(probs.argmax()),
            "confidence":       float(confidence),
            "abstained":        abstained,
            "correct":          correct,
            "unsafe_action":    unsafe,
            "intent":           intent,
            "latency_ms":       round(total_latency_ms, 3),
            "trial_duration_s": TRIAL_DURATION_S,
        })

    return pd.DataFrame(results)

# ── Main evaluation ────────────────────────────────────────────────────────────
all_trial_results = []
subject_summaries = []

for subj in SUBJECTS:
    print(f"Subject {subj}:")

    d       = np.load(f"{DATA_DIR}/within_subject_{subj}.npz")
    X_train = d["X_train"]
    y_train = d["y_train"]
    X_test  = d["X_test"]
    y_test  = d["y_test"]
    n_times = X_train.shape[2]

    model = train_model(X_train * CONFIG["scale"], y_train, n_times)

    pA = PolicyA_DirectControl()
    pB = PolicyB_ConfidenceGated(threshold=THRESHOLD)
    pC = PolicyC_SharedControl(
        threshold=THRESHOLD,
        state_machine=ArmStateMachine(
            log_path=os.path.join(
                RESULTS_DIR, f"sm_log_subj{subj}.csv"
            )
        )
    )

    for policy, name in [
        (pA, "Policy_A_Direct"),
        (pB, "Policy_B_ConfidenceGated"),
        (pC, "Policy_C_SharedControl"),
    ]:
        df_trials = evaluate_policy(
            policy, model, X_test, y_test, name, subj
        )
        all_trial_results.append(df_trials)

        n        = len(df_trials)
        n_acted  = (~df_trials["abstained"]).sum()
        n_abs    = df_trials["abstained"].sum()
        n_corr   = df_trials["correct"].sum()
        n_unsafe = df_trials["unsafe_action"].sum()
        coverage = n_acted / n
        success  = n_corr / n_acted if n_acted > 0 else 0
        mean_lat = df_trials["latency_ms"].mean()

        print(f"  {name}: "
              f"success={success:.3f}, "
              f"coverage={coverage:.3f}, "
              f"unsafe={n_unsafe}, "
              f"latency={mean_lat:.2f}ms")

        subject_summaries.append({
            "policy":          name,
            "subject":         subj,
            "n_trials":        n,
            "n_acted":         int(n_acted),
            "n_abstained":     int(n_abs),
            "n_correct":       int(n_corr),
            "n_unsafe":        int(n_unsafe),
            "coverage":        round(float(coverage), 4),
            "success_rate":    round(float(success), 4),
            "error_rate":      round(float(1 - success), 4),
            "abstention_rate": round(float(n_abs / n), 4),
            "mean_latency_ms": round(float(mean_lat), 3),
        })

# ── Save ──────────────────────────────────────────────────────────────────────
all_trials_df = pd.concat(all_trial_results, ignore_index=True)
all_trials_df.to_csv(
    os.path.join(RESULTS_DIR, "policy_comparison_trials.csv"),
    index=False
)

summary_df = pd.DataFrame(subject_summaries)
summary_df.to_csv(
    os.path.join(RESULTS_DIR, "policy_comparison_summary.csv"),
    index=False
)

# ── Print final table ─────────────────────────────────────────────────────────
print("\n=== Final Policy Comparison (mean across 9 subjects) ===")
print(f"{'Metric':<25} {'Policy A':>14} {'Policy B':>14} {'Policy C':>14}")
print("-" * 67)

policy_means = summary_df.groupby("policy").mean(numeric_only=True)
policy_sums  = summary_df.groupby("policy").sum(numeric_only=True)

rows = [
    ("Success rate",      "success_rate",    False),
    ("Error rate",        "error_rate",      False),
    ("Coverage",          "coverage",        False),
    ("Abstention rate",   "abstention_rate", False),
    ("Unsafe actions",    "n_unsafe",        True),
    ("Mean latency (ms)", "mean_latency_ms", False),
]

for label, col, use_sum in rows:
    src = policy_sums if use_sum else policy_means
    fmt = ".0f" if use_sum else ".3f"
    pA  = f"{src.loc['Policy_A_Direct', col]:{fmt}}"
    pB  = f"{src.loc['Policy_B_ConfidenceGated', col]:{fmt}}"
    pC  = f"{src.loc['Policy_C_SharedControl', col]:{fmt}}"
    print(f"{label:<25} {pA:>14} {pB:>14} {pC:>14}")

print(f"\nDecoder:   ShallowConvNet, within-subject, seed={SEED}")
print(f"Threshold: {THRESHOLD} (frozen on dev subjects 7-9)")
print(f"Results saved to {RESULTS_DIR}")
print("Note: OFFLINE REPLAY SIMULATION per publication plan.")
