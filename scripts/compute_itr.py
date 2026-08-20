import pandas as pd
import numpy as np

TRIAL_DURATION_SEC = 4.5
N_CLASSES = 2

def itr_bits_per_min(accuracy, trial_duration_sec, n_classes=2):
    p = np.clip(accuracy, 1e-6, 1 - 1e-6)
    if p == 1.0:
        bits_per_trial = 1.0
    else:
        bits_per_trial = 1 + p * np.log2(p) + (1 - p) * np.log2(1 - p)
    trials_per_min = 60 / trial_duration_sec
    return bits_per_trial * trials_per_min

csp = pd.read_csv("csp_lda_results.csv")
csp["model"] = "CSP+LDA"

deep = pd.read_csv("deep_baseline_results.csv")
transformer = pd.read_csv("transformer_results.csv")

all_results = pd.concat([csp, deep, transformer], ignore_index=True)
all_results["ITR_bits_per_min"] = all_results["accuracy"].apply(
    lambda a: itr_bits_per_min(a, TRIAL_DURATION_SEC, N_CLASSES)
)

all_results.to_csv("all_baseline_results.csv", index=False)

summary = all_results.groupby(["model", "split"])[["accuracy", "f1", "ITR_bits_per_min"]].mean()
print(summary)
summary.to_csv("baseline_summary.csv")
print("\nSaved all_baseline_results.csv and baseline_summary.csv")
