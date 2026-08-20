import pandas as pd
import numpy as np
from scipy import stats
import os

RESULTS_DIR = "/Users/zarakhursheed/bci_paper/results"
SEEDS = [42, 0, 1, 2, 3]

dfs = []
for seed in SEEDS:
    path = os.path.join(RESULTS_DIR, f"seed_{seed}",
                        f"cross_dataset_seed{seed}.csv")
    if os.path.exists(path):
        dfs.append(pd.read_csv(path))
    else:
        print(f"Missing: {path}")

all_results = pd.concat(dfs, ignore_index=True)

rows = []
for model, group in all_results.groupby("model"):
    accs = group["accuracy"].values
    mean = accs.mean()
    std  = accs.std()
    ci   = stats.t.interval(0.95, df=len(accs)-1,
                             loc=mean, scale=stats.sem(accs))
    rows.append({
        "model": model,
        "source": "BCI-IV-2b",
        "target": "BCI-IV-2a",
        "mean_accuracy": round(mean, 4),
        "std": round(std, 4),
        "ci_lower": round(ci[0], 4),
        "ci_upper": round(ci[1], 4),
        "n_observations": len(accs)
    })

summary = pd.DataFrame(rows)
print("\n=== Cross-Dataset Transfer Results (BCI IV-2b → BCI IV-2a) ===")
print(summary.to_string(index=False))

# Load within-dataset results for comparison
ci_path = os.path.join(RESULTS_DIR, "results_with_ci.csv")
within = pd.read_csv(ci_path)

print("\n=== Domain Robustness Gap (mean across 5 seeds) ===")
for model in ["EEGNet", "ShallowConvNet"]:
    within_acc = within[
        (within["model"] == model) &
        (within["split"] == "cross")
    ]["mean_accuracy"].values[0]
    cross_ds = summary[summary["model"] == model]["mean_accuracy"].values[0]
    gap = within_acc - cross_ds
    print(f"  {model}: within-dataset={within_acc:.3f}, "
          f"cross-dataset={cross_ds:.3f}, "
          f"domain gap={gap:.3f} ({gap*100:.1f} percentage points)")

out = os.path.join(RESULTS_DIR, "cross_dataset_ci.csv")
summary.to_csv(out, index=False)
print(f"\nSaved to {out}")
