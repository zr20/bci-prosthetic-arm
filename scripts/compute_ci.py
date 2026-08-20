import pandas as pd
import numpy as np
from scipy import stats
import os

RESULTS_DIR = "/Users/zarakhursheed/bci_paper/results"
SEEDS = [42, 0, 1, 2, 3]

# Load all seed results
dfs = []
for seed in SEEDS:
    path = os.path.join(
        RESULTS_DIR, f"seed_{seed}",
        f"deep_baselines_seed{seed}.csv"
    )
    if os.path.exists(path):
        dfs.append(pd.read_csv(path))
    else:
        print(f"Missing: {path}")

all_results = pd.concat(dfs, ignore_index=True)

# Compute mean, std, 95% CI per model/split
rows = []
for (model, split), group in all_results.groupby(["model", "split"]):
    accs = group["accuracy"].values
    mean = accs.mean()
    std  = accs.std()
    n    = len(accs)
    # 95% confidence interval using t-distribution
    ci   = stats.t.interval(0.95, df=n-1,
                             loc=mean,
                             scale=stats.sem(accs))
    rows.append({
        "model": model,
        "split": split,
        "mean_accuracy": round(mean, 4),
        "std": round(std, 4),
        "ci_lower": round(ci[0], 4),
        "ci_upper": round(ci[1], 4),
        "n_seeds": n
    })

summary = pd.DataFrame(rows)
print("\n=== Results with 95% Confidence Intervals ===")
print(summary.to_string(index=False))

out = os.path.join(RESULTS_DIR, "results_with_ci.csv")
summary.to_csv(out, index=False)
print(f"\nSaved to {out}")
