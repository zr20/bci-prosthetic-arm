"""
Completion time analysis.

For each subject and policy, computes how many trials
were needed to achieve one accepted command, and converts
that to time using the fixed trial duration of 4.5 seconds.

Completion time = trials needed × 4.5 seconds
For Policy A: always 1 trial = 4.5 seconds
For Policy B/C: depends on how many rejections/proposals
before confirmation
"""

import os
import pandas as pd
import numpy as np

RESULTS_DIR      = "/Users/zarakhursheed/bci_paper/results"
TRIAL_DURATION_S = 4.5

trials_df = pd.read_csv(
    os.path.join(RESULTS_DIR, "policy_comparison_trials_v2.csv")
)

rows = []

for policy in ["Policy_A_Direct",
               "Policy_B_ConfidenceGated",
               "Policy_C_SharedControl"]:

    pdf = trials_df[trials_df["policy"] == policy].copy()

    for subj in sorted(pdf["subject"].unique()):
        sdf = pdf[pdf["subject"] == subj].reset_index(drop=True)

        # Find groups of trials leading to each accepted command
        # A group starts after the previous accepted trial
        # and ends at the next accepted trial
        accepted_indices = sdf[~sdf["abstained"]].index.tolist()

        if len(accepted_indices) == 0:
            mean_trials = float("nan")
            mean_time_s = float("nan")
        else:
            group_sizes = []
            prev_idx = -1
            for idx in accepted_indices:
                group_size = idx - prev_idx
                group_sizes.append(group_size)
                prev_idx = idx

            mean_trials = np.mean(group_sizes)
            mean_time_s = mean_trials * TRIAL_DURATION_S

        rows.append({
            "policy":            policy,
            "subject":           subj,
            "n_accepted":        len(accepted_indices),
            "mean_trials_per_command": round(float(mean_trials), 2)
                                       if not pd.isna(mean_trials)
                                       else "N/A",
            "mean_completion_time_s":  round(float(mean_time_s), 2)
                                       if not pd.isna(mean_time_s)
                                       else "N/A",
        })

df = pd.DataFrame(rows)

print("=== Completion Time Analysis ===")
print(f"Trial duration: {TRIAL_DURATION_S}s")
print(f"Completion time = mean trials needed per accepted command "
      f"× {TRIAL_DURATION_S}s\n")

print(f"{'Policy':<30} {'Subject':>8} {'N Accepted':>12} "
      f"{'Mean Trials':>13} {'Mean Time (s)':>14}")
print("-" * 80)

for _, row in df.iterrows():
    print(f"{row['policy']:<30} {row['subject']:>8} "
          f"{row['n_accepted']:>12} "
          f"{str(row['mean_trials_per_command']):>13} "
          f"{str(row['mean_completion_time_s']):>14}")

print("\n=== Summary — Mean Across 9 Subjects ===")
numeric_df = df[df["mean_completion_time_s"] != "N/A"].copy()
numeric_df["mean_completion_time_s"] = numeric_df[
    "mean_completion_time_s"
].astype(float)
numeric_df["mean_trials_per_command"] = numeric_df[
    "mean_trials_per_command"
].astype(float)

summary = numeric_df.groupby("policy")[
    ["mean_trials_per_command", "mean_completion_time_s"]
].mean().round(3)
print(summary.to_string())

out = os.path.join(RESULTS_DIR, "completion_time.csv")
df.to_csv(out, index=False)
print(f"\nSaved to {out}")
