"""
Paired subject-level statistical comparison: Policy A vs Policy B and C.

Saves two files:
  paired_statistics.csv          — summary (one row per comparison)
  paired_statistics_per_subject.csv — per-subject differences for every metric
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = "/Users/zarakhursheed/bci_paper/results"

print("=== Paired Statistical Comparisons ===")
print("Decoder: ShallowConvNet, within-subject, seed=42")
print("Threshold: 0.83 (frozen)\n")

df = pd.read_csv(os.path.join(RESULTS_DIR,
                               "policy_comparison_summary.csv"))

pA = df[df["policy"] == "Policy_A_Direct"].set_index("subject")
pB = df[df["policy"] == "Policy_B_ConfidenceGated"].set_index("subject")
pC = df[df["policy"] == "Policy_C_SharedControl"].set_index("subject")

subjects = sorted(pA.index.tolist())

def cohen_d(x, y):
    diff = np.array(x) - np.array(y)
    return diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0

def analyse_pair(name_a, name_b, values_a, values_b, metric):
    print(f"\n--- {name_a} vs {name_b} ({metric}) ---")

    a    = np.array([values_a.loc[s, metric] for s in subjects])
    b    = np.array([values_b.loc[s, metric] for s in subjects])
    diff = b - a

    print(f"Per-subject differences ({name_b} − {name_a}):")
    for s, d in zip(subjects, diff):
        print(f"  Subject {s}: {d:+.3f}")

    mean_diff = diff.mean()
    se_diff   = stats.sem(diff)
    ci        = stats.t.interval(0.95, df=len(diff)-1,
                                  loc=mean_diff, scale=se_diff) \
                if diff.std(ddof=1) > 0 else (float('nan'), float('nan'))

    t_stat, t_p = stats.ttest_rel(b, a) \
        if diff.std(ddof=1) > 0 else (float('nan'), float('nan'))

    try:
        w_stat, w_p = stats.wilcoxon(b, a) \
            if diff.std(ddof=1) > 0 else (float('nan'), float('nan'))
    except ValueError:
        w_stat, w_p = float('nan'), float('nan')

    d = cohen_d(b, a)
    d_interp = (
        "negligible" if abs(d) < 0.2 else
        "small"      if abs(d) < 0.5 else
        "medium"     if abs(d) < 0.8 else
        "large"
    )

    print(f"\nMean difference:       {mean_diff:+.4f} ({mean_diff*100:+.1f} pp)")
    print(f"95% CI:                [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    t_sig = "✓ significant" if (not np.isnan(t_p) and t_p < 0.05) else "✗ not significant"
    w_sig = "✓ significant" if (not np.isnan(w_p) and w_p < 0.05) else "✗ not significant"
    print(f"Paired t-test:         t={t_stat:.3f}, p={t_p:.4f}  {t_sig} (α=0.05)")
    print(f"Wilcoxon signed-rank:  W={w_stat:.1f}, p={w_p:.4f}  {w_sig} (α=0.05)")
    print(f"Cohen's d:             {d:.3f}  → {d_interp} effect")

    return {
        "comparison":    f"{name_a} vs {name_b}",
        "metric":        metric,
        "mean_diff":     round(mean_diff, 4),
        "ci_lower":      round(ci[0], 4) if not np.isnan(ci[0]) else "N/A",
        "ci_upper":      round(ci[1], 4) if not np.isnan(ci[1]) else "N/A",
        "t_stat":        round(t_stat, 4) if not np.isnan(t_stat) else "N/A",
        "t_p":           round(t_p, 4)    if not np.isnan(t_p)    else "N/A",
        "t_significant": not np.isnan(t_p) and t_p < 0.05,
        "wilcoxon_stat": round(w_stat, 4) if not np.isnan(w_stat) else "N/A",
        "wilcoxon_p":    round(w_p, 4)    if not np.isnan(w_p)    else "N/A",
        "w_significant": not np.isnan(w_p) and w_p < 0.05,
        "cohens_d":      round(d, 4),
        "effect_size":   d_interp,
    }

# ── Run comparisons ───────────────────────────────────────────────────────────
summary_rows = []
summary_rows.append(analyse_pair("Policy A","Policy B",pA,pB,"success_rate"))
summary_rows.append(analyse_pair("Policy A","Policy C",pA,pC,"success_rate"))
summary_rows.append(analyse_pair("Policy B","Policy C",pB,pC,"success_rate"))
summary_rows.append(analyse_pair("Policy A","Policy B",pA,pB,"coverage"))
summary_rows.append(analyse_pair("Policy A","Policy B",pA,pB,"error_rate"))
summary_rows.append(analyse_pair("Policy A","Policy B",pA,pB,"abstention_rate"))

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n\n=== Summary Table ===")
print(f"{'Comparison':<25} {'Metric':<16} {'Mean Diff':>10} "
      f"{'p (t-test)':>11} {'p (Wilcox)':>11} {'Cohen d':>9} {'Effect':>12}")
print("-" * 98)
for r in summary_rows:
    t_p_str = f"{r['t_p']:.4f}" if r['t_p'] != "N/A" else "N/A"
    w_p_str = f"{r['wilcoxon_p']:.4f}" if r['wilcoxon_p'] != "N/A" else "N/A"
    t_sig = "✓" if r["t_significant"] else "✗"
    w_sig = "✓" if r["w_significant"] else "✗"
    print(f"{r['comparison']:<25} {r['metric']:<16} "
          f"{r['mean_diff']:>+10.4f} "
          f"{t_p_str:>9} {t_sig}  "
          f"{w_p_str:>9} {w_sig}  "
          f"{r['cohens_d']:>9.3f}  "
          f"{r['effect_size']:>12}")

# ── Save summary CSV ──────────────────────────────────────────────────────────
summary_out = os.path.join(RESULTS_DIR, "paired_statistics.csv")
pd.DataFrame(summary_rows).to_csv(summary_out, index=False)
print(f"\nSummary saved to {summary_out}")

# ── Save per-subject CSV ──────────────────────────────────────────────────────
per_subject_rows = []

comparisons = [
    ("Policy A vs Policy B", pA, pB),
    ("Policy A vs Policy C", pA, pC),
    ("Policy B vs Policy C", pB, pC),
]

for comp_name, va, vb in comparisons:
    for metric in ["success_rate", "error_rate",
                   "coverage", "abstention_rate", "n_unsafe"]:
        for s in subjects:
            av = va.loc[s, metric]
            bv = vb.loc[s, metric]
            per_subject_rows.append({
                "comparison":  comp_name,
                "metric":      metric,
                "subject":     s,
                "policy_a_value": round(float(av), 4),
                "policy_b_value": round(float(bv), 4),
                "difference":     round(float(bv - av), 4),
            })

per_subject_out = os.path.join(RESULTS_DIR,
                                "paired_statistics_per_subject.csv")
pd.DataFrame(per_subject_rows).to_csv(per_subject_out, index=False)
print(f"Per-subject data saved to {per_subject_out}")

print("\n=== Interpretation Guide ===")
print("p < 0.05    → statistically significant")
print("Cohen's d:")
print("  0.0–0.2   → negligible")
print("  0.2–0.5   → small")
print("  0.5–0.8   → medium")
print("  > 0.8     → large (practically meaningful)")
print("\nPrefer Wilcoxon over t-test for reporting — n=9 is small.")
