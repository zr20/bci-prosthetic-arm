import pandas as pd

df = pd.read_csv("all_baseline_results.csv")

within = df[df["split"] == "within"].groupby(["model", "subject"])["accuracy"].mean().reset_index()
cross = df[df["split"] == "cross"].groupby(["model", "subject"])["accuracy"].mean().reset_index()

merged = within.merge(cross, on=["model", "subject"], suffixes=("_within", "_cross"))
merged["gap"] = merged["accuracy_within"] - merged["accuracy_cross"]

summary = merged.groupby("model")[["accuracy_within", "accuracy_cross", "gap"]].mean()
summary.columns = ["Within-subject", "Cross-subject", "Generalisation Gap"]
summary = summary.sort_values("Within-subject", ascending=False)

print("=== Generalisation Gap (within - cross accuracy) ===")
print(summary.round(3))
summary.to_csv("generalisation_gap.csv")
print("\nSaved generalisation_gap.csv")
