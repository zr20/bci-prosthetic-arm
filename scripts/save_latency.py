import pandas as pd

data = {
    "model":    ["CSP+LDA", "EEGNet", "ShallowConvNet", "Transformer"],
    "mean_ms":  [0.21,  0.35,  0.31,  15.01],
    "std_ms":   [0.17,  0.15,  0.08,  0.96],
    "min_ms":   [0.16,  0.28,  0.27,  13.04],
    "max_ms":   [1.68,  1.24,  0.98,  16.82],
    "pass_100ms":[True, True,  True,  True],
}
pd.DataFrame(data).to_csv("latency_results.csv", index=False)
print("Saved latency_results.csv")

