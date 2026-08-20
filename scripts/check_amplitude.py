import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery

DATA_DIR = "/Users/zarakhursheed/processed_data_2b"

print("=== BCI IV-2b amplitude statistics (after x1e6 scaling) ===")
for subj in [1, 4, 9]:
    d = np.load(f"{DATA_DIR}/within_subject_{subj}.npz")
    X = d["X_train"] * 1e6
    print(f"Subject {subj}: mean={X.mean():.2f}, "
          f"std={X.std():.2f}, "
          f"min={X.min():.2f}, "
          f"max={X.max():.2f}")

print("\n=== BCI IV-2a amplitude statistics (after x1e6 scaling) ===")
dataset_2a = BNCI2014_001()
paradigm_2a = LeftRightImagery(fmin=4, fmax=40)
C3_IDX, CZ_IDX, C4_IDX = 7, 9, 11

for subj in [1, 4, 9]:
    X, labels, meta = paradigm_2a.get_data(
        dataset=dataset_2a, subjects=[subj]
    )
    X_subset = X[:, [C3_IDX, CZ_IDX, C4_IDX], :876] * 1e6
    print(f"Subject {subj}: mean={X_subset.mean():.2f}, "
          f"std={X_subset.std():.2f}, "
          f"min={X_subset.min():.2f}, "
          f"max={X_subset.max():.2f}")
