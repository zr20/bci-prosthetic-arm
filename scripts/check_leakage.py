import numpy as np
import hashlib

DATA_DIR = "/Users/zarakhursheed/processed_data_2b"

for subj in range(1, 10):
    d = np.load(f"{DATA_DIR}/within_subject_{subj}.npz")
    X_train = d["X_train"]
    X_test  = d["X_test"]

    # Hash the exact bytes of each trial — truly unique fingerprint
    def hash_trial(t):
        return hashlib.md5(t.tobytes()).hexdigest()

    train_hashes = set(hash_trial(t) for t in X_train)
    test_hashes  = [hash_trial(t) for t in X_test]

    overlap = sum(1 for h in test_hashes if h in train_hashes)
    status = "CLEAN" if overlap == 0 else "LEAKAGE DETECTED"
    print(f"Subject {subj}: train={len(X_train)}, "
          f"test={len(X_test)}, "
          f"overlap={overlap} — {status}")
