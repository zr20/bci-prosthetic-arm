import numpy as np
import os
import pandas as pd
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

DATA_DIR = "processed_data_2b"
SUBJECTS = list(range(1, 10))

def load_split(path):
    d = np.load(path)
    return d["X_train"], d["y_train"], d["X_test"], d["y_test"]

def run_csp_lda(X_train, y_train, X_test, y_test, n_components=2):
    # CSP expects float64
    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)

    clf = Pipeline([
        ("CSP", CSP(n_components=n_components, reg=None, log=True, norm_trace=False)),
        ("LDA", LinearDiscriminantAnalysis())
    ])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return acc, f1

results = []

# --- Within-subject ---
print("=== Within-subject ===")
for subj in SUBJECTS:
    path = f"{DATA_DIR}/within_subject_{subj}.npz"
    X_train, y_train, X_test, y_test = load_split(path)
    acc, f1 = run_csp_lda(X_train, y_train, X_test, y_test)
    print(f"Subject {subj}: acc={acc:.3f}, f1={f1:.3f}")
    results.append({"split": "within", "subject": subj, "accuracy": acc, "f1": f1})

# --- Cross-subject ---
print("\n=== Cross-subject (leave-one-out) ===")
for subj in SUBJECTS:
    path = f"{DATA_DIR}/cross_subject_holdout_{subj}.npz"
    X_train, y_train, X_test, y_test = load_split(path)
    acc, f1 = run_csp_lda(X_train, y_train, X_test, y_test)
    print(f"Holdout subject {subj}: acc={acc:.3f}, f1={f1:.3f}")
    results.append({"split": "cross", "subject": subj, "accuracy": acc, "f1": f1})

df = pd.DataFrame(results)
df.to_csv("csp_lda_results.csv", index=False)

print("\n=== Summary ===")
print(df.groupby("split")[["accuracy", "f1"]].mean())
print("\nFull results saved to csp_lda_results.csv")
