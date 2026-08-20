import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from braindecode.models import EEGNetv4
from sklearn.metrics import accuracy_score
import pandas as pd

DATA_DIR = "processed_data_2b"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SUBJECTS = list(range(1, 10))

# ── Viterbi decoder ───────────────────────────────────────────────────────────
def viterbi_decode(probs, transition_prob=0.1, n_classes=2):
    """
    probs: (T, n_classes) softmax probabilities over T trials
    transition_prob: probability of switching intent between trials
    Returns: sequence of decoded class labels
    """
    T = len(probs)
    # Transition matrix: stay = 1 - transition_prob, switch = transition_prob / (n_classes-1)
    trans = np.full((n_classes, n_classes), transition_prob / (n_classes - 1))
    np.fill_diagonal(trans, 1 - transition_prob)
    log_trans = np.log(trans + 1e-10)

    log_probs = np.log(probs + 1e-10)
    dp = np.full((T, n_classes), -np.inf)
    backptr = np.zeros((T, n_classes), dtype=int)

    dp[0] = log_probs[0] + np.log(1.0 / n_classes)
    for t in range(1, T):
        for s in range(n_classes):
            scores = dp[t-1] + log_trans[:, s]
            backptr[t, s] = np.argmax(scores)
            dp[t, s] = scores[backptr[t, s]] + log_probs[t, s]

    # Backtrack
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(dp[-1])
    for t in range(T-2, -1, -1):
        path[t] = backptr[t+1, path[t+1]]
    return path

# ── Train EEGNet and get softmax probabilities ────────────────────────────────
def train_model(X_train, y_train, n_times, epochs=30):
    model = EEGNetv4(n_chans=3, n_outputs=2, n_times=n_times).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train * 1e6, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        ), batch_size=128, shuffle=True
    )
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
    return model

def get_probs(model, X_test):
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X_test * 1e6, dtype=torch.float32).to(DEVICE)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs

results = []

print("=== Viterbi vs Argmax decoding (within-subject, EEGNet) ===")
for subj in SUBJECTS:
    d = np.load(f"{DATA_DIR}/within_subject_{subj}.npz")
    X_train, y_train = d["X_train"], d["y_train"]
    X_test, y_test = d["X_test"], d["y_test"]
    n_times = X_train.shape[2]

    model = train_model(X_train, y_train, n_times)
    probs = get_probs(model, X_test)

    # Standard argmax
    argmax_preds = np.argmax(probs, axis=1)
    argmax_acc = accuracy_score(y_test, argmax_preds)

    # Viterbi
    viterbi_preds = viterbi_decode(probs, transition_prob=0.1)
    viterbi_acc = accuracy_score(y_test, viterbi_preds)

    print(f"Subject {subj}: argmax={argmax_acc:.3f}  viterbi={viterbi_acc:.3f}  "
          f"delta={viterbi_acc - argmax_acc:+.3f}")
    results.append({
        "subject": subj,
        "argmax_acc": argmax_acc,
        "viterbi_acc": viterbi_acc,
        "delta": viterbi_acc - argmax_acc
    })

df = pd.DataFrame(results)
df.to_csv("viterbi_results.csv", index=False)
print(f"\nMean argmax: {df['argmax_acc'].mean():.3f}")
print(f"Mean viterbi: {df['viterbi_acc'].mean():.3f}")
print(f"Mean delta: {df['delta'].mean():+.3f}")
print("Saved viterbi_results.csv")
