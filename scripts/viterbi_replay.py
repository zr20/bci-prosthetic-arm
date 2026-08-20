import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from braindecode.models import EEGNetv4
from sklearn.metrics import accuracy_score
import pandas as pd

DATA_DIR = "processed_data_2b"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def viterbi_decode(probs, transition_prob=0.1, n_classes=2):
    T = len(probs)
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
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(dp[-1])
    for t in range(T-2, -1, -1):
        path[t] = backptr[t+1, path[t+1]]
    return path

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

def get_probs(model, X):
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X * 1e6, dtype=torch.float32).to(DEVICE)
        return torch.softmax(model(xb), dim=1).cpu().numpy()

# Use Subject 4 — best signal, most meaningful temporal sequence
subj = 4
d = np.load(f"{DATA_DIR}/within_subject_{subj}.npz")
X_train, y_train = d["X_train"], d["y_train"]
X_test, y_test = d["X_test"], d["y_test"]
n_times = X_train.shape[2]

print(f"Training EEGNet on Subject {subj}...")
model = train_model(X_train, y_train, n_times)
probs = get_probs(model, X_test)

argmax_preds = np.argmax(probs, axis=1)
argmax_acc = accuracy_score(y_test, argmax_preds)

results = []
for trans_prob in [0.05, 0.1, 0.2, 0.3]:
    viterbi_preds = viterbi_decode(probs, transition_prob=trans_prob)
    viterbi_acc = accuracy_score(y_test, viterbi_preds)
    delta = viterbi_acc - argmax_acc
    print(f"  transition_prob={trans_prob}: viterbi={viterbi_acc:.3f}  "
          f"argmax={argmax_acc:.3f}  delta={delta:+.3f}")
    results.append({
        "subject": subj,
        "transition_prob": trans_prob,
        "argmax_acc": argmax_acc,
        "viterbi_acc": viterbi_acc,
        "delta": delta
    })

df = pd.DataFrame(results)
df.to_csv("viterbi_results.csv", index=False)
print(f"\nArgmax baseline: {argmax_acc:.3f}")
print(f"Best Viterbi: {df['viterbi_acc'].max():.3f} "
      f"(transition_prob={df.loc[df['viterbi_acc'].idxmax(), 'transition_prob']})")
print("Saved viterbi_results.csv")
