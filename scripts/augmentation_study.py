import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from braindecode.models import EEGNetv4
from sklearn.metrics import accuracy_score
import pandas as pd

DATA_DIR = "processed_data_2b"
SUBJECTS = list(range(1, 10))
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def time_shift(X, max_shift=25):
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(X, shift, axis=-1)

def gaussian_noise(X, std=0.1):
    return X + np.random.randn(*X.shape) * std

def frequency_mask(X, max_mask=20):
    import numpy.fft as fft
    Xf = fft.rfft(X, axis=-1)
    f0 = np.random.randint(0, Xf.shape[-1] - max_mask)
    Xf[..., f0:f0+max_mask] = 0
    return fft.irfft(Xf, n=X.shape[-1], axis=-1)

def augment(X, aug_type):
    X = X.copy()
    if aug_type == "none":
        return X
    elif aug_type == "time_shift":
        return np.stack([time_shift(x) for x in X])
    elif aug_type == "gaussian_noise":
        return gaussian_noise(X)
    elif aug_type == "freq_mask":
        return np.stack([frequency_mask(x) for x in X])
    elif aug_type == "all":
        X = np.stack([time_shift(x) for x in X])
        X = gaussian_noise(X)
        X = np.stack([frequency_mask(x) for x in X])
        return X

def make_loader(X, y, batch_size=128, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)

def train_and_eval(X_train, y_train, X_test, y_test, aug_type, epochs=30):
    X_train_aug = augment(X_train * 1e6, aug_type)
    X_test_s = X_test * 1e6
    n_times = X_train.shape[2]

    model = EEGNetv4(n_chans=3, n_outputs=2, n_times=n_times).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_loader = make_loader(X_train_aug, y_train)
    test_loader = make_loader(X_test_s, y_test, shuffle=False)

    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            preds.extend(model(xb).argmax(dim=1).cpu().numpy())
            trues.extend(yb.numpy())
    return accuracy_score(trues, preds)

AUG_TYPES = ["none", "time_shift", "gaussian_noise", "freq_mask", "all"]
results = []

for aug in AUG_TYPES:
    print(f"\n=== Augmentation: {aug} ===")
    for subj in SUBJECTS:
        d = np.load(f"{DATA_DIR}/cross_subject_holdout_{subj}.npz")
        acc = train_and_eval(d["X_train"], d["y_train"], d["X_test"], d["y_test"], aug)
        print(f"  Holdout {subj}: acc={acc:.3f}")
        results.append({"augmentation": aug, "subject": subj, "accuracy": acc})

df = pd.DataFrame(results)
df.to_csv("augmentation_results.csv", index=False)

print("\n=== Summary: mean accuracy by augmentation ===")
summary = df.groupby("augmentation")["accuracy"].mean().sort_values(ascending=False)
print(summary.round(3))
print("\nSaved augmentation_results.csv")
