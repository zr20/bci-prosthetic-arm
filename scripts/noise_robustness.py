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

def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X_test * 1e6, dtype=torch.float32).to(DEVICE)
        preds = model(xb).argmax(dim=1).cpu().numpy()
    return accuracy_score(y_test, preds)

def add_channel_noise(X, noise_std=0.5):
    """Add Gaussian noise scaled relative to signal std"""
    signal_std = np.std(X)
    return X + np.random.randn(*X.shape) * noise_std * signal_std

def simulate_electrode_shift(X, shift_channels=1):
    """Simulate electrode displacement by permuting/attenuating channels"""
    X_shifted = X.copy()
    for i in range(shift_channels):
        ch = np.random.randint(0, X.shape[1])
        X_shifted[:, ch, :] *= 0.3  # attenuate shifted channel by 70%
    return X_shifted

NOISE_LEVELS = [0.0, 0.25, 0.5, 1.0, 2.0]
results = []

print("=== Noise robustness (EEGNet, within-subject) ===")
for subj in SUBJECTS:
    d = np.load(f"{DATA_DIR}/within_subject_{subj}.npz")
    X_train, y_train = d["X_train"], d["y_train"]
    X_test, y_test = d["X_test"], d["y_test"]
    n_times = X_train.shape[2]

    model = train_model(X_train, y_train, n_times)
    clean_acc = evaluate(model, X_test, y_test)

    print(f"\nSubject {subj} (clean acc: {clean_acc:.3f})")

    for noise_std in NOISE_LEVELS:
        # Channel noise
        X_noisy = add_channel_noise(X_test, noise_std)
        noisy_acc = evaluate(model, X_noisy, y_test)

        # Electrode shift
        X_shifted = simulate_electrode_shift(X_test, shift_channels=1)
        shifted_acc = evaluate(model, X_shifted, y_test)

        drop_noisy = clean_acc - noisy_acc
        drop_shifted = clean_acc - shifted_acc

        print(f"  noise={noise_std:.2f}: channel_noise_acc={noisy_acc:.3f} "
              f"(drop={drop_noisy:+.3f})  "
              f"electrode_shift_acc={shifted_acc:.3f} "
              f"(drop={drop_shifted:+.3f})")

        results.append({
            "subject": subj,
            "noise_level": noise_std,
            "clean_acc": clean_acc,
            "channel_noise_acc": noisy_acc,
            "electrode_shift_acc": shifted_acc,
            "drop_channel_noise": drop_noisy,
            "drop_electrode_shift": drop_shifted
        })

df = pd.DataFrame(results)
df.to_csv("noise_robustness_results.csv", index=False)

print("\n=== Summary: mean accuracy by noise level ===")
summary = df.groupby("noise_level")[
    ["clean_acc", "channel_noise_acc", "electrode_shift_acc"]
].mean()
print(summary.round(3))
print("\nSaved noise_robustness_results.csv")
