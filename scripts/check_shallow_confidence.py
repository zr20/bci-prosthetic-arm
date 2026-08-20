import sys
import os
sys.path.insert(0, '/Users/zarakhursheed/bci_paper/configs')
from config import CONFIG

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from braindecode.models import ShallowFBCSPNet

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DATA_DIR = CONFIG["data_dir"]

def make_loader(X, y, batch_size=128, shuffle=True):
    return DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        ), batch_size=batch_size, shuffle=shuffle
    )

def train_model(X_train, y_train, n_times):
    torch.manual_seed(SEED)
    model = ShallowFBCSPNet(n_chans=3, n_outputs=2,
                             n_times=n_times).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss()
    loader = make_loader(X_train, y_train)
    model.train()
    for _ in range(CONFIG["epochs"]):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
    return model

print("=== ShallowConvNet Confidence Distribution (within-subject) ===\n")
all_confidences = []

for subj in range(1, 10):
    d = np.load(f"{DATA_DIR}/within_subject_{subj}.npz")
    X_train = d["X_train"] * CONFIG["scale"]
    y_train = d["y_train"]
    X_test  = d["X_test"]  * CONFIG["scale"]
    y_test  = d["y_test"]

    model = train_model(X_train, y_train, X_train.shape[2])
    model.eval()
    with torch.no_grad():
        xb    = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        probs = torch.softmax(model(xb), dim=1).cpu().numpy()

    confs = probs.max(axis=1)
    all_confidences.extend(confs.tolist())

    print(f"Subject {subj}: "
          f"mean={confs.mean():.3f}, "
          f"min={confs.min():.3f}, "
          f"max={confs.max():.3f}, "
          f">0.55: {(confs>0.55).mean():.1%}, "
          f">0.60: {(confs>0.60).mean():.1%}, "
          f">0.65: {(confs>0.65).mean():.1%}, "
          f">0.70: {(confs>0.70).mean():.1%}, "
          f">0.75: {(confs>0.75).mean():.1%}, "
          f">0.80: {(confs>0.80).mean():.1%}")

all_confidences = np.array(all_confidences)
print(f"\nAll subjects combined:")
print(f"  Mean confidence:  {all_confidences.mean():.3f}")
print(f"  % trials > 0.55:  {(all_confidences>0.55).mean():.1%}")
print(f"  % trials > 0.60:  {(all_confidences>0.60).mean():.1%}")
print(f"  % trials > 0.65:  {(all_confidences>0.65).mean():.1%}")
print(f"  % trials > 0.70:  {(all_confidences>0.70).mean():.1%}")
print(f"  % trials > 0.75:  {(all_confidences>0.75).mean():.1%}")
print(f"  % trials > 0.80:  {(all_confidences>0.80).mean():.1%}")
