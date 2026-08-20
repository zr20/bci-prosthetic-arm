import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from braindecode.models import EEGNetv4, ShallowFBCSPNet

DATA_DIR = "processed_data_2b"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

d = np.load(f"{DATA_DIR}/within_subject_4.npz")
X_train = d["X_train"] * 1e6
y_train = d["y_train"]
n_times = X_train.shape[2]

def train(model, epochs=30):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        ), batch_size=128, shuffle=True
    )
    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb.to(DEVICE)), yb.to(DEVICE)).backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} done")
    return model

print("Training EEGNet...")
eegnet = EEGNetv4(n_chans=3, n_outputs=2, n_times=n_times)
eegnet = train(eegnet)
torch.save(eegnet.state_dict(), "EEGNet_Subject4_WithinSubject_Weights.pt")
print("Saved EEGNet_Subject4_WithinSubject_Weights.pt")

print("Training ShallowConvNet...")
shallow = ShallowFBCSPNet(n_chans=3, n_outputs=2, n_times=n_times)
shallow = train(shallow)
torch.save(shallow.state_dict(), "ShallowConvNet_Subject4_WithinSubject_Weights.pt")
print("Saved ShallowConvNet_Subject4_WithinSubject_Weights.pt")

print("Done.")
