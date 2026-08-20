import numpy as np
import torch
from braindecode.models import EEGNetv4
from sklearn.metrics import accuracy_score

DATA_DIR = "processed_data_2b"
d = np.load(f"{DATA_DIR}/within_subject_4.npz")
X_test = d["X_test"] * 1e6
y_test = d["y_test"]
n_times = X_test.shape[2]

model = EEGNetv4(n_chans=3, n_outputs=2, n_times=n_times)
model.load_state_dict(torch.load(
    "EEGNet_Subject4_WithinSubject_Weights.pt",
    map_location="cpu"
))
model.eval()

with torch.no_grad():
    preds = model(torch.tensor(X_test, dtype=torch.float32)).argmax(dim=1).numpy()

acc = accuracy_score(y_test, preds)
print(f"EEGNet saved weights accuracy: {acc:.3f} ({acc*100:.1f}%)")

