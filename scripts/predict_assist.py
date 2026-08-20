import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from braindecode.models import EEGNetv4
from sklearn.metrics import accuracy_score
import pandas as pd

DATA_DIR = "processed_data_2b"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

d = np.load(f"{DATA_DIR}/within_subject_4.npz")
X_train = d["X_train"] * 1e6
y_train = d["y_train"]
X_test = d["X_test"] * 1e6
y_test = d["y_test"]
n_times = X_train.shape[2]

print("Training base EEGNet encoder...")
base_model = EEGNetv4(n_chans=3, n_outputs=2, n_times=n_times).to(DEVICE)
optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

loader = DataLoader(
    TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    ), batch_size=128, shuffle=True
)
base_model.train()
for epoch in range(30):
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        criterion(base_model(xb), yb).backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"  Epoch {epoch+1}/30 done")
print("Base encoder ready.")

# Hook drop_2 — rich intermediate features before classification head
embeddings_store = []
def hook_fn(module, input, output):
    embeddings_store.append(output.detach().cpu())

hook = base_model.drop_2.register_forward_hook(hook_fn)

base_model.eval()
with torch.no_grad():
    base_model(torch.tensor(X_train, dtype=torch.float32).to(DEVICE))
embeddings_train = torch.cat(embeddings_store, dim=0)
embeddings_train = embeddings_train.view(embeddings_train.shape[0], -1)

embeddings_store.clear()
with torch.no_grad():
    base_model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE))
embeddings_test = torch.cat(embeddings_store, dim=0)
embeddings_test = embeddings_test.view(embeddings_test.shape[0], -1)
hook.remove()

emb_dim = embeddings_train.shape[1]
print(f"Embedding dimension: {emb_dim}")

class ForecastHead(nn.Module):
    def __init__(self, emb_dim, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        return self.net(x)

# Shifted pairs: input=embedding[t], target=label[t+1]
X_fore = embeddings_train[:-1]
y_fore = torch.tensor(y_train[1:], dtype=torch.long)

forecast_model = ForecastHead(emb_dim).to(DEVICE)
opt_fore = torch.optim.Adam(forecast_model.parameters(), lr=1e-3, weight_decay=1e-4)
fore_loader = DataLoader(
    TensorDataset(X_fore, y_fore), batch_size=32, shuffle=True
)

print("Training forecast head...")
forecast_model.train()
for epoch in range(50):
    for xb, yb in fore_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt_fore.zero_grad()
        criterion(forecast_model(xb), yb).backward()
        opt_fore.step()
    if (epoch+1) % 10 == 0:
        print(f"  Epoch {epoch+1}/50 done")

forecast_model.eval()
X_fore_test = embeddings_test[:-1].to(DEVICE)
y_fore_test = y_test[1:]

with torch.no_grad():
    fore_preds = forecast_model(X_fore_test).argmax(dim=1).cpu().numpy()

fore_acc = accuracy_score(y_fore_test, fore_preds)

base_model.eval()
with torch.no_grad():
    reactive_preds = base_model(
        torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    ).argmax(dim=1).cpu().numpy()
reactive_acc = accuracy_score(y_test, reactive_preds)

print(f"\n=== Predict-to-Assist Results (Subject 4) ===")
print(f"Reactive control accuracy:  {reactive_acc:.3f}")
print(f"Forecast head accuracy:     {fore_acc:.3f}")
print(f"Forecast gain:              {fore_acc - reactive_acc:+.3f}")

TRIAL_DURATION = 4.5
PRE_ARM_FRACTION = 0.5
n_sim = min(30, len(y_test) - 1)

reactive_times = []
predictive_times = []
correct_reactive = 0
correct_predictive = 0

for i in range(n_sim):
    true_label = y_test[i]
    react_pred = reactive_preds[i]
    fore_pred = fore_preds[i] if i < len(fore_preds) else react_pred

    reactive_times.append(TRIAL_DURATION)
    if react_pred == true_label:
        correct_reactive += 1

    if fore_pred == true_label:
        predictive_times.append(TRIAL_DURATION * PRE_ARM_FRACTION)
        correct_predictive += 1
    else:
        predictive_times.append(TRIAL_DURATION)

print(f"\n=== Simulated task timing (30 trials) ===")
print(f"Reactive:   mean time={np.mean(reactive_times):.2f}s  "
      f"success={correct_reactive/n_sim*100:.1f}%")
print(f"Predictive: mean time={np.mean(predictive_times):.2f}s  "
      f"success={correct_predictive/n_sim*100:.1f}%")
print(f"Time saving: {np.mean(reactive_times)-np.mean(predictive_times):.2f}s "
      f"per trial "
      f"({(1-np.mean(predictive_times)/np.mean(reactive_times))*100:.1f}% faster)")

pd.DataFrame({
    "metric": ["reactive_acc", "forecast_acc", "forecast_gain",
               "reactive_time_s", "predictive_time_s", "time_saving_s"],
    "value": [reactive_acc, fore_acc, fore_acc - reactive_acc,
              np.mean(reactive_times), np.mean(predictive_times),
              np.mean(reactive_times) - np.mean(predictive_times)]
}).to_csv("predict_assist_results.csv", index=False)
print("\nSaved predict_assist_results.csv")
