import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from braindecode.models import EEGNetv4, ShallowFBCSPNet
from sklearn.metrics import accuracy_score
import pandas as pd

DATA_DIR = "processed_data_2b"
SUBJECTS = list(range(1, 10))
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SHOTS = [10, 20, 40, 80]  # number of trials for adaptation (approx 1-5 min)

class TinyEEGTransformer(nn.Module):
    def __init__(self, n_chans, n_times, d_model=32, n_heads=4, n_layers=2, n_classes=2):
        super().__init__()
        self.input_proj = nn.Linear(n_chans, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, n_times, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=64,
            dropout=0.2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_proj(x) + self.pos_embedding
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)

def get_model(model_name, n_times):
    if model_name == "EEGNet":
        return EEGNetv4(n_chans=3, n_outputs=2, n_times=n_times)
    elif model_name == "ShallowConvNet":
        return ShallowFBCSPNet(n_chans=3, n_outputs=2, n_times=n_times)
    else:
        return TinyEEGTransformer(3, n_times)

def freeze_encoder(model, model_name):
    if model_name == "EEGNet":
        for name, param in model.named_parameters():
            if "final_layer" not in name:
                param.requires_grad = False
    elif model_name == "ShallowConvNet":
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    else:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

def train_cross_subject(model, X_train, y_train, epochs=30):
    X_train = X_train * 1e6
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
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
    return model

def few_shot_adapt(model, X_adapt, y_adapt, n_shots, epochs=10):
    idx = np.random.choice(len(X_adapt), min(n_shots, len(X_adapt)), replace=False)
    X_s = (X_adapt[idx] * 1e6)
    y_s = y_adapt[idx]
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_s, dtype=torch.float32),
            torch.tensor(y_s, dtype=torch.long)
        ), batch_size=min(32, n_shots), shuffle=True
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
    X_test = X_test * 1e6
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        preds = model(xb).argmax(dim=1).cpu().numpy()
    return accuracy_score(y_test, preds)

results = []

for model_name in ["EEGNet", "ShallowConvNet"]:
    print(f"\n=== {model_name} few-shot personalisation ===")
    for test_subj in SUBJECTS:
        train_subjs = [s for s in SUBJECTS if s != test_subj]
        d_cross = np.load(f"{DATA_DIR}/cross_subject_holdout_{test_subj}.npz")
        X_train, y_train = d_cross["X_train"], d_cross["y_train"]
        X_test, y_test = d_cross["X_test"], d_cross["y_test"]

        n_times = X_train.shape[2]
        model = get_model(model_name, n_times)
        model = train_cross_subject(model, X_train, y_train)

        # Zero-shot (no adaptation)
        zero_shot_acc = evaluate(model, X_test, y_test)

        # Few-shot adaptation
        for n_shots in SHOTS:
            import copy
            adapted = copy.deepcopy(model)
            freeze_encoder(adapted, model_name)
            adapted = few_shot_adapt(adapted, X_test, y_test, n_shots)
            acc = evaluate(adapted, X_test, y_test)
            print(f"  Subject {test_subj} | {n_shots} shots: {acc:.3f} (zero-shot: {zero_shot_acc:.3f})")
            results.append({
                "model": model_name,
                "subject": test_subj,
                "n_shots": n_shots,
                "zero_shot_acc": zero_shot_acc,
                "few_shot_acc": acc,
                "gain": acc - zero_shot_acc
            })

df = pd.DataFrame(results)
df.to_csv("few_shot_results.csv", index=False)

print("\n=== Summary: mean accuracy by model and shots ===")
summary = df.groupby(["model", "n_shots"])[["zero_shot_acc", "few_shot_acc", "gain"]].mean()
print(summary.round(3))
print("\nSaved few_shot_results.csv")
