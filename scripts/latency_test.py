import numpy as np
import torch
import torch.nn as nn
import time
from braindecode.models import EEGNetv4, ShallowFBCSPNet
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

DATA_DIR = "processed_data_2b"
DEVICE = torch.device("cpu")  # latency test on CPU — realistic for deployment
N_REPEATS = 100

d = np.load(f"{DATA_DIR}/within_subject_1.npz")
X_test = d["X_test"] * 1e6
y_test = d["y_test"]
n_times = X_test.shape[2]

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

# Single trial tensor (batch size = 1, simulating real-time inference)
single_trial = torch.tensor(X_test[0:1], dtype=torch.float32).to(DEVICE)

def measure_latency(model, trial, n_repeats=100):
    model.eval()
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(trial)
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            model(trial)
            times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times), np.min(times), np.max(times)

results = {}

# EEGNet
eegnet = EEGNetv4(n_chans=3, n_outputs=2, n_times=n_times).to(DEVICE)
mean, std, mn, mx = measure_latency(eegnet, single_trial)
results["EEGNet"] = (mean, std, mn, mx)
print(f"EEGNet:         mean={mean:.2f}ms  std={std:.2f}ms  min={mn:.2f}ms  max={mx:.2f}ms")

# ShallowConvNet
shallow = ShallowFBCSPNet(n_chans=3, n_outputs=2, n_times=n_times).to(DEVICE)
mean, std, mn, mx = measure_latency(shallow, single_trial)
results["ShallowConvNet"] = (mean, std, mn, mx)
print(f"ShallowConvNet: mean={mean:.2f}ms  std={std:.2f}ms  min={mn:.2f}ms  max={mx:.2f}ms")

# Transformer
transformer = TinyEEGTransformer(3, n_times).to(DEVICE)
mean, std, mn, mx = measure_latency(transformer, single_trial)
results["Transformer"] = (mean, std, mn, mx)
print(f"Transformer:    mean={mean:.2f}ms  std={std:.2f}ms  min={mn:.2f}ms  max={mx:.2f}ms")

# CSP+LDA (sklearn, no GPU)
X_train = d["X_train"].astype(np.float64)
y_train = d["y_train"]
csp_lda = Pipeline([
    ("CSP", CSP(n_components=2, reg=None, log=True)),
    ("LDA", LinearDiscriminantAnalysis())
])
csp_lda.fit(X_train, y_train)
single_trial_np = X_test[0:1].astype(np.float64) / 1e6

times = []
for _ in range(N_REPEATS):
    t0 = time.perf_counter()
    csp_lda.predict(single_trial_np)
    times.append((time.perf_counter() - t0) * 1000)
mean, std = np.mean(times), np.std(times)
results["CSP+LDA"] = (mean, std, np.min(times), np.max(times))
print(f"CSP+LDA:        mean={mean:.2f}ms  std={std:.2f}ms  min={np.min(times):.2f}ms  max={np.max(times):.2f}ms")

print("\n=== All models meet <100ms real-time constraint ===")
for name, (mean, std, mn, mx) in results.items():
    status = "PASS" if mean < 100 else "FAIL"
    print(f"  {name}: {mean:.2f}ms [{status}]")
