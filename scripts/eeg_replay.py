import numpy as np
import torch
import torch.nn as nn
import mujoco
import time
import os
import sys

sys.path.insert(0, os.path.expanduser("~/BioSimRL"))
from intent_mapper import IntentMapper

DATA_DIR = os.path.expanduser("~/processed_data_2b")
d = np.load(f"{DATA_DIR}/within_subject_4.npz")
X_test = d["X_test"] * 1e6
y_test = d["y_test"]
n_times = X_test.shape[2]
print(f"Loaded {len(X_test)} test trials from Subject 4")

from braindecode.models import EEGNetv4
DEVICE = torch.device("cpu")

X_train = d["X_train"] * 1e6
y_train = d["y_train"]

model = EEGNetv4(n_chans=3, n_outputs=2, n_times=n_times).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("Training EEGNet decoder (30 epochs)...")
from torch.utils.data import DataLoader, TensorDataset
loader = DataLoader(
    TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    ), batch_size=64, shuffle=True
)
model.train()
for epoch in range(30):
    for xb, yb in loader:
        optimizer.zero_grad()
        criterion(model(xb), yb).backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"  Epoch {epoch+1}/30 done")
model.eval()
print("Decoder ready.")

XML_PATH = os.path.expanduser("~/BioSimRL/BioSim_Mujoco/biosim.xml")
mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
mj_data = mujoco.MjData(mj_model)
renderer = mujoco.Renderer(mj_model, height=480, width=640)

N_TRIALS = min(30, len(X_test))
STEPS_PER_TRIAL = 50
frames = []
latencies = []
successes = []

print(f"\nStarting EEG replay: {N_TRIALS} trials x {STEPS_PER_TRIAL} steps")
print("-" * 60)

for trial_idx in range(N_TRIALS):
    # Reset mapper at start of each trial so window doesn't bleed across trials
    mapper = IntentMapper(smoothing_window=3)

    eeg_trial = torch.tensor(X_test[trial_idx:trial_idx+1], dtype=torch.float32)
    true_label = y_test[trial_idx]

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(eeg_trial)
        pred = int(logits.argmax(dim=1).item())
    latency_ms = (time.perf_counter() - t0) * 1000

    intent = mapper.update(pred)
    command = mapper.get_command()
    latencies.append(latency_ms)

    true_intent = "Reach-Left" if true_label == 0 else "Reach-Right"
    correct = (intent == true_intent)
    successes.append(correct)

    print(f"Trial {trial_idx+1:02d}: EEG→{intent:12s} "
          f"(true: {true_intent:12s}) "
          f"{'✓' if correct else '✗'}  "
          f"latency={latency_ms:.1f}ms")

    mujoco.mj_resetData(mj_model, mj_data)
    for step in range(STEPS_PER_TRIAL):
        mj_data.ctrl[:] = command
        mujoco.mj_step(mj_model, mj_data)
        if step % 5 == 0:
            renderer.update_scene(mj_data)
            frames.append(renderer.render().copy())

print("-" * 60)
print(f"\nResults:")
print(f"  Trials: {N_TRIALS}")
print(f"  Success rate: {np.mean(successes)*100:.1f}%")
print(f"  Mean latency: {np.mean(latencies):.2f}ms")
print(f"  Max latency:  {np.max(latencies):.2f}ms")
print(f"  All latencies <100ms: {all(l < 100 for l in latencies)}")
print(f"  Frames captured: {len(frames)}")

print("\nSaving demo video...")
try:
    import imageio
    video_path = os.path.expanduser("~/prosthetic_demo.mp4")
    imageio.mimsave(video_path, frames, fps=10)
    print(f"Saved demo video: {video_path}")
except ImportError:
    print("imageio not installed — saving frames as PNG instead")
    os.makedirs(os.path.expanduser("~/demo_frames"), exist_ok=True)
    from PIL import Image
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(
            os.path.expanduser(f"~/demo_frames/frame_{i:04d}.png")
        )
    print(f"Saved {len(frames)} frames to ~/demo_frames/")

import pandas as pd
pd.DataFrame({
    "trial": range(N_TRIALS),
    "latency_ms": latencies,
    "success": successes
}).to_csv(os.path.expanduser("~/replay_results.csv"), index=False)
print("Saved replay_results.csv")

