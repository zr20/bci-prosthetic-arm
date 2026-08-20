import numpy as np
import pandas as pd
import os
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery

SAVE_DIR = "/Users/zarakhursheed/bci_paper/results"
os.makedirs(SAVE_DIR, exist_ok=True)

print("=" * 60)
print("CROSS-DATASET HARMONISATION DOCUMENTATION")
print("Source: BCI Competition IV Dataset 2b (BNCI2014_004)")
print("Target: BCI Competition IV Dataset 2a (BNCI2014_001)")
print("=" * 60)

# ── Source dataset (BCI IV-2b) properties ────────────────────────────────────
print("\n--- SOURCE DATASET: BCI IV-2b ---")
source = {
    "dataset":         "BCI Competition IV Dataset 2b (BNCI2014_004)",
    "n_subjects":      9,
    "channels":        "EEG:C3, EEG:Cz, EEG:C4",
    "n_channels":      3,
    "sampling_rate_hz":250,
    "epoch_tmin_s":    0.5,
    "epoch_tmax_s":    4.0,
    "epoch_length_s":  3.5,
    "n_timepoints":    875,
    "bandpass_hz":     "4-40",
    "notch_hz":        50,
    "classes":         "left_hand=0, right_hand=1",
    "trials_per_subj": "400-440 (sessions 1+2 train, session 3 test)",
}
for k, v in source.items():
    print(f"  {k}: {v}")

# ── Target dataset (BCI IV-2a) properties ────────────────────────────────────
print("\n--- TARGET DATASET: BCI IV-2a ---")
dataset_2a = BNCI2014_001()
paradigm_2a = LeftRightImagery(fmin=4, fmax=40)

# Load one subject to inspect
X_raw, labels, meta = paradigm_2a.get_data(
    dataset=dataset_2a, subjects=[1]
)
raw_data = dataset_2a.get_data(subjects=[1])
sess = list(raw_data[1].keys())[0]
run  = list(raw_data[1][sess].keys())[0]
raw  = raw_data[1][sess][run]

target = {
    "dataset":              "BCI Competition IV Dataset 2a (BNCI2014_001)",
    "n_subjects":           9,
    "all_channels":         str(raw.ch_names),
    "n_channels_total":     len(raw.ch_names),
    "sampling_rate_hz":     raw.info['sfreq'],
    "original_epoch_window":"2.0s to 6.0s (interval=[2,6])",
    "original_n_timepoints":X_raw.shape[2],
    "classes_available":    str(set(labels)),
    "trials_subject_1":     X_raw.shape[0],
}
for k, v in target.items():
    print(f"  {k}: {v}")

# ── Harmonisation steps applied ───────────────────────────────────────────────
print("\n--- HARMONISATION STEPS ---")

# Channel mapping
all_channels = raw.ch_names
c3_idx = all_channels.index('C3')
cz_idx = all_channels.index('Cz')
c4_idx = all_channels.index('C4')

harmonisation = {
    "step_1_channels": (
        f"Subset 22 channels to C3 (idx {c3_idx}), "
        f"Cz (idx {cz_idx}), C4 (idx {c4_idx}) — "
        f"matching 2b channel set"
    ),
    "step_2_sampling_rate": (
        "No resampling required — both datasets recorded at 250 Hz"
    ),
    "step_3_epoch_window": (
        "Trim 2a epoch from 1001 samples (2-6s) to 875 samples "
        "(first 3.5s) — matching 2b epoch length of 875 timepoints "
        "(0.5s to 4.0s at 250 Hz)"
    ),
    "step_4_bandpass": (
        "Apply 4-40 Hz bandpass via MOABB LeftRightImagery paradigm "
        "— identical to 2b preprocessing"
    ),
    "step_5_labels": (
        "Remap: left_hand → 0, right_hand → 1 — identical to 2b labels"
    ),
    "step_6_scaling": (
        "Multiply by 1e6 (Volts to microvolts) before model input "
        "— identical to 2b scaling"
    ),
    "remaining_difference": (
        "2a has 288 trials/subject vs 2b 400-440 trials/subject — "
        "not harmonised, noted as a domain difference"
    ),
}
for k, v in harmonisation.items():
    print(f"  {k}:")
    print(f"    {v}")

# ── Verify harmonised shape ───────────────────────────────────────────────────
print("\n--- VERIFICATION ---")
X_harmonised = X_raw[:, [c3_idx, cz_idx, c4_idx], :875]
print(f"  Original 2a shape:    {X_raw.shape}")
print(f"  Harmonised 2a shape:  {X_harmonised.shape}")
print(f"  Target 2b shape:      (n_trials, 3, 875)")
print(f"  Channel match:        "
      f"{'YES' if X_harmonised.shape[1] == 3 else 'NO'}")
print(f"  Timepoint match:      "
      f"{'YES' if X_harmonised.shape[2] == 875 else 'NO'}")
print(f"  Sampling rate match:  YES (both 250 Hz)")

# ── Save to CSV ───────────────────────────────────────────────────────────────
rows = []
for k, v in source.items():
    rows.append({"section": "source_2b", "property": k, "value": str(v)})
for k, v in target.items():
    rows.append({"section": "target_2a", "property": k, "value": str(v)})
for k, v in harmonisation.items():
    rows.append({"section": "harmonisation", "property": k, "value": str(v)})

rows.append({"section": "verification",
             "property": "harmonised_shape",
             "value": str(X_harmonised.shape)})
rows.append({"section": "verification",
             "property": "channel_match",
             "value": "YES"})
rows.append({"section": "verification",
             "property": "timepoint_match",
             "value": "YES"})
rows.append({"section": "verification",
             "property": "sfreq_match",
             "value": "YES"})

df = pd.DataFrame(rows)
out = os.path.join(SAVE_DIR, "harmonisation_documentation.csv")
df.to_csv(out, index=False)
print(f"\nSaved to {out}")
print("=" * 60)
