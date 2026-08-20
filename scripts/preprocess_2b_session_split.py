import mne
import numpy as np
import os
from collections import defaultdict

DATA_DIR = "/Users/zarakhursheed/Downloads/NYU/DATASETS/Dataset 2b BCICIV/Training"
SAVE_DIR = "/Users/zarakhursheed/processed_data_2b"
os.makedirs(SAVE_DIR, exist_ok=True)

SUBJECTS = list(range(1, 10))
# Sessions 1 and 2 = train, Session 3 = test
# This is session-separated evaluation — no temporal leakage
TRAIN_SESSIONS = [1, 2]
TEST_SESSIONS  = [3]

def load_session(subject, session):
    fname = f"B{subject:02d}{session:02d}T.gdf"
    path  = os.path.join(DATA_DIR, fname)

    raw = mne.io.read_raw_gdf(path, preload=True, verbose=False)
    raw.pick(['EEG:C3', 'EEG:Cz', 'EEG:C4'])

    # Fit filter on this session's data only (no leakage)
    raw.filter(4, 40, fir_design='firwin', verbose=False)
    raw.notch_filter(50, verbose=False)

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    picks_id = {k: v for k, v in event_id.items() if k in ['769', '770']}

    if not picks_id:
        print(f"  No motor imagery events in S{subject} sess{session} — skipping")
        return None, None

    epochs = mne.Epochs(raw, events, event_id=picks_id,
                        tmin=0.5, tmax=4.0, baseline=None,
                        preload=True, verbose=False)

    X = epochs.get_data()   # (n_trials, 3, n_times)
    code_to_label = {
        v: (0 if k == '769' else 1)
        for k, v in picks_id.items()
    }
    y = np.array([code_to_label[e[2]] for e in epochs.events])
    return X, y

print("=== Session-separated preprocessing ===")
print("Train: sessions 1+2  |  Test: session 3")
print()

all_data = {}

for subj in SUBJECTS:
    print(f"Subject {subj}:")

    # ── Training data: sessions 1 + 2 ────────────────────────────────────
    X_train_list, y_train_list = [], []
    for sess in TRAIN_SESSIONS:
        X, y = load_session(subj, sess)
        if X is not None:
            X_train_list.append(X)
            y_train_list.append(y)
            print(f"  Session {sess} (train): {X.shape[0]} trials")

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    # ── Test data: session 3 only ─────────────────────────────────────────
    X_test_list, y_test_list = [], []
    for sess in TEST_SESSIONS:
        X, y = load_session(subj, sess)
        if X is not None:
            X_test_list.append(X)
            y_test_list.append(y)
            print(f"  Session {sess} (test):  {X.shape[0]} trials")

    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    all_data[subj] = {
        "X_train": X_train, "y_train": y_train,
        "X_test":  X_test,  "y_test":  y_test
    }

    print(f"  -> Total: train={len(y_train)}, test={len(y_test)}")

    # Save within-subject split (session-separated)
    np.savez(
        f"{SAVE_DIR}/within_subject_{subj}.npz",
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test
    )

print("\nWithin-subject session-separated splits saved.")

# ── Cross-subject: leave-one-subject-out ──────────────────────────────────
# Held-out subject's data is NEVER touched during training
# Model/hyperparameter selection must happen within training subjects only
print("\nBuilding cross-subject splits...")

for test_subj in SUBJECTS:
    train_subjs = [s for s in SUBJECTS if s != test_subj]

    X_train = np.concatenate(
        [all_data[s]["X_train"] for s in train_subjs] +
        [all_data[s]["X_test"]  for s in train_subjs],
        axis=0
    )
    y_train = np.concatenate(
        [all_data[s]["y_train"] for s in train_subjs] +
        [all_data[s]["y_test"]  for s in train_subjs],
        axis=0
    )

    # Held-out subject: test on session 3 only (same as within-subject test)
    X_test = all_data[test_subj]["X_test"]
    y_test = all_data[test_subj]["y_test"]

    np.savez(
        f"{SAVE_DIR}/cross_subject_holdout_{test_subj}.npz",
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test
    )
    print(f"  Holdout {test_subj}: "
          f"train={len(y_train)}, test={len(y_test)}")

print(f"\nAll splits saved to {SAVE_DIR}")
print("Session-separated evaluation — no temporal leakage.")
