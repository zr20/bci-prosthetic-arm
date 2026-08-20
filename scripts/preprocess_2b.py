epochs = mne.Epochs(raw, events, event_id=picks_id,
                     tmin=0.5, tmax=2.5, baseline=None,
                     preload=True, verbose=False)
import mne
import numpy as np
import os
from sklearn.model_selection import train_test_split

DATA_DIR = "/Users/zarakhursheed/Downloads/NYU/DATASETS/Dataset 2b BCICIV/Training"
SAVE_DIR = "processed_data_2b"
os.makedirs(SAVE_DIR, exist_ok=True)

SUBJECTS = list(range(1, 10))
SESSIONS = [1, 2, 3]

def load_session(subject, session):
    fname = f"B{subject:02d}{session:02d}T.gdf"
    path = os.path.join(DATA_DIR, fname)
    raw = mne.io.read_raw_gdf(path, preload=True, verbose=False)

    # Keep only the 3 EEG motor channels
    raw.pick(['EEG:C3', 'EEG:Cz', 'EEG:C4'])

    # Bandpass + notch
    raw.filter(4, 40, fir_design='firwin', verbose=False)
    raw.notch_filter(50, verbose=False)

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    # 769 = left hand, 770 = right hand (codes confirmed from your file)
    picks_id = {k: v for k, v in event_id.items() if k in ['769', '770']}

    epochs = mne.Epochs(raw, events, event_id=picks_id,
                         tmin=-0.5, tmax=4.0, baseline=None,
                         preload=True, verbose=False)

    X = epochs.get_data()  # (n_trials, 3, n_timepoints)
    # Map: 769 (left) -> 0, 770 (right) -> 1
    code_to_label = {v: (0 if k == '769' else 1) for k, v in picks_id.items()}
    y = np.array([code_to_label[e[2]] for e in epochs.events])
    return X, y

all_data = {}
for subj in SUBJECTS:
    X_list, y_list = [], []
    for sess in SESSIONS:
        try:
            X, y = load_session(subj, sess)
            X_list.append(X)
            y_list.append(y)
            print(f"Subject {subj}, session {sess}: {X.shape[0]} trials")
        except FileNotFoundError:
            print(f"Missing file for subject {subj}, session {sess} — skipping")
    X_subj = np.concatenate(X_list, axis=0)
    y_subj = np.concatenate(y_list, axis=0)
    all_data[subj] = {"X": X_subj, "y": y_subj}
    print(f"  -> Subject {subj} total: {X_subj.shape[0]} trials, shape {X_subj.shape}")

# Within-subject splits
for subj, d in all_data.items():
    X_train, X_test, y_train, y_test = train_test_split(
        d["X"], d["y"], test_size=0.2, stratify=d["y"], random_state=42
    )
    np.savez(f"{SAVE_DIR}/within_subject_{subj}.npz",
             X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
print("Within-subject splits saved.")

# Cross-subject (leave-one-subject-out)
for test_subj in SUBJECTS:
    train_subjs = [s for s in SUBJECTS if s != test_subj]
    X_train = np.concatenate([all_data[s]["X"] for s in train_subjs], axis=0)
    y_train = np.concatenate([all_data[s]["y"] for s in train_subjs], axis=0)
    X_test = all_data[test_subj]["X"]
    y_test = all_data[test_subj]["y"]
    np.savez(f"{SAVE_DIR}/cross_subject_holdout_{test_subj}.npz",
             X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print(f"Cross-subject (holdout {test_subj}): train={X_train.shape[0]}, test={X_test.shape[0]}")

print("All splits saved to", SAVE_DIR)
