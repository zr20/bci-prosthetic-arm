# Confidence-Aware Shared Control for Two-Class Motor-Imagery Interaction

**Paper:** Confidence-Aware Shared Control for Two-Class Motor-Imagery
Interaction with an Assistive Robotic Arm
**Target venue:** CHI 2027
**Author:** Zara Khursheed — NYU Abu Dhabi eBrain Lab
**Supervisor:** Professor Abdul Basit

---

## Dataset

- **Source dataset:** BCI Competition IV Dataset 2b (BNCI2014_004)
- **Target dataset:** BCI Competition IV Dataset 2a (BNCI2014_001)
- **Download:** Both datasets downloaded automatically via MOABB on first run
- **Subjects:** 9 per dataset
- **Sessions used:** T-files (sessions 1, 2, 3 per subject)
- **Channels:** C3, Cz, C4
- **Sampling rate:** 250 Hz

---

## Environment Setup

Requires: Python 3.10, conda

```bash
conda env create -f environment.yml
conda activate bci-prosthetic
```

Verify:

```bash
python -c "import torch, mne, moabb, braindecode; print('OK')"
```

---

## Repository Structure

bci_paper/
├── configs/
│ └── config.py
├── scripts/
│ ├── preprocess_2b_session_split.py
│ ├── check_leakage.py
│ ├── deep_baselines_seeded.py
│ ├── compute_ci.py
│ ├── confidence_calibration.py
│ ├── compute_calibration_ci.py
│ ├── temperature_scaling.py
│ ├── cross_dataset_transfer.py
│ ├── compute_cross_dataset_ci.py
│ ├── document_harmonisation.py
│ ├── confusion_matrices.py
│ ├── parameter_counts.py
│ ├── threshold_selection_shallow.py
│ ├── state_machine.py
│ ├── control_policies.py
│ ├── policy_comparison.py
│ └── paired_statistics.py
├── results/
├── logs/
├── checkpoints/
├── environment.yml
└── requirements.txt


Data splits saved to: `~/processed_data_2b/`

---

## Reproduction Guide — One Command Per Table

Run all from `~/bci_paper/scripts/` with `conda activate bci-prosthetic`.

### Step 1: Preprocessing

```bash
python preprocess_2b_session_split.py
python check_leakage.py
```

**Outputs:** `~/processed_data_2b/*.npz` (18 files), leakage report

---

### Step 2: Baselines — Table 2

```bash
python deep_baselines_seeded.py 42
python deep_baselines_seeded.py 0
python deep_baselines_seeded.py 1
python deep_baselines_seeded.py 2
python deep_baselines_seeded.py 3
python compute_ci.py
```

**Outputs:** `results/seed_*/deep_baselines_seed*.csv`,
`results/results_with_ci.csv`

---

### Step 3: Confusion Matrices and Parameter Counts

```bash
python confusion_matrices.py
python parameter_counts.py
```

**Outputs:** `results/confusion_matrices.csv`,
`results/confusion_summary.csv`,
`results/parameter_counts.csv`

---

### Step 4: Confidence Calibration — Table 3

```bash
python confidence_calibration.py 42
python confidence_calibration.py 0
python confidence_calibration.py 1
python confidence_calibration.py 2
python confidence_calibration.py 3
python compute_calibration_ci.py
```

**Outputs:** `results/seed_*/calibration_seed*.csv`,
`results/calibration_summary.csv`

---

### Step 5: Temperature Scaling

Calibrates ShallowConvNet probabilities on held-out calibration
data (20% of training set per subject, never test data).

```bash
python temperature_scaling.py
```

**Outputs:** `results/temperature_scaling.csv`

**Results:** Mean ECE before: 0.136, Mean ECE after: 0.102.
Subjects 8 and 9 showed slight ECE increase — calibration set
too small to generalise for already well-calibrated subjects.

---

### Step 6: Cross-Dataset Transfer — Table 4

NOTE: BCI IV-2a NOT scaled by 1e6 — MOABB already returns microvolts.

```bash
python cross_dataset_transfer.py 42
python cross_dataset_transfer.py 0
python cross_dataset_transfer.py 1
python cross_dataset_transfer.py 2
python cross_dataset_transfer.py 3
python compute_cross_dataset_ci.py
```

**Outputs:** `results/seed_*/cross_dataset_seed*.csv`,
`results/cross_dataset_ci.csv`,
`results/harmonisation_documentation.csv`

**Results:** EEGNet domain gap 16.8pp, ShallowConvNet 14.8pp.

---

### Step 7: Threshold Selection (frozen at 0.83)

Must be run BEFORE Step 8.
Uses only dev subjects [7, 8, 9].
Evaluation subjects [1-6] are never touched during this step.

```bash
python threshold_selection_shallow.py
```

**Outputs:** `results/selected_threshold.txt` (threshold = 0.83),
`results/threshold_sweep_shallow.csv`

---

### Step 8: Three-Policy Comparison — Table 5

Uses ShallowConvNet decoder, frozen threshold 0.83, all 9 subjects.
Policy B and C use confirmation logic — 2 consecutive confident
same-class predictions required before acting.
Labelled as OFFLINE REPLAY SIMULATION.

```bash
python policy_comparison.py
```

**Outputs:**
- `results/policy_comparison_summary_v2.csv`
- `results/policy_comparison_trials_v2.csv`
- `results/latency_breakdown.csv`
- `results/sm_log_subj*.csv`

**Results (mean across 9 subjects):**

| Metric | Policy A | Policy B | Policy C |
|---|---|---|---|
| Success rate | 73.0% | 74.8% | 74.8% |
| Coverage | 100% | 15.4% | 15.4% |
| Unsafe actions | 389 | 42 | 42 |
| Retries | 0 | 222 | 222 |
| Mean latency | 2.792ms | 2.235ms | 4.711ms |

---

### Step 9: Paired Statistics

```bash
python paired_statistics.py
```

**Outputs:** `results/paired_statistics.csv`,
`results/paired_statistics_per_subject.csv`

**Results:** Policy A vs B success rate: mean diff +6.78pp,
p=0.0027 (Wilcoxon p=0.0039), Cohen's d=1.427 (large effect).

---

## Key Hyperparameters

All defined in `configs/config.py`:

| Parameter | Value |
|---|---|
| Seeds | [42, 0, 1, 2, 3] |
| Epochs | 30 |
| Learning rate | 1e-3 (Adam) |
| Batch size | 128 |
| EEG scaling | x1e6 (BCI IV-2b only) |
| Confidence threshold | 0.83 (frozen on dev subjects 7-9) |
| Confirmation | 2 consecutive confident same-class predictions |
| Dev subjects | [7, 8, 9] |
| Eval subjects | [1, 2, 3, 4, 5, 6] |
| Min coverage | 60% |

---

## Reproducibility Notes

- All random seeds fixed via `random.seed()`, `numpy.random.seed()`,
  `torch.manual_seed()`, `torch.mps.manual_seed()`
- Session-separated evaluation: train sessions 1+2, test session 3
- Leakage verification: MD5 hash comparison — all 9 subjects CLEAN
- Threshold selection used only dev subjects [7,8,9]
- BCI IV-2b data scaled x1e6 (raw Volts to microvolts)
- BCI IV-2a NOT scaled — MOABB already returns microvolts
- Temperature scaling fitted on 20% held-out calibration split
  per subject — never fitted on test data
- Hardware: MacBook Air M2 (MPS for CNN training)

---

## Latency Breakdown

| Component | Duration | Notes |
|---|---|---|
| EEG window accumulation | 3,500ms | Fixed by trial duration |
| Preprocessing | ~0.1ms | Fixed constant scaling |
| Model inference | ~2.1ms | ShallowConvNet on CPU |
| Policy decision | ~0.7ms | Threshold + confirmation |
| Communication | 0ms | Not applicable in simulation |
| Actuation | 0ms | Not measured in simulation |
| **Total decode-to-command** | **~2.8ms** | Measured |
| **Full sensing-to-action** | **~3,503ms** | Estimated |

---

## Future Work (Not Implemented — Requires Hardware)

The following items from the publication plan require either live
EEG recording sessions or physical hardware and are documented
here as planned future work:

**1. Rest/no-control data collection**
BCI IV-2b contains no rest epochs. Idle state validation requires
recording a live session where the participant sits quietly. Future
work will train a three-class decoder (left, right, rest).

**2. Live control state display**
A real deployment requires a GUI showing: current state (IDLE,
TARGET, REACHING, GRASPING), decoded command, confidence/acceptance
status, and a cancel/reset button triggering the state machine's
`cancel()` method. Not implemented in replay simulation.

**3. Synchronized event logging**
Full deployment requires logging every EEG window, command,
rejection, state transition, retry, and robot event with
synchronized timestamps to one shared log file. The replay
simulation logs decisions to `policy_comparison_trials_v2.csv`
and state transitions to `sm_log_subj*.csv` but these are not
timestamp-synchronized.

**4. Hardware safety limits**
Before connecting to a physical arm: workspace limits, speed and
force caps, and real-time collision checking must be implemented
at the hardware controller level. MuJoCo handles these
automatically in simulation.

**5. Balanced reach-grasp-release task scripts**
Full task sequences (reach → grasp → release) with a physical
object in the simulation. Current evaluation covers the reach
phase only. The state machine has all required states implemented.

---

## Parameter Counts

| Model | Parameters | Size |
|---|---|---|
| EEGNet | 2,018 | 7.9 KB |
| ShallowConvNet | 10,082 | 39.4 KB |
| Tiny Transformer | ~15,000 | 58.6 KB |
| CSP + LDA | N/A | < 1 KB |

---

## GitHub

Repository: https://github.com/zr20/bci-prosthetic-arm
