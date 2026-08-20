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
- **Download:** Both datasets are downloaded automatically via MOABB
  on first run. No manual download required.
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

To verify the environment:

```bash
python -c "import torch, mne, moabb, braindecode; print('OK')"
```

---

## Repository Structure
