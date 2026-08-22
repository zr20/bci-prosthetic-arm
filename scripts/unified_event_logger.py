"""
Unified synchronized event logger.

Merges per-trial policy decisions and state machine transitions
into one log file with synchronized reference timestamps.

Every event type is captured:
  - WINDOW    : EEG epoch received and processed
  - INFERENCE : decoder output (class + confidence)
  - ACCEPTED  : trial acted on (above threshold + confirmed)
  - REJECTED  : trial abstained (below threshold)
  - PROPOSAL  : first confident prediction — waiting for confirmation
  - RETRY     : user repeated same signal after rejection
  - STATE     : state machine transition
  - RESET     : state machine returned to IDLE (safety)

All events share a single reference clock (trial-relative ms).
"""

import os
import sys
import numpy as np
import pandas as pd

RESULTS_DIR = "/Users/zarakhursheed/bci_paper/results"

# ── Load per-trial policy data ────────────────────────────────────────────────
trials_path = os.path.join(RESULTS_DIR,
                            "policy_comparison_trials_v2.csv")
trials_df   = pd.read_csv(trials_path)

print(f"Loaded {len(trials_df)} trial records from policy comparison.")
print(f"Policies: {trials_df['policy'].unique().tolist()}")
print(f"Subjects: {sorted(trials_df['subject'].unique().tolist())}")
print()

TRIAL_DURATION_MS = 4500.0   # 4.5 seconds per trial

unified_rows = []

for policy in ["Policy_A_Direct",
               "Policy_B_ConfidenceGated",
               "Policy_C_SharedControl"]:

    policy_df   = trials_df[trials_df["policy"] == policy].copy()
    policy_df   = policy_df.sort_values(
        ["subject", "trial"]
    ).reset_index(drop=True)

    # Load corresponding state machine log if it exists
    sm_logs = {}
    for subj in policy_df["subject"].unique():
        sm_path = os.path.join(
            RESULTS_DIR, f"sm_log_subj{subj}_v2.csv"
        )
        if not os.path.exists(sm_path):
            sm_path = os.path.join(
                RESULTS_DIR, f"sm_log_subj{subj}.csv"
            )
        if os.path.exists(sm_path):
            sm_logs[subj] = pd.read_csv(sm_path)

    # Track previous trial for retry detection
    prev_by_subj = {}

    for _, row in policy_df.iterrows():
        subj    = row["subject"]
        trial   = row["trial"]
        conf    = row["confidence"]
        decoded = row["decoded"]
        abstain = row["abstained"]
        correct = row["correct"]
        unsafe  = row["unsafe_action"]
        lat     = row["latency_ms"]
        intent  = row["intent"]

        # Reference timestamp: trial start relative to subject start
        t_ref_ms = (trial - 1) * TRIAL_DURATION_MS

        # ── EVENT 1: WINDOW ──────────────────────────────────────────────────
        unified_rows.append({
            "policy":         policy,
            "subject":        subj,
            "trial":          trial,
            "t_ref_ms":       t_ref_ms,
            "event_type":     "WINDOW",
            "decoded_class":  decoded,
            "confidence":     conf,
            "intent":         None,
            "correct":        None,
            "unsafe":         None,
            "note":           f"EEG epoch received — "
                              f"trial {trial}, subject {subj}",
        })

        # ── EVENT 2: INFERENCE ────────────────────────────────────────────────
        unified_rows.append({
            "policy":         policy,
            "subject":        subj,
            "trial":          trial,
            "t_ref_ms":       t_ref_ms + 0.1,
            "event_type":     "INFERENCE",
            "decoded_class":  decoded,
            "confidence":     conf,
            "intent":         None,
            "correct":        None,
            "unsafe":         None,
            "note":           f"Decoder output: class={decoded}, "
                              f"conf={conf:.3f}, latency={lat:.2f}ms",
        })

        # ── EVENT 3a: REJECTED (abstained — below threshold) ─────────────────
        # ── EVENT 3b: PROPOSAL (first confident, awaiting confirmation) ───────
        # ── EVENT 3c: ACCEPTED (confirmed — command issued) ───────────────────
        if abstain:
            if conf < 0.83:
                event_type = "REJECTED"
                note = (f"Below threshold (conf={conf:.3f} < 0.83) "
                        f"— no command issued")
            else:
                event_type = "PROPOSAL"
                note = (f"First confident prediction "
                        f"(conf={conf:.3f} ≥ 0.83, class={decoded}) "
                        f"— awaiting confirmation")
        else:
            event_type = "ACCEPTED"
            note = (f"Confirmed command issued: {intent} "
                    f"(conf={conf:.3f}, correct={correct}, "
                    f"unsafe={unsafe})")

        unified_rows.append({
            "policy":         policy,
            "subject":        subj,
            "trial":          trial,
            "t_ref_ms":       t_ref_ms + lat,
            "event_type":     event_type,
            "decoded_class":  decoded,
            "confidence":     conf,
            "intent":         intent if not abstain else None,
            "correct":        correct if not abstain else None,
            "unsafe":         unsafe if not abstain else None,
            "note":           note,
        })

        # ── EVENT 4: RETRY detection ──────────────────────────────────────────
        prev = prev_by_subj.get(subj)
        if prev is not None:
            if (prev["abstained"] and
                    not abstain and
                    prev["decoded"] == decoded):
                unified_rows.append({
                    "policy":         policy,
                    "subject":        subj,
                    "trial":          trial,
                    "t_ref_ms":       t_ref_ms + lat + 0.1,
                    "event_type":     "RETRY",
                    "decoded_class":  decoded,
                    "confidence":     conf,
                    "intent":         intent,
                    "correct":        correct,
                    "unsafe":         unsafe,
                    "note":           f"User repeated class={decoded} "
                                      f"after trial {prev['trial']} "
                                      f"was abstained",
                })

        prev_by_subj[subj] = {
            "trial":    trial,
            "decoded":  decoded,
            "abstained": abstain,
        }

    # ── EVENT 5: STATE TRANSITIONS from state machine log ────────────────────
    if policy == "Policy_C_SharedControl":
        for subj, sm_df in sm_logs.items():
            for _, sm_row in sm_df.iterrows():
                step  = sm_row.get("step", 0)
                t_ref = (step - 1) * TRIAL_DURATION_MS if step > 0 else 0

                event = sm_row.get("event_type", "STATE")
                if event in ["CANCEL", "RESET"]:
                    evt_type = "RESET"
                else:
                    evt_type = "STATE"

                unified_rows.append({
                    "policy":         policy,
                    "subject":        subj,
                    "trial":          step,
                    "t_ref_ms":       t_ref + 0.5,
                    "event_type":     evt_type,
                    "decoded_class":  sm_row.get("decoded_class"),
                    "confidence":     sm_row.get("confidence"),
                    "intent":         None,
                    "correct":        None,
                    "unsafe":         None,
                    "note":           sm_row.get("note", ""),
                })

# ── Save unified log ──────────────────────────────────────────────────────────
unified_df = pd.DataFrame(unified_rows)
unified_df = unified_df.sort_values(
    ["policy", "subject", "t_ref_ms"]
).reset_index(drop=True)

out = os.path.join(RESULTS_DIR, "unified_event_log.csv")
unified_df.to_csv(out, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print("=== Unified Event Log Summary ===\n")
print(f"Total events logged: {len(unified_df):,}")
print()

counts = unified_df.groupby(
    ["policy", "event_type"]
)["trial"].count().unstack(fill_value=0)
print(counts.to_string())

print(f"\nSaved to {out}")
print()
print("Event types:")
print("  WINDOW   — EEG epoch received")
print("  INFERENCE— decoder output (class + confidence)")
print("  PROPOSAL — first confident prediction, awaiting confirmation")
print("  ACCEPTED — confirmed command issued")
print("  REJECTED — below threshold, no command")
print("  RETRY    — user repeated signal after rejection")
print("  STATE    — state machine transition (Policy C only)")
print("  RESET    — state machine returned to IDLE")
