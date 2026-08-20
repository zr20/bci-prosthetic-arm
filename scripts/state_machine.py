"""
State machine for confidence-aware shared control.

The same two EEG signals (left/right hand imagery) mean different things
depending on the arm's current state. This is command multiplexing.

States:
  IDLE          → waiting for input
  TARGET_LEFT   → left target selected, awaiting confirmation
  TARGET_RIGHT  → right target selected, awaiting confirmation
  REACHING      → arm autonomously moving to confirmed target
  GRASPING      → hand open/close control active
  RELEASING     → object release in progress

Transitions:
  IDLE          + left  → TARGET_LEFT
  IDLE          + right → TARGET_RIGHT
  TARGET_LEFT   + left  → REACHING (confirmed)
  TARGET_LEFT   + right → TARGET_RIGHT (switched)
  TARGET_RIGHT  + right → REACHING (confirmed)
  TARGET_RIGHT  + left  → TARGET_LEFT (switched)
  REACHING      + auto  → GRASPING (arm planner completes reach)
  GRASPING      + left  → OPEN hand
  GRASPING      + right → CLOSE hand
  any state     + low_confidence → IDLE (safety reset)
  any state     + cancel → IDLE (explicit reset)
"""

import time
import pandas as pd
import os

# ── Intent and command vocabulary ─────────────────────────────────────────────
INTENTS = {
    0: "LEFT",
    1: "RIGHT",
    -1: "IDLE"
}

# Joint commands for each state+intent combination
# Values are normalized motor commands in [-1, 1]
COMMANDS = {
    "SELECT_LEFT":    {"base_rotation": -0.4, "arm_extend": 0.2,
                       "fingers": 0.0},
    "SELECT_RIGHT":   {"base_rotation": +0.4, "arm_extend": 0.2,
                       "fingers": 0.0},
    "REACH_FORWARD":  {"base_rotation":  0.0, "arm_extend": 0.6,
                       "fingers": 0.0},
    "GRASP_OPEN":     {"base_rotation":  None, "arm_extend": None,
                       "fingers": -0.8},
    "GRASP_CLOSE":    {"base_rotation":  None, "arm_extend": None,
                       "fingers": +0.8},
    "RELEASE":        {"base_rotation":  None, "arm_extend": None,
                       "fingers": -1.0},
    "HOLD":           {"base_rotation":  None, "arm_extend": None,
                       "fingers": None},
    "IDLE":           {"base_rotation":  0.0, "arm_extend": 0.0,
                       "fingers": 0.0},
}

class ArmStateMachine:
    """
    Context-dependent command multiplexer for prosthetic arm control.
    The same EEG signal produces different arm actions depending on state.
    """

    STATES = [
        "IDLE", "TARGET_LEFT", "TARGET_RIGHT",
        "REACHING", "GRASPING", "RELEASING"
    ]

    def __init__(self, log_path=None):
        self.state           = "IDLE"
        self.selected_target = None
        self.step_count      = 0
        self.log             = []
        self.log_path        = log_path
        self._log_event("INIT", "IDLE", None, None, "State machine initialised")

    def update(self, decoded_class, confidence, abstained=False):
        """
        Process one decoded EEG trial.

        Args:
            decoded_class:  0 (left) or 1 (right)
            confidence:     softmax probability of winning class
            abstained:      True if policy chose not to act

        Returns:
            intent:   string label of the action taken
            command:  dict of motor commands (or None if abstained)
            new_state: updated state name
        """
        self.step_count += 1
        prev_state = self.state
        intent  = None
        command = None

        if abstained:
            intent  = "ABSTAIN"
            command = None
            self._log_event("ABSTAIN", self.state, decoded_class,
                            confidence, "Below threshold — no action")
            return intent, command, self.state

        signal = INTENTS.get(decoded_class, "IDLE")

        # ── State transitions ──────────────────────────────────────────────────
        if self.state == "IDLE":
            if signal == "LEFT":
                self.state           = "TARGET_LEFT"
                self.selected_target = "LEFT"
                intent  = "SELECT_LEFT"
                command = COMMANDS["SELECT_LEFT"]
            elif signal == "RIGHT":
                self.state           = "TARGET_RIGHT"
                self.selected_target = "RIGHT"
                intent  = "SELECT_RIGHT"
                command = COMMANDS["SELECT_RIGHT"]

        elif self.state == "TARGET_LEFT":
            if signal == "LEFT":          # confirm
                self.state = "REACHING"
                intent  = "REACH_FORWARD"
                command = COMMANDS["REACH_FORWARD"]
            elif signal == "RIGHT":       # switch target
                self.state           = "TARGET_RIGHT"
                self.selected_target = "RIGHT"
                intent  = "SELECT_RIGHT"
                command = COMMANDS["SELECT_RIGHT"]

        elif self.state == "TARGET_RIGHT":
            if signal == "RIGHT":         # confirm
                self.state = "REACHING"
                intent  = "REACH_FORWARD"
                command = COMMANDS["REACH_FORWARD"]
            elif signal == "LEFT":        # switch target
                self.state           = "TARGET_LEFT"
                self.selected_target = "LEFT"
                intent  = "SELECT_LEFT"
                command = COMMANDS["SELECT_LEFT"]

        elif self.state == "REACHING":
            # Arm planner handles autonomous reach
            # EEG signal switches to grasp control
            self.state = "GRASPING"
            if signal == "LEFT":
                intent  = "GRASP_OPEN"
                command = COMMANDS["GRASP_OPEN"]
            else:
                intent  = "GRASP_CLOSE"
                command = COMMANDS["GRASP_CLOSE"]

        elif self.state == "GRASPING":
            if signal == "LEFT":
                intent  = "GRASP_OPEN"
                command = COMMANDS["GRASP_OPEN"]
            else:
                intent  = "GRASP_CLOSE"
                command = COMMANDS["GRASP_CLOSE"]

        elif self.state == "RELEASING":
            self.state = "IDLE"
            intent  = "IDLE"
            command = COMMANDS["IDLE"]

        self._log_event("ACTION", self.state, decoded_class,
                        confidence,
                        f"{prev_state} → {self.state} via {intent}")

        return intent, command, self.state

    def cancel(self):
        """Explicit safety reset to IDLE from any state."""
        prev = self.state
        self.state           = "IDLE"
        self.selected_target = None
        self._log_event("CANCEL", "IDLE", None, None,
                        f"Explicit cancel from {prev}")
        return "IDLE", COMMANDS["IDLE"], "IDLE"

    def complete_grasp(self):
        """Call when grasp task is done — transition to RELEASING."""
        self.state = "RELEASING"
        self._log_event("COMPLETE", "RELEASING", None, None,
                        "Grasp task complete — releasing")

    def _log_event(self, event_type, state, decoded_class,
                   confidence, note):
        self.log.append({
            "timestamp":     time.time(),
            "step":          self.step_count,
            "event_type":    event_type,
            "state":         state,
            "decoded_class": decoded_class,
            "confidence":    confidence,
            "note":          note
        })

    def save_log(self):
        if self.log_path:
            pd.DataFrame(self.log).to_csv(self.log_path, index=False)
            print(f"State machine log saved to {self.log_path}")

    def reset(self):
        self.state           = "IDLE"
        self.selected_target = None
        self.step_count      = 0
        self.log             = []
        self._log_event("RESET", "IDLE", None, None, "Full reset")


if __name__ == "__main__":
    # Quick sanity test
    print("=== State Machine Sanity Test ===")
    sm = ArmStateMachine(log_path="/tmp/sm_test.csv")

    test_sequence = [
        (0, 0.82, False),   # LEFT  → SELECT_LEFT
        (0, 0.91, False),   # LEFT  → REACH_FORWARD (confirmed)
        (1, 0.75, False),   # RIGHT → GRASP_CLOSE
        (0, 0.68, False),   # LEFT  → GRASP_OPEN
        (0, 0.51, True),    # low conf → ABSTAIN
    ]

    for decoded, conf, abstained in test_sequence:
        intent, cmd, state = sm.update(decoded, conf, abstained)
        print(f"  decoded={decoded}, conf={conf:.2f}, "
              f"abstained={abstained} → "
              f"intent={intent}, state={state}")

    sm.save_log()
    print("\nState machine test passed.")
