"""
Three control policies for the confidence-aware shared control paper.

Policy A: Raw argmax — direct mapping, always acts
Policy B: Confidence-gated with abstention AND confirmation
          (requires two consecutive confident same-class predictions)
Policy C: Confidence-aware shared control — uses state machine +
          context-dependent command multiplexing
"""

import numpy as np


class PolicyA_DirectControl:
    """
    Raw argmax direct mapping.
    Always acts on the decoder output regardless of confidence.
    Baseline policy — no confidence awareness.
    """
    name = "Policy A: Direct Control"

    def __init__(self):
        self.n_acted     = 0
        self.n_abstained = 0

    def decide(self, probs):
        decoded    = int(probs.argmax())
        confidence = float(probs.max())
        self.n_acted += 1
        return decoded, confidence, False


class PolicyB_ConfidenceGated:
    """
    Confidence-gated mapping with abstention AND confirmation.

    Two-stage logic:
      Stage 1 — Propose: first confident prediction above threshold
                is recorded but does NOT trigger a command yet.
      Stage 2 — Confirm: if the NEXT confident prediction matches
                the proposed class, act. If it differs or is below
                threshold, reset and start over.

    This means two consecutive confident same-class predictions
    are required before any command is issued. This halves the
    risk of acting on a single misclassification.

    Low confidence at any point resets the proposal.
    """
    name = "Policy B: Confidence-Gated with Confirmation"

    def __init__(self, threshold=0.83):
        self.threshold      = threshold
        self.last_confident = None   # proposed class waiting confirmation
        self.n_acted        = 0
        self.n_abstained    = 0

    def decide(self, probs):
        """
        Returns:
            decoded_class: int or None (None if abstained)
            confidence:    float — winning class probability
            abstained:     bool — True if no command issued
        """
        decoded    = int(probs.argmax())
        confidence = float(probs.max())

        # Below threshold — reset proposal, abstain
        if confidence < self.threshold:
            self.last_confident = None
            self.n_abstained += 1
            return None, confidence, True

        # Above threshold — check against proposal
        if self.last_confident is not None \
                and decoded == self.last_confident:
            # Confirmed — act and reset proposal
            self.last_confident = None
            self.n_acted += 1
            return decoded, confidence, False

        # Above threshold but no matching proposal yet — propose
        self.last_confident = decoded
        self.n_abstained += 1
        return None, confidence, True   # proposing, not yet acting

    @property
    def abstention_rate(self):
        total = self.n_acted + self.n_abstained
        return self.n_abstained / total if total > 0 else 0.0

    @property
    def coverage(self):
        return 1.0 - self.abstention_rate


class PolicyC_SharedControl:
    """
    Confidence-aware shared control with context-dependent
    command multiplexing via the state machine.

    Same two-stage confidence logic as Policy B.
    Additionally routes confirmed commands through the state
    machine so the same EEG signal means different arm actions
    depending on the current state.
    """
    name = "Policy C: Confidence-Aware Shared Control"

    def __init__(self, threshold=0.83, state_machine=None):
        self.threshold     = threshold
        self.state_machine = state_machine
        self.last_confident = None
        self.n_acted        = 0
        self.n_abstained    = 0

    def decide(self, probs):
        decoded    = int(probs.argmax())
        confidence = float(probs.max())

        # Below threshold — reset, abstain
        if confidence < self.threshold:
            self.last_confident = None
            if self.state_machine:
                _, _, _ = self.state_machine.update(
                    decoded, confidence, abstained=True
                )
            self.n_abstained += 1
            return None, confidence, True

        # Above threshold — check confirmation
        if self.last_confident is not None \
                and decoded == self.last_confident:
            # Confirmed — route through state machine
            self.last_confident = None
            if self.state_machine:
                intent, command, new_state = self.state_machine.update(
                    decoded, confidence, abstained=False
                )
            else:
                intent = f"CLASS_{decoded}"
            self.n_acted += 1
            return intent, confidence, False

        # Proposing — do not act yet
        self.last_confident = decoded
        if self.state_machine:
            _, _, _ = self.state_machine.update(
                decoded, confidence, abstained=True
            )
        self.n_abstained += 1
        return None, confidence, True

    @property
    def abstention_rate(self):
        total = self.n_acted + self.n_abstained
        return self.n_abstained / total if total > 0 else 0.0

    @property
    def coverage(self):
        return 1.0 - self.abstention_rate


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/zarakhursheed/bci_paper/scripts')
    from state_machine import ArmStateMachine

    print("=== Control Policy Sanity Test (with confirmation) ===\n")

    # Sequence designed to test confirmation logic:
    # Trials 1+2: both left, high conf → should confirm on trial 2
    # Trial 3: right high conf → proposes right (no action yet)
    # Trial 4: left high conf → differs from proposal → resets, proposes left
    # Trial 5: low conf → resets proposal
    # Trial 6+7: right twice → confirms on trial 7
    test_probs = [
        np.array([0.88, 0.12]),   # 1. Left, high → propose left
        np.array([0.91, 0.09]),   # 2. Left, high → CONFIRM → act
        np.array([0.15, 0.85]),   # 3. Right, high → propose right
        np.array([0.87, 0.13]),   # 4. Left, high → mismatch → propose left
        np.array([0.55, 0.45]),   # 5. Low conf → reset
        np.array([0.20, 0.80]),   # 6. Right, high → propose right
        np.array([0.18, 0.82]),   # 7. Right, high → CONFIRM → act
    ]

    expected = [
        "abstain (propose left)",
        "ACT left (confirmed)",
        "abstain (propose right)",
        "abstain (mismatch, propose left)",
        "abstain (low conf, reset)",
        "abstain (propose right)",
        "ACT right (confirmed)",
    ]

    print("Policy B (Confidence-Gated with Confirmation, threshold=0.83):")
    pB = PolicyB_ConfidenceGated(threshold=0.83)
    all_passed = True
    for i, (p, exp) in enumerate(zip(test_probs, expected)):
        dec, conf, abs_ = pB.decide(p)
        acted = not abs_
        print(f"  Trial {i+1}: conf={conf:.2f}, "
              f"decoded={dec}, abstained={abs_}  |  "
              f"Expected: {exp}")

    print(f"\nAbstention rate: {pB.abstention_rate:.2f}")
    print(f"Coverage:        {pB.coverage:.2f}")

    print("\nPolicy C (Shared Control with Confirmation, threshold=0.83):")
    sm = ArmStateMachine()
    pC = PolicyC_SharedControl(threshold=0.83, state_machine=sm)
    for i, p in enumerate(test_probs):
        intent, conf, abs_ = pC.decide(p)
        print(f"  Trial {i+1}: conf={conf:.2f}, "
              f"intent={intent}, abstained={abs_}, "
              f"state={sm.state}")

    print("\nSanity test complete.")
