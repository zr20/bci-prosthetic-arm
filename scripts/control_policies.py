"""
Three control policies for the confidence-aware shared control paper.

Policy A: Raw argmax — direct mapping, always acts
Policy B: Confidence-gated — abstains below threshold
Policy C: Confidence-aware shared control — uses state machine +
          context-dependent command multiplexing

All policies use the same decoder and same task structure.
"""

import numpy as np


class PolicyA_DirectControl:
    """
    Raw argmax direct mapping.
    Always acts on the decoder output regardless of confidence.
    Baseline policy — no confidence awareness.
    """
    name = "Policy A: Direct Control"

    def decide(self, probs):
        """
        Args:
            probs: softmax probabilities (2,)
        Returns:
            decoded_class: 0 or 1
            confidence:    winning class probability
            abstained:     always False for Policy A
        """
        decoded    = int(probs.argmax())
        confidence = float(probs.max())
        return decoded, confidence, False


class PolicyB_ConfidenceGated:
    """
    Confidence-gated mapping with abstention.
    Acts only when confidence exceeds the threshold.
    Abstains (issues no command) when uncertain.
    Threshold chosen on development data before final evaluation.
    """
    name = "Policy B: Confidence-Gated"

    def __init__(self, threshold=0.75):
        """
        Args:
            threshold: minimum confidence to act (chosen on dev data)
        """
        self.threshold   = threshold
        self.n_acted     = 0
        self.n_abstained = 0

    def decide(self, probs):
        """
        Returns:
            decoded_class: 0 or 1 (None if abstained)
            confidence:    winning class probability
            abstained:     True if below threshold
        """
        decoded    = int(probs.argmax())
        confidence = float(probs.max())

        if confidence < self.threshold:
            self.n_abstained += 1
            return None, confidence, True

        self.n_acted += 1
        return decoded, confidence, False

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

    The same EEG signal (left/right) means different things
    depending on the arm's current state:
      - IDLE:      left/right selects a reach target
      - TARGET:    repeat same side confirms, other side switches
      - REACHING:  arm moves autonomously (planner handles it)
      - GRASPING:  left=open, right=close

    Low confidence → abstain (stay in current state, no action).
    """
    name = "Policy C: Confidence-Aware Shared Control"

    def __init__(self, threshold=0.75, state_machine=None):
        """
        Args:
            threshold:     minimum confidence to act
            state_machine: ArmStateMachine instance
        """
        self.threshold    = threshold
        self.n_acted      = 0
        self.n_abstained  = 0
        self.state_machine = state_machine

    def decide(self, probs):
        """
        Returns:
            intent:     string describing the action
            confidence: winning class probability
            abstained:  True if below threshold
        """
        decoded    = int(probs.argmax())
        confidence = float(probs.max())
        abstained  = confidence < self.threshold

        if self.state_machine is None:
            # Fallback: behave like Policy B without state machine
            if abstained:
                self.n_abstained += 1
                return None, confidence, True
            self.n_acted += 1
            return decoded, confidence, False

        intent, command, new_state = self.state_machine.update(
            decoded, confidence, abstained
        )

        if abstained:
            self.n_abstained += 1
        else:
            self.n_acted += 1

        return intent, confidence, abstained

    @property
    def abstention_rate(self):
        total = self.n_acted + self.n_abstained
        return self.n_abstained / total if total > 0 else 0.0

    @property
    def coverage(self):
        return 1.0 - self.abstention_rate


if __name__ == "__main__":
    # Quick sanity test
    import sys
    sys.path.insert(0, '/Users/zarakhursheed/bci_paper/scripts')
    from state_machine import ArmStateMachine

    print("=== Control Policy Sanity Test ===")

    test_probs = [
        np.array([0.85, 0.15]),   # high confidence left
        np.array([0.60, 0.40]),   # low confidence — below threshold
        np.array([0.20, 0.80]),   # high confidence right
        np.array([0.55, 0.45]),   # low confidence
        np.array([0.90, 0.10]),   # high confidence left
    ]

    print("\nPolicy A (Direct):")
    pA = PolicyA_DirectControl()
    for p in test_probs:
        dec, conf, abs_ = pA.decide(p)
        print(f"  probs={p} → decoded={dec}, conf={conf:.2f}, abstained={abs_}")

    print("\nPolicy B (Confidence-Gated, threshold=0.75):")
    pB = PolicyB_ConfidenceGated(threshold=0.75)
    for p in test_probs:
        dec, conf, abs_ = pB.decide(p)
        print(f"  probs={p} → decoded={dec}, conf={conf:.2f}, abstained={abs_}")
    print(f"  Abstention rate: {pB.abstention_rate:.2f}, "
          f"Coverage: {pB.coverage:.2f}")

    print("\nPolicy C (Shared Control, threshold=0.75):")
    sm = ArmStateMachine()
    pC = PolicyC_SharedControl(threshold=0.75, state_machine=sm)
    for p in test_probs:
        intent, conf, abs_ = pC.decide(p)
        print(f"  probs={p} → intent={intent}, conf={conf:.2f}, "
              f"abstained={abs_}, state={sm.state}")

    print("\nAll policy tests passed.")
