import numpy as np
from collections import deque

INTENTS = {
    0: "Reach-Left",
    1: "Reach-Right",
    -1: "Idle"
}

INTENT_COMMANDS = {
    "Reach-Left": np.array([
        -0.5, 0.3, 0.0,
        0.3, 0.3, 0.3,
        0.3, 0.3, 0.3,
        0.3, 0.3, 0.3,
        0.3, 0.3, 0.3,
        0.3, 0.3
    ]),
    "Reach-Right": np.array([
        0.5, 0.3, 0.0,
        0.3, 0.3, 0.3,
        0.3, 0.3, 0.3,
        0.3, 0.3, 0.3,
        0.3, 0.3, 0.3,
        0.3, 0.3
    ]),
    "Idle": np.zeros(17)
}

class IntentMapper:
    def __init__(self, smoothing_window=5):
        self.window = deque(maxlen=smoothing_window)
        self.current_intent = "Idle"
        self.ema_command = np.zeros(17)
        self.ema_alpha = 0.3

    def update(self, decoder_output):
        self.window.append(decoder_output)
        counts = np.bincount(list(self.window), minlength=2)
        voted = int(np.argmax(counts))
        self.current_intent = INTENTS[voted]
        return self.current_intent

    def get_command(self):
        target = INTENT_COMMANDS[self.current_intent]
        self.ema_command = (self.ema_alpha * target +
                           (1 - self.ema_alpha) * self.ema_command)
        return self.ema_command.copy()

    def reset(self):
        self.window.clear()
        self.current_intent = "Idle"
        self.ema_command = np.zeros(17)
