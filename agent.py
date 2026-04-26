"""Q-Learning agent. Tabular, ε-greedy, Bellman update."""

import random
from collections import defaultdict

import numpy as np

from game import NUM_ACTIONS

ALPHA = 0.25
GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.97


class QLearningAgent:
    def __init__(self):
        self.Q = defaultdict(lambda: np.zeros(NUM_ACTIONS))
        self.eps = EPS_START

    def state(self, farm, x, y, money):
        """Tiny state: (cell_state, watered, money_bucket). ~48 states total."""
        c = farm.cell(x, y)
        bucket = 0 if money < 20 else 1 if money < 50 else 2 if money < 100 else 3
        return (c.state, int(c.watered), bucket)

    def pick(self, s):
        if random.random() < self.eps:
            return random.randint(0, NUM_ACTIONS - 1)
        q = self.Q[s]
        return int(np.random.choice(np.flatnonzero(q == np.max(q))))

    def update(self, s, a, r, s_next, done):
        target = r if done else r + GAMMA * np.max(self.Q[s_next])
        self.Q[s][a] += ALPHA * (target - self.Q[s][a])

    def decay(self):
        self.eps = max(EPS_END, self.eps * EPS_DECAY)
