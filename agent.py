"""Q-Learning agent. Tabular, ε-greedy, Bellman update."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from game import Farm, NUM_ACTIONS

ALPHA = 0.25
GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.97

State = tuple[int, int, int]


class QLearningAgent:
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.Q: dict[State, np.ndarray] = defaultdict(
            lambda: np.zeros(NUM_ACTIONS)
        )
        self.eps = EPS_START
        self.rng = rng if rng is not None else np.random.default_rng()

    def state(self, farm: Farm, x: int, y: int, money: int) -> State:
        """Compact state: (cell_state, watered, money_bucket). ~48 states total."""
        c = farm.cell(x, y)
        if money < 20:
            bucket = 0
        elif money < 50:
            bucket = 1
        elif money < 100:
            bucket = 2
        else:
            bucket = 3
        return (c.state, int(c.watered), bucket)

    def pick(self, s: State) -> int:
        """ε-greedy action selection. Ties broken uniformly at random."""
        if self.rng.random() < self.eps:
            return int(self.rng.integers(NUM_ACTIONS))
        q = self.Q[s]
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def update(self, s: State, a: int, r: float, s_next: State, done: bool) -> None:
        target = r if done else r + GAMMA * self.Q[s_next].max()
        self.Q[s][a] += ALPHA * (target - self.Q[s][a])

    def decay(self) -> None:
        self.eps = max(EPS_END, self.eps * EPS_DECAY)

    # ---- persistence ----
    def save(self, path: str | Path) -> None:
        """Persist the Q-table to disk as a plain dict."""
        np.save(path, dict(self.Q), allow_pickle=True)

    def load(self, path: str | Path) -> None:
        loaded = np.load(path, allow_pickle=True).item()
        self.Q = defaultdict(lambda: np.zeros(NUM_ACTIONS), loaded)
