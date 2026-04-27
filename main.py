"""Training loop + entry point. Run with `python main.py [options]`."""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pygame

from agent import QLearningAgent
from game import (
    GRID_W, GRID_H, START_MONEY, FPS, WINDOW_W, WINDOW_H,
    Farm, Farmer, apply_action,
)
from view import GifRecorder, Renderer, plot_rewards


# ---- Animation tuning (rendered episodes only; training stays fast) ----
WORK_FRAMES = 6
ARRIVAL_MAX_FRAMES = 25
MAX_ANIMATED_STEPS = 30

# ---- Early stopping ----
ES_WINDOW = 40       # rolling-mean window
ES_MIN_EP = 301      # don't stop before this
ES_PATIENCE = 50     # no-improvement episodes before stopping
ES_MIN_DELTA = 2.0


@dataclass
class TrainConfig:
    episodes: int = 400
    max_steps: int = 120
    seed: int | None = None
    render: bool = True
    save_path: Path | None = None
    load_path: Path | None = None


def cell_cycle(rng: random.Random) -> Iterator[tuple[int, int]]:
    """Infinite shuffled pass over every cell — guarantees the farmer visits
    every plot evenly within any window of GRID_W*GRID_H steps."""
    cells = [(x, y) for y in range(GRID_H) for x in range(GRID_W)]
    while True:
        rng.shuffle(cells)
        yield from cells


def should_render(ep: int, total: int) -> bool:
    return ep <= 2 or ep % 50 == 0 or ep >= total - 1


def converged(
    rewards: list[float], best: float, stable: int
) -> tuple[bool, float, int]:
    if len(rewards) < ES_MIN_EP:
        return False, best, 0
    recent = float(np.mean(rewards[-ES_WINDOW:]))
    if recent > best + ES_MIN_DELTA:
        return False, recent, 0
    return stable + 1 >= ES_PATIENCE, best, stable + 1


# ---------------------------------------------------------------------
# Trainer — owns the world, the agent, and the optional renderer.
# ---------------------------------------------------------------------
class Trainer:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.py_rng = random.Random(cfg.seed)
        np_rng = np.random.default_rng(cfg.seed)
        # Seed module-level RNGs too, so any unseeded callers stay reproducible.
        if cfg.seed is not None:
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)

        self.farm = Farm()
        self.farmer = Farmer()
        self.agent = QLearningAgent(rng=np_rng)
        if cfg.load_path and cfg.load_path.exists():
            self.agent.load(cfg.load_path)
            print(f"[load] Q-table from {cfg.load_path}")

        if cfg.render:
            pygame.init()
            pygame.display.set_caption("Farm Q-Learning")
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
            self.clock = pygame.time.Clock()
            self.renderer: Renderer | None = Renderer(self.screen)
            self.recorder = GifRecorder(every=max(10, cfg.episodes // 8))
        else:
            # Headless: pygame.font still needs init for any font work,
            # but we never open a display.
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.init()
            self.screen = pygame.display.set_mode((1, 1))
            self.clock = pygame.time.Clock()
            self.renderer = None
            self.recorder = GifRecorder(every=10**9)  # effectively disabled

    # ---- one rendered step (walk → pause on action) ----
    def _animate(
        self,
        target: tuple[int, int],
        hud: tuple,
        label: str,
    ) -> bool:
        assert self.renderer is not None
        frames = 0
        while not self.farmer.arrived() and frames < ARRIVAL_MAX_FRAMES:
            dt = self.clock.tick(FPS) / 1000.0
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return False
            self.renderer.draw(self.farm, self.farmer, target, hud, dt)
            pygame.display.flip()
            self.recorder.capture(self.screen, label)
            frames += 1
        self.farmer.snap_to_target()

        for _ in range(WORK_FRAMES):
            dt = self.clock.tick(FPS) / 1000.0
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return False
            self.renderer.draw(self.farm, self.farmer, target, hud, dt)
            pygame.display.flip()
            self.recorder.capture(self.screen, label)
        return True

    # ---- one episode ----
    def run_episode(self, ep: int) -> float | None:
        self.farm.reset()
        self.farmer.reset()
        total_reward = 0.0
        money = START_MONEY
        harvested = 0
        withered = 0

        rendered = self.cfg.render and should_render(ep, self.cfg.episodes)
        self.recorder.start(ep)
        label = f"Ep {ep}   eps={self.agent.eps:.2f}"
        cells = cell_cycle(self.py_rng)
        animated = 0

        for step in range(self.cfg.max_steps):
            if self.cfg.render:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        return None

            tx, ty = next(cells)
            s = self.agent.state(self.farm, tx, ty, money)
            a = self.agent.pick(s)
            r, money, done, info = apply_action(
                self.farm, self.farmer, a, tx, ty, money
            )

            total_reward += r
            harvested += info["harvested"]
            withered += info["withered"]

            wants_anim = (
                rendered and animated < MAX_ANIMATED_STEPS
            ) or self.recorder.active()
            if wants_anim and self.renderer is not None:
                self.renderer.popup(r, tx, ty)
                hud = (ep, self.cfg.episodes, step, total_reward, money,
                       self.farmer.action, self.agent.eps, harvested, withered)
                if not self._animate((tx, ty), hud, label):
                    return None
                animated += 1
            else:
                self.farmer.snap_to_target()

            s_next = self.agent.state(self.farm, tx, ty, money)
            self.agent.update(s, a, r, s_next, done)
            if done:
                break

        self.agent.decay()
        return total_reward

    # ---- full run ----
    def train(self) -> tuple[list[float], int | None, bool]:
        rewards: list[float] = []
        best = -float("inf")
        stable = 0
        stopped: int | None = None
        quit_requested = False

        for ep in range(1, self.cfg.episodes + 1):
            r = self.run_episode(ep)
            if r is None:
                quit_requested = True
                break
            rewards.append(r)

            conv, best, stable = converged(rewards, best, stable)
            if conv:
                stopped = ep
                print(f"[train] converged at ep {ep} — last-{ES_WINDOW} mean "
                      f"{np.mean(rewards[-ES_WINDOW:]):+.1f}")
                break

        return rewards, stopped, quit_requested

    def shutdown(self) -> None:
        pygame.quit()
        if self.cfg.save_path:
            self.agent.save(self.cfg.save_path)
            print(f"[save] Q-table to {self.cfg.save_path}")
        self.recorder.save()


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    p = argparse.ArgumentParser(description="Farm Q-Learning trainer.")
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--seed", type=int, default=None,
                   help="Set for reproducible runs.")
    p.add_argument("--no-render", action="store_true",
                   help="Headless mode — much faster, no GIF.")
    p.add_argument("--save", type=Path, default=None,
                   help="Save Q-table to this path after training.")
    p.add_argument("--load", type=Path, default=None,
                   help="Load Q-table from this path before training.")
    a = p.parse_args(argv)
    return TrainConfig(
        episodes=a.episodes,
        max_steps=a.max_steps,
        seed=a.seed,
        render=not a.no_render,
        save_path=a.save,
        load_path=a.load,
    )


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(argv)
    trainer = Trainer(cfg)
    rewards, stopped, quit_requested = trainer.train()
    trainer.shutdown()

    if rewards:
        plot_rewards(rewards, stopped_early=stopped, show=cfg.render)
    if quit_requested:
        sys.exit(0)


if __name__ == "__main__":
    main()
