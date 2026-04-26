"""Training loop + entry point. Run with `python main.py`."""

import random
import sys

import numpy as np
import pygame

from game import (
    WINDOW_W, WINDOW_H, FPS,
    GRID_W, GRID_H, START_MONEY,
    Farm, Farmer, apply_action,
)
from agent import QLearningAgent
from view import Renderer, GifRecorder, plot_rewards


# ---- Training config ----
NUM_EPISODES = 400
MAX_STEPS = 120

# ---- Animation tuning (rendered episodes only; training stays fast) ----
WORK_FRAMES = 6
ARRIVAL_MAX_FRAMES = 25
MAX_ANIMATED_STEPS = 30

# ---- Early stopping ----
ES_WINDOW = 40       # rolling-mean window
ES_MIN_EP = 100      # don't stop before this
ES_PATIENCE = 50     # no-improvement episodes before stopping
ES_MIN_DELTA = 2.0


def _cell_cycle():
    """Infinite shuffled pass over every cell — guarantees the farmer visits
    the whole field evenly within any window of GRID_W*GRID_H steps."""
    cells = [(x, y) for y in range(GRID_H) for x in range(GRID_W)]
    while True:
        random.shuffle(cells)
        for c in cells:
            yield c


def _should_render(ep, total):
    return ep <= 2 or ep % 50 == 0 or ep >= total - 1


def _converged(rewards, best, stable):
    if len(rewards) < ES_MIN_EP:
        return False, best, 0
    recent = float(np.mean(rewards[-ES_WINDOW:]))
    if recent > best + ES_MIN_DELTA:
        return False, recent, 0
    return stable + 1 >= ES_PATIENCE, best, stable + 1


def _animate_step(renderer, screen, clock, farm, farmer, target, hud,
                  recorder, label):
    """Render until farmer arrives, then pause briefly on the action.
    Returns False if the user closed the window."""
    # Walk to target
    frames = 0
    while not farmer.arrived() and frames < ARRIVAL_MAX_FRAMES:
        dt = clock.tick(FPS) / 1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
        renderer.draw(farm, farmer, target, hud, dt)
        pygame.display.flip()
        recorder.capture(screen, label)
        frames += 1
    farmer.px = farmer.target_px
    farmer.py = farmer.target_py

    # Work pause so the action badge is readable
    for _ in range(WORK_FRAMES):
        dt = clock.tick(FPS) / 1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
        renderer.draw(farm, farmer, target, hud, dt)
        pygame.display.flip()
        recorder.capture(screen, label)
    return True


def run_episode(ep, total, farm, farmer, agent, renderer, screen, clock, recorder):
    farm.reset()
    farmer.reset()
    total_reward = 0.0
    money = START_MONEY
    harvested = 0
    withered = 0

    rendered = _should_render(ep, total)
    recorder.start(ep)
    label = f"Ep {ep}   eps={agent.eps:.2f}"
    cells = _cell_cycle()
    animated = 0

    for step in range(MAX_STEPS):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return None

        tx, ty = next(cells)
        s = agent.state(farm, tx, ty, money)
        a = agent.pick(s)
        r, money, done, info = apply_action(farm, farmer, a, tx, ty, money)

        total_reward += r
        harvested += info["harvested"]
        withered += info["withered"]

        animate = (rendered and animated < MAX_ANIMATED_STEPS) or recorder.active()
        if animate:
            renderer.popup(r, tx, ty)
            hud = (ep, total, step, total_reward, money,
                   farmer.action, agent.eps, harvested, withered)
            if not _animate_step(renderer, screen, clock, farm, farmer,
                                 (tx, ty), hud, recorder, label):
                return None
            animated += 1
        else:
            farmer.px = farmer.target_px
            farmer.py = farmer.target_py

        s_next = agent.state(farm, tx, ty, money)
        agent.update(s, a, r, s_next, done)
        if done:
            break

    agent.decay()
    return total_reward


def main():
    pygame.init()
    pygame.display.set_caption("Farm Q-Learning")
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock = pygame.time.Clock()

    farm = Farm()
    farmer = Farmer()
    agent = QLearningAgent()
    renderer = Renderer(screen)
    recorder = GifRecorder(every=max(10, NUM_EPISODES // 8))

    rewards = []
    best = -float("inf")
    stable = 0
    stopped = None
    quit_requested = False

    for ep in range(1, NUM_EPISODES + 1):
        r = run_episode(ep, NUM_EPISODES, farm, farmer, agent,
                        renderer, screen, clock, recorder)
        if r is None:
            quit_requested = True
            break
        rewards.append(r)

        conv, best, stable = _converged(rewards, best, stable)
        if conv:
            stopped = ep
            print(f"[train] converged at ep {ep} — last-{ES_WINDOW} mean "
                  f"{np.mean(rewards[-ES_WINDOW:]):+.1f}")
            break

    pygame.quit()
    recorder.save()
    if rewards:
        plot_rewards(rewards, stopped_early=stopped)
    if quit_requested:
        sys.exit(0)


if __name__ == "__main__":
    main()
