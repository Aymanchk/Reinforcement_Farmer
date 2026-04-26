"""All visuals in one file: scene rendering, GIF recording, learning plot."""

import math
import os

import numpy as np
import pygame

from game import (
    WINDOW_W, WINDOW_H, HUD_H,
    GRID_W, GRID_H, CELL_SIZE, FIELD_X, FIELD_Y, FIELD_W, FIELD_H,
    BG, GRASS, PLOWED, SEED_COLOR, SPROUT, WITHERED,
    WHEAT, CARROT, PUMPKIN, HUD_BG, HUD_TEXT,
    S_EMPTY, S_PLOWED, S_SEEDED, S_GROWING, S_READY, S_WITHERED,
    C_WHEAT, C_CARROT, C_PUMPKIN,
)


ACTION_COLORS = {
    "PLOW": (160, 110, 60),
    "PLANT_WHEAT": WHEAT,
    "PLANT_CARROT": CARROT,
    "PLANT_PUMPKIN": PUMPKIN,
    "WATER": (80, 160, 220),
    "HARVEST": (220, 80, 80),
    "MOVE": (180, 180, 190),
    "IDLE": (120, 120, 120),
}
CROP_COLORS = {C_WHEAT: WHEAT, C_CARROT: CARROT, C_PUMPKIN: PUMPKIN}


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------
class _Popup:
    __slots__ = ("text", "x", "y", "color", "life", "max_life")

    def __init__(self, text, x, y, color, life=0.9):
        self.text = text
        self.x = x
        self.y = y
        self.color = color
        self.life = life
        self.max_life = life


class Renderer:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.SysFont("arial", 16, bold=True)
        self.small = pygame.font.SysFont("arial", 12)
        self.big = pygame.font.SysFont("arial", 20, bold=True)
        self.tiny = pygame.font.SysFont("arial", 10, bold=True)
        self._pulse = 0.0
        self._popups = []

    def popup(self, reward, gx, gy):
        if abs(reward) < 0.5:
            return
        x = FIELD_X + gx * CELL_SIZE + CELL_SIZE // 2
        y = FIELD_Y + gy * CELL_SIZE + 10
        color = (80, 220, 110) if reward > 0 else (230, 90, 90)
        text = f"+{reward:.0f}" if reward > 0 else f"{reward:.0f}"
        self._popups.append(_Popup(text, x, y, color))

    def draw(self, farm, farmer, target, hud, dt):
        self._pulse += dt * 4.0
        for p in self._popups:
            p.life -= dt
            p.y -= 40 * dt
        self._popups = [p for p in self._popups if p.life > 0]
        farmer.step(dt)

        self.screen.fill(BG)
        self._field(farm, target)
        self._farmer(farmer)
        self._popups_draw()
        self._hud(hud)

    # ---- field ----
    def _field(self, farm, target):
        pygame.draw.rect(self.screen, (70, 50, 30),
                         (FIELD_X - 4, FIELD_Y - 4, FIELD_W + 8, FIELD_H + 8))
        for gy in range(GRID_H):
            for gx in range(GRID_W):
                self._cell(farm.cell(gx, gy),
                           FIELD_X + gx * CELL_SIZE,
                           FIELD_Y + gy * CELL_SIZE)
        if target is not None:
            gx, gy = target
            thickness = 3 + int((0.5 + 0.5 * math.sin(self._pulse * 1.5)) * 2)
            pygame.draw.rect(self.screen, (255, 230, 80),
                             (FIELD_X + gx * CELL_SIZE - 2,
                              FIELD_Y + gy * CELL_SIZE - 2,
                              CELL_SIZE + 4, CELL_SIZE + 4), thickness)

    def _cell(self, cell, px, py):
        rect = pygame.Rect(px, py, CELL_SIZE, CELL_SIZE)
        s = cell.state

        if s == S_EMPTY:
            pygame.draw.rect(self.screen, GRASS, rect)
        elif s == S_PLOWED:
            pygame.draw.rect(self.screen, PLOWED, rect)
            for i in range(4):
                y = py + 12 + i * 20
                pygame.draw.line(self.screen, (75, 50, 30),
                                 (px + 6, y), (px + CELL_SIZE - 6, y), 2)
        elif s == S_SEEDED:
            pygame.draw.rect(self.screen, PLOWED, rect)
            for dx, dy in ((22, 30), (45, 25), (65, 45), (35, 60), (60, 65)):
                pygame.draw.circle(self.screen, SEED_COLOR, (px + dx, py + dy), 3)
        elif s == S_GROWING:
            pygame.draw.rect(self.screen, PLOWED, rect)
            for sx in (px + 22, px + 45, px + 68):
                sy = py + CELL_SIZE - 14
                pygame.draw.polygon(self.screen, SPROUT,
                                    [(sx - 5, sy), (sx + 5, sy), (sx, sy - 14)])
        elif s == S_READY:
            pygame.draw.rect(self.screen, PLOWED, rect)
            color = CROP_COLORS.get(cell.crop, WHEAT)
            cx = px + CELL_SIZE // 2
            cy = py + CELL_SIZE // 2
            pygame.draw.circle(self.screen, color, (cx, cy), 22)
            pygame.draw.circle(self.screen, (30, 30, 30), (cx, cy), 22, 2)
        elif s == S_WITHERED:
            pygame.draw.rect(self.screen, WITHERED, rect)
            pygame.draw.line(self.screen, (70, 50, 35),
                             (px + 15, py + 20),
                             (px + CELL_SIZE - 15, py + CELL_SIZE - 20), 2)
            pygame.draw.line(self.screen, (70, 50, 35),
                             (px + CELL_SIZE - 15, py + 20),
                             (px + 15, py + CELL_SIZE - 20), 2)

        pygame.draw.rect(self.screen, (30, 25, 15), rect, 1)

        if cell.watered and s in (S_SEEDED, S_GROWING):
            pygame.draw.circle(self.screen, (80, 160, 220),
                               (px + CELL_SIZE - 10, py + 10), 4)

    # ---- farmer ----
    def _farmer(self, farmer):
        cx, cy = int(farmer.px), int(farmer.py)
        pygame.draw.ellipse(self.screen, (0, 0, 0), (cx - 13, cy + 20, 26, 6))
        pygame.draw.rect(self.screen, (60, 90, 150), (cx - 10, cy - 3, 20, 22))
        pygame.draw.circle(self.screen, (240, 200, 160), (cx, cy - 10), 8)
        pygame.draw.ellipse(self.screen, (212, 172, 92), (cx - 12, cy - 20, 24, 7))
        pygame.draw.rect(self.screen, (212, 172, 92), (cx - 6, cy - 24, 12, 6))
        self._badge(farmer.action, cx, cy - 36)

    def _badge(self, action, cx, cy):
        if not action or action == "IDLE":
            return
        label = action.replace("PLANT_", "")
        text = self.tiny.render(label, True, (20, 20, 20))
        tw, th = text.get_size()
        bw, bh = tw + 20, th + 6
        bx, by = cx - bw // 2, cy - bh // 2
        color = ACTION_COLORS.get(action, (200, 200, 200))
        pygame.draw.rect(self.screen, (250, 248, 240), (bx, by, bw, bh), border_radius=4)
        pygame.draw.rect(self.screen, color, (bx, by, bw, bh), 2, border_radius=4)
        pygame.draw.circle(self.screen, color, (bx + 8, cy), 3)
        self.screen.blit(text, (bx + 14, by + 3))

    def _popups_draw(self):
        for p in self._popups:
            alpha = int(max(0, min(1, p.life / p.max_life)) * 255)
            surf = self.big.render(p.text, True, p.color)
            surf.set_alpha(alpha)
            rect = surf.get_rect(center=(int(p.x), int(p.y)))
            self.screen.blit(surf, rect)

    def _hud(self, hud):
        episode, total, step, reward, money, action, eps, harvested, withered = hud
        pygame.draw.rect(self.screen, HUD_BG, (0, 0, WINDOW_W, HUD_H))

        self.screen.blit(self.small.render(
            f"Ep {episode}/{total}   step {step}   eps={eps:.2f}",
            True, HUD_TEXT), (15, 8))

        r_color = (120, 230, 120) if reward >= 0 else (230, 120, 120)
        a_color = ACTION_COLORS.get(action, (220, 220, 220))

        self.screen.blit(self.small.render(f"${money}", True, (255, 220, 100)), (15, 28))
        self.screen.blit(self.small.render(f"r={reward:+.0f}", True, r_color), (70, 28))
        self.screen.blit(self.small.render(action, True, a_color), (140, 28))
        self.screen.blit(self.small.render(
            f"harv:{harvested}  with:{withered}", True, HUD_TEXT),
            (WINDOW_W - 150, 28))


# ---------------------------------------------------------------------
# GIF recorder: two buffers — episode 1 (early) and the most recent
# periodic episode (late). Saved together so you see the contrast.
# ---------------------------------------------------------------------
class GifRecorder:
    def __init__(self, out_path="training.gif", scale=0.6,
                 stride=2, fps=14, max_frames=180, every=60):
        self.out_path = out_path
        self.scale = scale
        self.stride = max(1, int(stride))
        self.fps = fps
        self.max_frames = max_frames
        self.every = max(1, int(every))
        self._early = []
        self._late = []
        self._cur = []
        self._kind = None
        self._counter = 0

    def start(self, ep):
        self._commit()
        self._cur = []
        self._counter = 0
        if ep == 1:
            self._kind = "early"
        elif ep % self.every == 0:
            self._kind = "late"
        else:
            self._kind = None

    def _commit(self):
        if not self._cur or self._kind is None:
            return
        if self._kind == "early" and not self._early:
            self._early = self._cur
        elif self._kind == "late":
            self._late = self._cur

    def active(self):
        return self._kind is not None and len(self._cur) < self.max_frames

    def capture(self, screen, label=None):
        if not self.active():
            return
        self._counter += 1
        if (self._counter - 1) % self.stride != 0:
            return
        arr = np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2))
        if self.scale != 1.0:
            h, w = arr.shape[:2]
            nh, nw = int(h * self.scale), int(w * self.scale)
            ys = np.linspace(0, h - 1, nh).astype(np.int32)
            xs = np.linspace(0, w - 1, nw).astype(np.int32)
            arr = arr[ys[:, None], xs[None, :]]
        if label:
            arr = self._stamp(arr, label)
        self._cur.append(arr.astype(np.uint8))

    @staticmethod
    def _stamp(arr, label):
        h, w = arr.shape[:2]
        sh = 20
        surf = pygame.Surface((w, sh))
        surf.fill((18, 22, 30))
        font = pygame.font.SysFont("arial", 12, bold=True)
        surf.blit(font.render(label, True, (240, 240, 220)), (8, 3))
        strip = np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))
        out = arr.copy()
        out[:sh] = strip
        return out

    def save(self):
        self._commit()
        frames = list(self._early) + list(self._late)
        if not frames:
            return False
        try:
            import imageio.v2 as imageio
        except ImportError:
            print("[gif] imageio missing; skipping")
            return False
        imageio.mimsave(self.out_path, frames, duration=1.0 / self.fps, loop=0)
        size_kb = os.path.getsize(self.out_path) / 1024
        print(f"[gif] {len(frames)} frames -> {self.out_path} ({size_kb:.0f} KB)")
        return True


# ---------------------------------------------------------------------
# Learning curve: rolling-mean reward per episode.
#   * line rises  -> agent is improving
#   * line flat   -> agent has converged (no more learning to do)
#   * line noisy  -> still exploring
# ---------------------------------------------------------------------
def plot_rewards(rewards, save_path="training_progress.png", stopped_early=None):
    import matplotlib.pyplot as plt

    r = np.array(rewards, dtype=float)
    n = len(r)
    if n == 0:
        return
    ep = np.arange(1, n + 1)

    # Running mean from episode 1 onwards: smoothed[i] = mean(r[:i+1]).
    # By construction this is monotonic-ish — it can only move as much as a
    # single new episode pulls the mean — so the curve rises while the
    # agent improves, then flattens as each new episode has a smaller
    # effect on the accumulated average.
    smoothed = np.cumsum(r) / np.arange(1, n + 1)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.scatter(ep, r, s=12, alpha=0.18, color="#4a90d9",
               edgecolors="none", label="Per-episode reward")
    ax.plot(ep, smoothed, linewidth=3.2, color="#1f7a3e",
            label="Smoothed reward (learning curve)")
    ax.axhline(0, color="#555", linewidth=1, linestyle="--", alpha=0.5)

    if stopped_early and 1 <= stopped_early <= n:
        ax.axvline(stopped_early, color="#b8860b", linestyle=":",
                   linewidth=1.6, alpha=0.9)
        ax.annotate(
            f"learned & stopped (ep {stopped_early})",
            xy=(stopped_early, r[stopped_early - 1]),
            xytext=(-12, 20), textcoords="offset points",
            ha="right", fontsize=10, color="#8a6508",
            arrowprops=dict(arrowstyle="->", color="#b8860b", lw=1),
        )

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Reward per episode", fontsize=12)
    ax.set_title(
        "Learning curve — rises while learning, flattens once converged",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, n)

    last = min(50, n)
    fig.text(
        0.5, 0.012,
        f"Episodes: {n}   "
        f"Last-{last} avg: {r[-last:].mean():+.1f}   "
        f"Best: {r.max():+.1f}   "
        f"Worst: {r.min():+.1f}",
        ha="center", fontsize=10, style="italic", color="#333",
    )

    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
