"""Farm game: constants, world state, and the action mechanics.

The agent sees the world through `apply_action` and the cell states below.
Nothing here depends on pygame — this module is pure game logic.
"""

from __future__ import annotations

import math
from typing import TypedDict

WINDOW_W, WINDOW_H = 640, 540
HUD_H = 50
CELL_SIZE = 90
GRID_W, GRID_H = 3, 4
FIELD_W = GRID_W * CELL_SIZE
FIELD_H = GRID_H * CELL_SIZE
FIELD_X = (WINDOW_W - FIELD_W) // 2
FIELD_Y = HUD_H + 25
FPS = 30

BG = (34, 42, 50)
GRASS = (95, 145, 65)
PLOWED = (115, 75, 45)
SEED_COLOR = (50, 35, 20)
SPROUT = (80, 180, 70)
WITHERED = (125, 100, 75)
WHEAT = (232, 200, 70)
CARROT = (232, 120, 40)
PUMPKIN = (222, 120, 30)
HUD_BG = (40, 50, 60)
HUD_TEXT = (240, 240, 220)

S_EMPTY, S_PLOWED, S_SEEDED, S_GROWING, S_READY, S_WITHERED = range(6)

C_NONE, C_WHEAT, C_CARROT, C_PUMPKIN = range(4)
GROW_STEPS = {C_WHEAT: 5, C_CARROT: 8, C_PUMPKIN: 12}
PRICE = {C_WHEAT: 10, C_CARROT: 20, C_PUMPKIN: 40}

WITHER_STEPS = 40
PLANT_COST = 5
START_MONEY = 50

A_PLOW, A_WHEAT, A_CARROT, A_PUMPKIN, A_WATER, A_HARVEST, A_MOVE = range(7)
NUM_ACTIONS = 7
ACTION_NAMES = ["PLOW", "PLANT_WHEAT", "PLANT_CARROT",
                "PLANT_PUMPKIN", "WATER", "HARVEST", "MOVE"]
_ACTION_TO_CROP = {A_WHEAT: C_WHEAT, A_CARROT: C_CARROT, A_PUMPKIN: C_PUMPKIN}

R_SUCCESS = 1
R_ILLEGAL = -1
R_WITHER = -5
R_BANKRUPT = -20
R_MOVE = -0.05


class Cell:
    __slots__ = ("state", "crop", "grow", "ready", "watered")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.state: int = S_EMPTY
        self.crop: int = C_NONE
        self.grow: int = 0
        self.ready: int = 0
        self.watered: bool = False


class Farm:
    def __init__(self) -> None:
        self.grid: list[list[Cell]] = [
            [Cell() for _ in range(GRID_W)] for _ in range(GRID_H)
        ]

    def reset(self) -> None:
        for row in self.grid:
            for c in row:
                c.reset()

    def cell(self, x: int, y: int) -> Cell:
        return self.grid[y][x]

    def tick(self) -> int:
        """Advance biology by one step. Returns number of newly-withered cells."""
        withered = 0
        for row in self.grid:
            for c in row:
                if c.state == S_SEEDED:
                    c.state = S_GROWING
                    c.grow = 1
                elif c.state == S_GROWING:
                    c.grow += 1
                    need = GROW_STEPS[c.crop]
                    if (c.watered and c.grow >= need - 1) or c.grow >= need:
                        c.state = S_READY
                        c.ready = 0
                        c.watered = False
                elif c.state == S_READY:
                    c.ready += 1
                    if c.ready >= WITHER_STEPS:
                        c.state = S_WITHERED
                        withered += 1
        return withered


class Farmer:
    TRAVEL_TIME = 0.4

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.x: int = 0
        self.y: int = 0
        self.px: float = FIELD_X + CELL_SIZE // 2
        self.py: float = FIELD_Y + CELL_SIZE // 2
        self.target_px: float = self.px
        self.target_py: float = self.py
        self.speed: float = 0.0
        self.action: str = "IDLE"

    def go_to(self, gx: int, gy: int) -> None:
        self.x = gx
        self.y = gy
        tp_x = FIELD_X + gx * CELL_SIZE + CELL_SIZE // 2
        tp_y = FIELD_Y + gy * CELL_SIZE + CELL_SIZE // 2
        dist = math.hypot(tp_x - self.px, tp_y - self.py)
        self.target_px = tp_x
        self.target_py = tp_y
        self.speed = max(60.0, dist / self.TRAVEL_TIME) if dist > 0 else 0.0

    def step(self, dt: float) -> None:
        dx = self.target_px - self.px
        dy = self.target_py - self.py
        dist = math.hypot(dx, dy)
        if dist < 1:
            self.px = self.target_px
            self.py = self.target_py
            return
        s = min(dist, self.speed * dt)
        self.px += dx / dist * s
        self.py += dy / dist * s

    def snap_to_target(self) -> None:
        """Teleport to the current target — used to skip animation."""
        self.px = self.target_px
        self.py = self.target_py

    def arrived(self) -> bool:
        return (abs(self.target_px - self.px) < 1
                and abs(self.target_py - self.py) < 1)


class StepInfo(TypedDict):
    harvested: bool
    withered: int


def apply_action(
    farm: Farm,
    farmer: Farmer,
    action: int,
    tx: int,
    ty: int,
    money: int,
) -> tuple[float, int, bool, StepInfo]:
    """Execute `action` at cell (tx, ty). Returns (reward, money, done, info)."""
    reward = 0.0
    done = False
    harvested = False
    cell = farm.cell(tx, ty)
    farmer.go_to(tx, ty)
    farmer.action = ACTION_NAMES[action]

    if action == A_PLOW:
        if cell.state in (S_EMPTY, S_WITHERED):
            cell.reset()
            cell.state = S_PLOWED
            reward += R_SUCCESS
        else:
            reward += R_ILLEGAL
    elif action in _ACTION_TO_CROP:
        crop = _ACTION_TO_CROP[action]
        if cell.state == S_PLOWED and money >= PLANT_COST:
            cell.state = S_SEEDED
            cell.crop = crop
            cell.grow = 0
            money -= PLANT_COST
            reward += R_SUCCESS
        else:
            reward += R_ILLEGAL
    elif action == A_WATER:
        if cell.state in (S_SEEDED, S_GROWING):
            cell.watered = True
            reward += R_SUCCESS
        else:
            reward += R_ILLEGAL
    elif action == A_HARVEST:
        if cell.state == S_READY:
            money += PRICE[cell.crop]
            reward += PRICE[cell.crop]
            cell.reset()
            harvested = True
        else:
            reward += R_ILLEGAL
    elif action == A_MOVE:
        reward += R_MOVE

    withered_count = farm.tick()
    reward += R_WITHER * withered_count

    if money < 0:
        reward += R_BANKRUPT
        done = True

    return reward, money, done, {"harvested": harvested, "withered": withered_count}
