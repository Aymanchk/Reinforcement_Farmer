# Farm Q-Learning

A minimal visual Reinforcement-Learning demo: a pixel farmer learns to run a 3×4 farm using tabular Q-Learning.

## Run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

Produces `training.gif` (early vs. trained episode) and `training_progress.png` (learning curve).

## Files

| File | Purpose |
|---|---|
| `game.py` | constants, farm grid, farmer, `apply_action` — pure game logic |
| `agent.py` | tabular Q-Learning agent (ε-greedy, Bellman update) |
| `view.py` | pygame renderer + GIF recorder + matplotlib plot |
| `main.py` | training loop + entry point |

~750 lines total.

## How it works

### State / actions
- State: `(cell_state, watered, money_bucket)` — only ~48 distinct states.
- Actions: `PLOW`, `PLANT_WHEAT`, `PLANT_CARROT`, `PLANT_PUMPKIN`, `WATER`, `HARVEST`, `MOVE`.
- Update: `Q[s][a] += α · (r + γ · max(Q[s']) − Q[s][a])`

### Training loop
Each episode runs `MAX_STEPS` = 120 steps. The target cell is picked by a *shuffled cycle* over all 12 cells, so the farmer visits every plot evenly. The agent picks an action based on the cell's state, runs it, receives a reward, updates Q.

ε starts at 1.0 (pure exploration), decays ×0.97 each episode. Early stopping kicks in once the rolling mean plateaus (~episode 160).

### Rewards
- Successful action (plow/plant/water): +1
- Harvest: +10 / +20 / +40 (wheat / carrot / pumpkin)
- Illegal action for cell state: −1
- Crop wilted: −5
- Bankruptcy (money < 0): −20 and episode ends

## The learning curve

X = Episode, Y = Accumulated reward.

- **Green** segment: the last episode earned positive reward.
- **Red** segment: lost reward (exploration / wrong actions).
- Early episodes wiggle (red dips); once the policy stabilises the line becomes a clean green slope.
- Dotted vertical line marks where early-stopping triggered.
