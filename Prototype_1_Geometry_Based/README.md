# Prototype 1 — Geometry-Based RL (Baseline)

This is the **first prototype** of the project. It uses **simulator geometry** (true 3D positions from PyBullet) to build the RL observation and to do a base-angle search.

## Workflow
1. Reset robot to a fixed “snake” pose.
2. Spawn ball at a random position (within limits).
3. **Search phase:** rotate base joint (joint 0) across a range and pick the angle that minimizes head↔ball distance.
4. Lock the base at the best angle.
5. PPO controls joints **[1, 3, 5]** to reach the ball.
6. Success is counted when the robot makes contact with the ball (baseline condition).

## Files
- `train.py` → trains PPO and saves model to `models/prototype1_model.zip`
- `test.py` → loads the trained model and runs continuous testing

## Run (from repo root)
Train:
```bash
python3 Prototype_1_Geometry_Based/train.py