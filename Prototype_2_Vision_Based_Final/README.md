# Prototype 2 — Vision-Based Search + Reach (Final)

This is the **final prototype** of the project.

## What this prototype does
1. A **virtual camera** is attached to the robot end-effector/head.
2. The ball is detected using a **segmentation mask** from PyBullet camera.
3. **Search phase:** the robot rotates the base joint until the ball is centered in the camera frame (x≈0).
4. **Reach phase:** PPO controls joints **[1, 3, 5]** to reach and touch the ball.
5. Testing script reports **accuracy** over multiple randomized episodes.

## Files
- `train.py` → fast PPO training (ball aligned in training for speed)
- `test.py` → full real pipeline: random ball X/Y + search + reach + accuracy

## Run (from repo root)
Train:
```bash
python3 Prototype_2_Vision_Based_Final/train.py