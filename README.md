# Vision Guided Robotic Ball Reaching
**Author:** Swapnadeep Paty

![Banner](assets/github_banner.png)

This repository contains **two different prototypes** of the same project, showing the evolution from a **geometry-based RL baseline** to a **vision-guided search + reach pipeline using PPO** in PyBullet.

---

## System Architecture
![Architecture](assets/architecture.png)

---

## Prototypes

### Prototype 1 — Geometry-Based (Baseline)
- Uses simulator geometry to compute relative position to the ball.
- Base joint performs a distance-based scan to select best orientation.
- PPO controls joints to reach the ball.

![Prototype 1 Workflow](assets/prototype1_workflow.png)

### Prototype 2 — Vision-Based (Final)
- Uses a virtual camera on the end-effector + segmentation mask.
- Search phase rotates base until ball is centered in camera view.
- PPO reaches and touches the ball.

![Prototype 2 Workflow](assets/prototype2_workflow.png)

---

## Demo Screenshots (add later)
Place these files inside `assets/screenshots/`:
- `search_phase.png`
- `reaching_phase.png`
- `contact.png`

Example section (will show once you add images):
![Search](assets/screenshots/search_phase.png)
![Reach](assets/screenshots/reaching_phase.png)
![Contact](assets/screenshots/contact.png)

---

## Demo Videos (add later)
Place these files inside `assets/videos/`:
- `prototype1_demo.mp4`
- `prototype2_demo.mp4`

---

## Installation

```bash
python3 -m venv rl_env
source rl_env/bin/activate
pip install -r requirements.txt