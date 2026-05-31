# Vision Guided Robotic Ball Reaching

**Author:** Swapnadeep Paty

![Banner](assets/github_banner.png)

A PyBullet + PPO (Stable-Baselines3) project demonstrating an end-to-end robotic manipulation pipeline:

**Search → Align → Reach → Contact**

This repository contains two different prototypes of the same project, showing the evolution from a geometry-based reinforcement learning baseline to a vision-guided robotic reaching system.

---

# System Architecture

![Architecture](assets/architecture.png)

---

# Prototypes

## Prototype 1 — Geometry-Based (Baseline)

### Overview

* Uses true simulator geometry to estimate the relative position of the target ball.
* Performs a distance-based base scan to find the optimal orientation.
* PPO controls the robot arm joints to reach the target.

### Features

* Geometry-based observation space
* Base-angle search phase
* PPO-based reaching policy
* Randomized ball positions
* PyBullet simulation environment

![Prototype 1 Workflow](assets/prototype1_workflow.png)

---

## Prototype 2 — Vision-Based (Final)

### Overview

* Uses a virtual camera attached to the robot end-effector.
* Detects the target using PyBullet segmentation masks.
* Performs camera-based search and alignment.
* PPO controls the robot arm to reach the target.

### Features

* Vision-guided search phase
* Segmentation-based target detection
* Camera-centered alignment
* PPO-based reaching policy
* End-effector contact validation
* Performance statistics and accuracy reporting

![Prototype 2 Workflow](assets/prototype2_workflow.png)

---

# Demo Screenshots

| Search Phase                                   | Reaching Phase                                  | Contact                                    |
| ---------------------------------------------- | ----------------------------------------------- | ------------------------------------------ |
| ![Search](assets/screenshots/search_phase.png) | ![Reach](assets/screenshots/reaching_phase.png) | ![Contact](assets/screenshots/contact.png) |

---

# Demo Videos

### Prototype 1 Demo

`assets/videos/prototype1_demo.mp4`

### Prototype 2 Demo

`assets/videos/prototype2_demo.mp4`

> If GitHub does not preview the videos directly in the README, open the files from the repository to view or download them.

---

# Results

Detailed evaluation outputs are available in:

* `results/prototype1_test_results.txt`
* `results/prototype2_test_results.txt`

---

# Installation

```bash
python3 -m venv rl_env
source rl_env/bin/activate
pip install -r requirements.txt
```

### Quick Sanity Check

```bash
python -c "import pybullet, gymnasium, stable_baselines3, torch; print('All OK')"
```

Expected output:

```text
All OK
```

---

# Run Prototype 1

### Train

```bash
python3 Prototype_1_Geometry_Based/train.py
```

### Test

```bash
python3 Prototype_1_Geometry_Based/test.py
```

---

# Run Prototype 2 (Final)

### Train

```bash
python3 Prototype_2_Vision_Based_Final/train.py
```

### Test (Search + Reach + Accuracy)

```bash
python3 Prototype_2_Vision_Based_Final/test.py
```

---

# Models

Pre-trained models are included in this repository:

* `models/prototype1_model.zip`
* `models/prototype2_model.zip`

> Stable-Baselines3 loads models using the filename without the `.zip` extension.

Example:

```python
model = PPO.load("models/prototype2_model")
```

---

# Repository Structure

```text
Vision-Guided-Robotic-Ball-Reaching-RL/
│
├── Prototype_1_Geometry_Based/
│   ├── train.py
│   ├── test.py
│   └── README.md
│
├── Prototype_2_Vision_Based_Final/
│   ├── train.py
│   ├── test.py
│   └── README.md
│
├── models/
│   ├── prototype1_model.zip
│   └── prototype2_model.zip
│
├── results/
│   ├── prototype1_test_results.txt
│   └── prototype2_test_results.txt
│
├── assets/
│   ├── github_banner.png
│   ├── architecture.png
│   ├── prototype1_workflow.png
│   ├── prototype2_workflow.png
│   ├── screenshots/
│   │   ├── search_phase.png
│   │   ├── reaching_phase.png
│   │   └── contact.png
│   │
│   └── videos/
│       ├── prototype1_demo.mp4
│       └── prototype2_demo.mp4
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

# Technical Stack

* Python
* PyBullet
* Gymnasium
* Stable-Baselines3 (PPO)
* NumPy
* Reinforcement Learning
* Computer Vision
* Robotic Manipulation

---

# Future Work

* Sim-to-real transfer
* Real robot deployment
* Multi-object target selection
* Depth-assisted perception
* Improved reward shaping
* Dynamic environments
* Mobile manipulation platform

---

# License

This project is licensed under the MIT License.

See the `LICENSE` file for details.

If you use this work in research, please cite the repository and acknowledge the original author.
