import math
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# ---------------------------------------------------------
# Paths (GitHub-friendly)
# RepoRoot/models/prototype1_model.zip
# ---------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "prototype1_model"  # load WITHOUT .zip

# ---------------------------------------------------------
# Config (must match training)
# ---------------------------------------------------------
BALL_X_MIN, BALL_X_MAX = 0.30, 0.80
BALL_Y_MIN, BALL_Y_MAX = 0.30, 0.60
BALL_Z = 0.05
BALL_SCALE = 0.15

SEARCH_MIN_DEG = -60
SEARCH_MAX_DEG = 60
SEARCH_STEP_DEG = 2

ACTIVE_JOINTS = [1, 3, 5]
LOCKED_JOINTS = [2, 4, 6]
END_EFFECTOR_LINK = 6

INITIAL_POSE = [0, 0.3, 0, -1.2, 0, 1.0, 0]

MAX_STEPS = 150
SIM_STEPS_PER_ACTION = 30


class ReachEnv(gym.Env):
    def __init__(self):
        super().__init__()
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.robot_id = None
        self.ball_id = None
        self.ball_pos = None
        self.fixed_base = 0.0

        self.step_count = 0
        self.success_count = 0
        self.fail_count = 0

        self.action_space = spaces.Box(low=-0.04, high=0.04, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-2, high=2, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[0, 0, 0])

        for j in range(p.getNumJoints(self.robot_id)):
            p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL, force=0)

        for j, pos in enumerate(INITIAL_POSE):
            p.resetJointState(self.robot_id, j, pos)

        for _ in range(120):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

        self.ball_pos = [
            float(np.random.uniform(BALL_X_MIN, BALL_X_MAX)),
            float(np.random.uniform(BALL_Y_MIN, BALL_Y_MAX)),
            float(BALL_Z),
        ]

        self.ball_id = p.loadURDF(
            "sphere2.urdf",
            basePosition=self.ball_pos,
            globalScaling=BALL_SCALE,
            useFixedBase=True
        )

        # search base
        best_dist = 1e9
        best_angle = 0.0

        for angle_deg in range(SEARCH_MIN_DEG, SEARCH_MAX_DEG + 1, SEARCH_STEP_DEG):
            rad = math.radians(angle_deg)
            p.setJointMotorControl2(self.robot_id, 0, p.POSITION_CONTROL, targetPosition=rad, force=500)

            for j in range(1, 7):
                p.setJointMotorControl2(
                    self.robot_id, j,
                    p.POSITION_CONTROL,
                    targetPosition=INITIAL_POSE[j],
                    force=500
                )

            for _ in range(60):
                p.stepSimulation()

            head = np.array(p.getLinkState(self.robot_id, END_EFFECTOR_LINK)[4], dtype=np.float32)
            dist = float(np.linalg.norm(head - np.array(self.ball_pos, dtype=np.float32)))

            if dist < best_dist:
                best_dist = dist
                best_angle = rad

        self.fixed_base = float(best_angle)
        p.setJointMotorControl2(self.robot_id, 0, p.POSITION_CONTROL, targetPosition=self.fixed_base, force=500)

        for _ in range(80):
            p.stepSimulation()

        return self._get_obs(), {}

    def _get_obs(self):
        head = p.getLinkState(self.robot_id, END_EFFECTOR_LINK)[4]
        dx = float(head[0] - self.ball_pos[0])
        dy = float(head[1] - self.ball_pos[1])
        return np.array([dx, dy], dtype=np.float32)

    def step(self, action):
        self.step_count += 1

        p.setJointMotorControl2(self.robot_id, 0, p.POSITION_CONTROL, targetPosition=self.fixed_base, force=500)

        for j in LOCKED_JOINTS:
            p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=INITIAL_POSE[j], force=500)

        for i, j in enumerate(ACTIVE_JOINTS):
            cur = p.getJointState(self.robot_id, j)[0]
            p.setJointMotorControl2(
                self.robot_id, j,
                p.POSITION_CONTROL,
                targetPosition=cur + float(action[i]),
                force=500
            )

        for _ in range(SIM_STEPS_PER_ACTION):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

        terminated = False
        truncated = False

        if len(p.getContactPoints(self.robot_id, self.ball_id)) > 0:
            terminated = True
            self.success_count += 1
            print(f"✅ Success: {self.success_count} | ❌ Fail: {self.fail_count}")

        if self.step_count >= MAX_STEPS and not terminated:
            truncated = True
            self.fail_count += 1
            print(f"✅ Success: {self.success_count} | ❌ Fail: {self.fail_count}")

        return self._get_obs(), 0.0, terminated, truncated, {}

# ----------------------------
# Run continuous demo testing
# ----------------------------
if __name__ == "__main__":
    env = ReachEnv()
    model = PPO.load(str(MODEL_PATH), env=env)

    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()