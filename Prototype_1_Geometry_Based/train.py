import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from pathlib import Path

# ===============================
# MODEL PATH (ONLY CHANGE)
# ===============================
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "prototype1_model"  # SB3 adds .zip automatically

# ===============================
# ENV
# ===============================
class ReachEnv(gym.Env):

    def __init__(self):
        super().__init__()

        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.robot_id = None
        self.ball_id = None
        self.prev_dist = None
        self.fixed_base = 0

        self.step_count = 0
        self.max_steps = 150

        # 🔥 COUNTERS ADDED
        self.success_count = 0
        self.fail_count = 0

        self.action_space = spaces.Box(low=-0.04, high=0.04, shape=(3,))
        self.observation_space = spaces.Box(low=-2, high=2, shape=(2,), dtype=np.float32)

        self.active_joints = [1, 3, 5]
        self.locked_joints = [2, 4, 6]

        self.initial_pose = [0, 0.3, 0, -1.2, 0, 1.0, 0]

    # ===========================
    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.step_count = 0

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        p.loadURDF("plane.urdf")

        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0,0,0])

        for i in range(7):
            p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL, force=0)

        for i in range(7):
            p.resetJointState(self.robot_id, i, self.initial_pose[i])

        for _ in range(240):
            p.stepSimulation()
            time.sleep(1./240.)

        # ===========================
        # RANDOM BALL POSITION
        # ===========================
        self.ball_pos = [
            np.random.uniform(0.3000, 0.8000),
            0.05,
            0.05
        ]
        print(f"Ball X: {self.ball_pos[0]:.6f}")

        self.ball_id = p.loadURDF(
            "sphere2.urdf",
            basePosition=self.ball_pos,
            globalScaling=0.15,
            useFixedBase=True
        )

        p.setCollisionFilterPair(self.robot_id, self.ball_id, -1, -1, 0)

        # ===========================
        # SEARCH PHASE
        # ===========================
        best_dist = 1e9
        best_angle = 0

        for angle in range(-60, 61, 2):

            rad = math.radians(angle)

            p.setJointMotorControl2(self.robot_id, 0, p.POSITION_CONTROL, targetPosition=rad)

            for j in range(1, 7):
                p.setJointMotorControl2(
                    self.robot_id, j,
                    p.POSITION_CONTROL,
                    targetPosition=self.initial_pose[j],
                    force=500
                )

            for _ in range(60):
                p.stepSimulation()

            link_state = p.getLinkState(self.robot_id, 6)
            head = link_state[4]

            dist = np.linalg.norm(np.array(head) - np.array(self.ball_pos))

            if dist < best_dist:
                best_dist = dist
                best_angle = rad

        self.fixed_base = best_angle

        p.setJointMotorControl2(self.robot_id, 0, p.POSITION_CONTROL, targetPosition=self.fixed_base)

        for _ in range(100):
            p.stepSimulation()

        return self.get_obs(), {}

    # ===========================
    def get_obs(self):
        link_state = p.getLinkState(self.robot_id, 6)
        head = link_state[4]

        dx = (head[0] - self.ball_pos[0]) / 1.0
        dy = (head[1] - self.ball_pos[1]) / 1.0

        return np.array([dx, dy], dtype=np.float32)

    # ===========================
    def get_dist(self):
        link_state = p.getLinkState(self.robot_id, 6)
        head = link_state[4]
        return np.linalg.norm(np.array(head) - np.array(self.ball_pos))

    # ===========================
    def step(self, action):

        self.step_count += 1

        p.setJointMotorControl2(self.robot_id, 0, p.POSITION_CONTROL, targetPosition=self.fixed_base)

        for j in self.locked_joints:
            p.setJointMotorControl2(
                self.robot_id, j,
                p.POSITION_CONTROL,
                targetPosition=self.initial_pose[j],
                force=500
            )

        for i, j in enumerate(self.active_joints):
            current = p.getJointState(self.robot_id, j)[0]

            p.setJointMotorControl2(
                self.robot_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=current + float(action[i])
            )

        for _ in range(30):
            p.stepSimulation()

        dist = self.get_dist()

        reward = -dist * 10 - 0.01

        terminated = False
        truncated = False

        contacts = p.getContactPoints(self.robot_id, self.ball_id)

        # ===========================
        # SUCCESS
        # ===========================
        if len(contacts) > 0:  #if dist < 0.15 or len(contacts) > 0:
            reward += 100
            terminated = True

            self.success_count += 1
            print(f"✅ Reached: {self.success_count} | ❌ Failed: {self.fail_count}")

        # ===========================
        # FAILURE
        # ===========================
        if self.step_count >= self.max_steps:
            truncated = True

            self.fail_count += 1
            print(f"✅ Reached: {self.success_count} | ❌ Failed: {self.fail_count}")

        return self.get_obs(), reward, terminated, truncated, {}

# ===============================
# TRAIN
# ===============================
if __name__ == "__main__":
    env = ReachEnv()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=128,
        gamma=0.98
    )

    model.learn(total_timesteps=500000)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))

    print(f"\n✅ Saved model as: {MODEL_PATH}.zip")
    print("\n✅ GENERALIZED TRAINING COMPLETE")