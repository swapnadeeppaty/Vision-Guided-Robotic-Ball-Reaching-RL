# final_train_no_search_with_counters.py
# ------------------------------------------------------------
# FINAL TRAINING CODE (NO SEARCH PHASE) + SUCCESS/FAIL COUNTERS
# (ORIGINAL LOGIC UNCHANGED)
# ------------------------------------------------------------

import math
import time
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from pathlib import Path

# ===============================
# MODEL PATH (ONLY CHANGE)
# ===============================
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "prototype2_model"   # SB3 adds .zip automatically

# =========================================================
# CAMERA / SEGMENTATION SETTINGS (small = fast)
# =========================================================
WIDTH, HEIGHT = 320, 240
CX0, CY0 = WIDTH // 2, HEIGHT // 2
FOV = 90
NEAR, FAR = 0.01, 5.0

# =========================================================
# ROBOT: Snake pose defaults
# =========================================================
SNAKE_J1 = 0.3
SNAKE_J3 = -1.2
SNAKE_J5 = 1.5

# Locked joints
LOCK_J2 = 0.0
LOCK_J4 = 0.0
LOCK_J6 = 0.0

# Base fixed (no search)
FIXED_BASE_ANGLE_DEG = 0.0
FIXED_BASE_ANGLE = math.radians(FIXED_BASE_ANGLE_DEG)

# =========================================================
# BALL: NO SEARCH (aligned by construction)
# Fix Y=0.00, randomize only X
# =========================================================
BALL_X_MIN, BALL_X_MAX = 0.30, 0.80
BALL_Y_FIXED = 0.00
BALL_Z = 0.05
BALL_SCALE = 0.15

# =========================================================
# VIRTUAL HEAD CAMERA (your verified frame)
# =========================================================
CAM_OFFSET = 0.05
CAM_LOOK_AHEAD = 0.5

# =========================================================
# PPO CONTROL
# =========================================================
ACTIVE_JOINTS = [1, 3, 5]
END_EFFECTOR_LINK = 6

ACTION_MAX_DELTA = 0.04
MAX_EPISODE_STEPS = 180

# =========================================================
# SUCCESS (VALID)
# =========================================================
SUCCESS_DIST = 0.05  # Start with 5cm for easier learning; later reduce to 0.03 then 0.02
SUCCESS_BONUS = 200.0

# =========================================================
# REWARD (Option 3 style)
# =========================================================
DIST_GAIN = 20.0
TIME_PENALTY = -0.01
VISIBLE_BONUS = 0.05
LOST_PENALTY = -1.0

# Stronger attraction when close:
PROX_GAIN = 0.20
PROX_EPS = 0.05

# =========================================================
# MOTOR / SIM
# =========================================================
BASE_FORCE = 500
HOLD_FORCE = 1200
SIM_STEPS_PER_ACTION = 6

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def norm_xy(cx, cy):
    x_norm = (cx - CX0) / float(CX0)
    y_norm = (cy - CY0) / float(CY0)
    return x_norm, y_norm


class ReachNoSearchEnv(gym.Env):
    """
    NO SEARCH PHASE.
    Ball is already aligned by construction (Ball Y fixed).
    """

    def __init__(self, render_mode="none"):
        super().__init__()
        self.render_mode = render_mode
        self.cid = None
        self.robot_id = None
        self.ball_id = None

        self.step_count = 0
        self.prev_dist = None

        # Counters
        self.success_count = 0
        self.fail_count = 0
        self.success_by_distance = 0
        self.success_by_contact = 0

        # Observation:
        # [ball_y_norm, area_norm, dist_norm, j1, j3, j5]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, -3.5, -3.5, -3.5], dtype=np.float32),
            high=np.array([ 1.0, 1.0, 1.0,  3.5,  3.5,  3.5], dtype=np.float32),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-ACTION_MAX_DELTA,
            high= ACTION_MAX_DELTA,
            shape=(3,),
            dtype=np.float32
        )

        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=FOV,
            aspect=WIDTH / HEIGHT,
            nearVal=NEAR,
            farVal=FAR
        )

    def _connect(self):
        if self.cid is not None:
            return

        if self.render_mode == "human":
            self.cid = p.connect(p.GUI)
        else:
            self.cid = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def _reset_sim(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[0, 0, 0])

        # Disable default motors
        for j in range(p.getNumJoints(self.robot_id)):
            p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL, force=0)

        # Ball: X randomized, Y fixed
        bx = float(np.random.uniform(BALL_X_MIN, BALL_X_MAX))
        by = float(BALL_Y_FIXED)

        self.ball_id = p.loadURDF(
            "sphere2.urdf",
            basePosition=[bx, by, BALL_Z],
            globalScaling=BALL_SCALE,
            useFixedBase=True
        )

    def _lock_base_and_locked_joints(self):
        # Base fixed
        p.setJointMotorControl2(self.robot_id, 0, p.POSITION_CONTROL,
                                targetPosition=FIXED_BASE_ANGLE, force=BASE_FORCE)

        # Locked joints fixed
        p.setJointMotorControl2(self.robot_id, 2, p.POSITION_CONTROL, targetPosition=LOCK_J2, force=HOLD_FORCE)
        p.setJointMotorControl2(self.robot_id, 4, p.POSITION_CONTROL, targetPosition=LOCK_J4, force=HOLD_FORCE)
        p.setJointMotorControl2(self.robot_id, 6, p.POSITION_CONTROL, targetPosition=LOCK_J6, force=HOLD_FORCE)

    def _set_initial_active_joints(self):
        p.setJointMotorControl2(self.robot_id, 1, p.POSITION_CONTROL, targetPosition=SNAKE_J1, force=HOLD_FORCE)
        p.setJointMotorControl2(self.robot_id, 3, p.POSITION_CONTROL, targetPosition=SNAKE_J3, force=HOLD_FORCE)
        p.setJointMotorControl2(self.robot_id, 5, p.POSITION_CONTROL, targetPosition=SNAKE_J5, force=HOLD_FORCE)

    def _get_view_from_head(self):
        link_state = p.getLinkState(self.robot_id, END_EFFECTOR_LINK, computeForwardKinematics=True)
        link_pos = np.array(link_state[4], dtype=np.float32)
        link_orn = link_state[5]
        rot = np.array(p.getMatrixFromQuaternion(link_orn), dtype=np.float32).reshape(3, 3)

        forward = rot[:, 2]
        up = -rot[:, 0]

        cam_pos = link_pos + forward * CAM_OFFSET
        cam_target = cam_pos + forward * CAM_LOOK_AHEAD

        view_matrix = p.computeViewMatrix(cam_pos.tolist(), cam_target.tolist(), up.tolist())
        return view_matrix

    def _capture_seg(self):
        view_matrix = self._get_view_from_head()
        renderer = p.ER_BULLET_HARDWARE_OPENGL if self.render_mode == "human" else p.ER_TINY_RENDERER

        img = p.getCameraImage(
            WIDTH, HEIGHT,
            viewMatrix=view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=renderer
        )

        seg = np.array(img[4]).reshape((HEIGHT, WIDTH))
        return seg

    def _vision_features(self):
        seg = self._capture_seg()
        mask = (seg == self.ball_id)

        if not np.any(mask):
            return False, 0.0, 0.0

        ys, xs = np.where(mask)
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))
        area = int(xs.shape[0])

        _, y_norm = norm_xy(cx, cy)
        area_norm = min(1.0, area / float(WIDTH * HEIGHT))
        return True, float(y_norm), float(area_norm)

    def _ee_pos(self):
        return np.array(
            p.getLinkState(self.robot_id, END_EFFECTOR_LINK, computeForwardKinematics=True)[4],
            dtype=np.float32
        )

    def _ball_pos(self):
        return np.array(p.getBasePositionAndOrientation(self.ball_id)[0], dtype=np.float32)

    def _distance_to_ball(self):
        return float(np.linalg.norm(self._ee_pos() - self._ball_pos()))

    def _eef_contact_only(self):
        contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.ball_id)
        for c in contacts:
            linkA = c[3]
            if linkA == END_EFFECTOR_LINK:
                return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._connect()
        self._reset_sim()

        self.step_count = 0

        self._lock_base_and_locked_joints()
        self._set_initial_active_joints()

        # settle
        for _ in range(10):
            p.stepSimulation()

        self.prev_dist = self._distance_to_ball()

        visible, y_norm, area_norm = self._vision_features()
        dist_norm = clamp(self.prev_dist / 1.0, 0.0, 1.0)

        j1 = p.getJointState(self.robot_id, 1)[0]
        j3 = p.getJointState(self.robot_id, 3)[0]
        j5 = p.getJointState(self.robot_id, 5)[0]

        obs = np.array([y_norm, area_norm, dist_norm, j1, j3, j5], dtype=np.float32)
        return obs, {}

    def step(self, action):
        self.step_count += 1

        self._lock_base_and_locked_joints()

        # Apply delta actions to joints 1,3,5
        for i, j in enumerate(ACTIVE_JOINTS):
            cur = p.getJointState(self.robot_id, j)[0]
            tgt = cur + float(action[i])
            p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=tgt, force=HOLD_FORCE)

        for _ in range(SIM_STEPS_PER_ACTION):
            p.stepSimulation()

        visible, y_norm, area_norm = self._vision_features()

        dist = self._distance_to_ball()
        reward = (self.prev_dist - dist) * DIST_GAIN
        self.prev_dist = dist

        reward += PROX_GAIN * (1.0 / (dist + PROX_EPS))
        reward += TIME_PENALTY
        reward += VISIBLE_BONUS if visible else LOST_PENALTY

        terminated = False
        truncated = False

        if dist <= SUCCESS_DIST:
            reward += SUCCESS_BONUS
            terminated = True

            self.success_count += 1
            self.success_by_distance += 1

            print("\n====================")
            print("✅ SUCCESS (DISTANCE)")
            print(f"Distance = {dist:.4f} m")
            print(f"Success = {self.success_count} | Fail = {self.fail_count}")
            print(f"Distance Success = {self.success_by_distance} | Contact Success = {self.success_by_contact}")
            print("====================\n")

        elif self._eef_contact_only():
            reward += SUCCESS_BONUS
            terminated = True

            self.success_count += 1
            self.success_by_contact += 1

            print("\n====================")
            print("✅ SUCCESS (EEF CONTACT)")
            print(f"Distance = {dist:.4f} m")
            print(f"Success = {self.success_count} | Fail = {self.fail_count}")
            print(f"Distance Success = {self.success_by_distance} | Contact Success = {self.success_by_contact}")
            print("====================\n")

        if self.step_count >= MAX_EPISODE_STEPS and not terminated:
            truncated = True
            self.fail_count += 1

            print("\n====================")
            print("❌ FAIL (TIMEOUT)")
            print(f"Distance = {dist:.4f} m")
            print(f"Success = {self.success_count} | Fail = {self.fail_count}")
            print(f"Distance Success = {self.success_by_distance} | Contact Success = {self.success_by_contact}")
            print("====================\n")

        dist_norm = clamp(dist / 1.0, 0.0, 1.0)

        j1 = p.getJointState(self.robot_id, 1)[0]
        j3 = p.getJointState(self.robot_id, 3)[0]
        j5 = p.getJointState(self.robot_id, 5)[0]
        obs = np.array([y_norm, area_norm, dist_norm, j1, j3, j5], dtype=np.float32)

        info = {"visible": visible, "dist": dist}
        return obs, float(reward), terminated, truncated, info

    def close(self):
        if self.cid is not None and p.isConnected(self.cid):
            p.disconnect(self.cid)
            self.cid = None


# =========================================================
# TRAIN (ORIGINAL FLOW UNCHANGED, ONLY PATH UPDATED)
# =========================================================
if __name__ == "__main__":
    env = ReachNoSearchEnv(render_mode="none")

    model = PPO.load(
        str(MODEL_PATH),   # ONLY CHANGE (path)
        env=env
    )

    try:
        model.learn(
            total_timesteps=300000,
            reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        print("\n⏹ Stopped by user (Ctrl+C). Saving model...")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))
    print(f"\n✅ Saved model as: {MODEL_PATH}.zip")