import math
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
from stable_baselines3 import PPO

# =========================================================
# PATHS (GitHub-friendly)
# RepoRoot/models/prototype2_model.zip
# =========================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "prototype2_model"   # IMPORTANT: no .zip here

# =========================================================
# CAMERA SETTINGS (must match training)
# =========================================================
WIDTH, HEIGHT = 320, 240
CX0, CY0 = WIDTH // 2, HEIGHT // 2
FOV = 90
NEAR, FAR = 0.01, 5.0

# =========================================================
# ROBOT POSE (snake defaults)
# =========================================================
SNAKE_J1 = 0.3
SNAKE_J3 = -1.2
SNAKE_J5 = 1.5
LOCK_J2 = 0.0
LOCK_J4 = 0.0
LOCK_J6 = 0.0

# =========================================================
# BALL RANDOM RANGE (both axes)
# =========================================================
BALL_X_MIN, BALL_X_MAX = 0.30, 0.80
BALL_Y_MIN, BALL_Y_MAX = -0.25, 0.25
BALL_Z = 0.05
BALL_SCALE = 0.15

# =========================================================
# CAMERA ATTACH (verified frame)
# =========================================================
CAM_OFFSET = 0.05
CAM_LOOK_AHEAD = 0.5

# =========================================================
# SEARCH SETTINGS (center x -> 0)
# =========================================================
SCAN_MIN_DEG = -60
SCAN_MAX_DEG = 60
SCAN_STEP = 0.01

PIX_TOL = 1
STABLE_FRAMES_REQUIRED = 6

KP = 0.00030
MAX_STEP = 0.01
MIN_STEP = 0.0006
SETTLE_STEPS_BEFORE_CAPTURE = 4
MAX_SEARCH_ITERS = 3000

# =========================================================
# REACH SETTINGS
# =========================================================
ACTIVE_JOINTS = [1, 3, 5]
END_EFFECTOR_LINK = 6
MAX_EP_STEPS = 180

# Valid success
SUCCESS_DIST = 0.05   # keep it (even if it stays 0 in stats)
BASE_FORCE = 500
HOLD_FORCE = 1200
SIM_STEPS_PER_ACTION = 6


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def norm_xy(cx, cy):
    x_norm = (cx - CX0) / float(CX0)
    y_norm = (cy - CY0) / float(CY0)
    return x_norm, y_norm


def eef_contact_only(robot_id, ball_id):
    """Count success only if robot link == END_EFFECTOR_LINK touches the ball."""
    contacts = p.getContactPoints(bodyA=robot_id, bodyB=ball_id)
    for c in contacts:
        linkA = c[3]  # robot link index
        if linkA == END_EFFECTOR_LINK:
            return True
    return False


def ee_pos(robot_id):
    return np.array(
        p.getLinkState(robot_id, END_EFFECTOR_LINK, computeForwardKinematics=True)[4],
        dtype=np.float32
    )


def ball_pos(ball_id):
    return np.array(p.getBasePositionAndOrientation(ball_id)[0], dtype=np.float32)


def dist_to_ball(robot_id, ball_id):
    return float(np.linalg.norm(ee_pos(robot_id) - ball_pos(ball_id)))


def get_view_from_head(robot_id):
    link_state = p.getLinkState(robot_id, END_EFFECTOR_LINK, computeForwardKinematics=True)
    link_pos = np.array(link_state[4], dtype=np.float32)
    link_orn = link_state[5]

    rot = np.array(p.getMatrixFromQuaternion(link_orn), dtype=np.float32).reshape(3, 3)

    forward = rot[:, 2]
    up = -rot[:, 0]

    cam_pos = link_pos + forward * CAM_OFFSET
    cam_target = cam_pos + forward * CAM_LOOK_AHEAD

    return p.computeViewMatrix(
        cameraEyePosition=cam_pos.tolist(),
        cameraTargetPosition=cam_target.tolist(),
        cameraUpVector=up.tolist()
    )


def capture_seg(robot_id, projection_matrix):
    view_matrix = get_view_from_head(robot_id)
    img = p.getCameraImage(
        WIDTH, HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL  # GUI mode
    )
    seg = np.array(img[4]).reshape((HEIGHT, WIDTH))
    return seg


def vision_features(robot_id, ball_id, projection_matrix):
    """Returns: visible, y_norm, area_norm, x_norm"""
    seg = capture_seg(robot_id, projection_matrix)
    mask = (seg == ball_id)

    if not np.any(mask):
        return False, 0.0, 0.0, 0.0

    ys, xs = np.where(mask)
    cx = int(np.mean(xs))
    cy = int(np.mean(ys))
    area = int(xs.shape[0])

    x_norm, y_norm = norm_xy(cx, cy)
    area_norm = min(1.0, area / float(WIDTH * HEIGHT))
    return True, float(y_norm), float(area_norm), float(x_norm)


def hold_snake_pose(robot_id, base_ang):
    # base
    p.setJointMotorControl2(robot_id, 0, p.POSITION_CONTROL, targetPosition=base_ang, force=BASE_FORCE)

    # snake defaults + locked joints
    p.setJointMotorControl2(robot_id, 1, p.POSITION_CONTROL, targetPosition=SNAKE_J1, force=HOLD_FORCE)
    p.setJointMotorControl2(robot_id, 2, p.POSITION_CONTROL, targetPosition=LOCK_J2, force=HOLD_FORCE)
    p.setJointMotorControl2(robot_id, 3, p.POSITION_CONTROL, targetPosition=SNAKE_J3, force=HOLD_FORCE)
    p.setJointMotorControl2(robot_id, 4, p.POSITION_CONTROL, targetPosition=LOCK_J4, force=HOLD_FORCE)
    p.setJointMotorControl2(robot_id, 5, p.POSITION_CONTROL, targetPosition=SNAKE_J5, force=HOLD_FORCE)
    p.setJointMotorControl2(robot_id, 6, p.POSITION_CONTROL, targetPosition=LOCK_J6, force=HOLD_FORCE)


def lock_base_and_locked(robot_id, base_ang):
    p.setJointMotorControl2(robot_id, 0, p.POSITION_CONTROL, targetPosition=base_ang, force=BASE_FORCE)
    p.setJointMotorControl2(robot_id, 2, p.POSITION_CONTROL, targetPosition=LOCK_J2, force=HOLD_FORCE)
    p.setJointMotorControl2(robot_id, 4, p.POSITION_CONTROL, targetPosition=LOCK_J4, force=HOLD_FORCE)
    p.setJointMotorControl2(robot_id, 6, p.POSITION_CONTROL, targetPosition=LOCK_J6, force=HOLD_FORCE)


def search_center_x(robot_id, ball_id, projection_matrix):
    """Returns (found: bool, fixed_base_angle: float)."""
    base_angle = math.radians(SCAN_MIN_DEG)
    stable = 0

    for _ in range(MAX_SEARCH_ITERS):
        hold_snake_pose(robot_id, base_angle)

        for _s in range(SETTLE_STEPS_BEFORE_CAPTURE):
            p.stepSimulation()
            time.sleep(1 / 240)

        seg = capture_seg(robot_id, projection_matrix)
        mask = (seg == ball_id)

        if np.any(mask):
            ys, xs = np.where(mask)
            cx = int(np.mean(xs))
            error = cx - CX0

            if abs(error) <= PIX_TOL:
                stable += 1
            else:
                stable = 0

            if stable >= STABLE_FRAMES_REQUIRED:
                return True, float(base_angle)

            step = -KP * error
            if abs(step) < MIN_STEP:
                step = MIN_STEP * (1 if step >= 0 else -1)
            step = clamp(step, -MAX_STEP, MAX_STEP)
            base_angle += step

        else:
            base_angle += SCAN_STEP
            if base_angle > math.radians(SCAN_MAX_DEG):
                base_angle = math.radians(SCAN_MIN_DEG)

        base_angle = clamp(base_angle, math.radians(SCAN_MIN_DEG), math.radians(SCAN_MAX_DEG))

    return False, float(base_angle)


def make_obs(robot_id, ball_id, projection_matrix):
    """
    MUST match training input:
    [ball_y_norm, area_norm, dist_norm, j1, j3, j5]
    """
    visible, y_norm, area_norm, _x_norm = vision_features(robot_id, ball_id, projection_matrix)
    d = dist_to_ball(robot_id, ball_id)
    dist_norm = clamp(d / 1.0, 0.0, 1.0)

    j1 = p.getJointState(robot_id, 1)[0]
    j3 = p.getJointState(robot_id, 3)[0]
    j5 = p.getJointState(robot_id, 5)[0]

    return np.array([y_norm, area_norm, dist_norm, j1, j3, j5], dtype=np.float32)


def run_episode(model, projection_matrix):
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")

    robot_id = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[0, 0, 0])
    for j in range(p.getNumJoints(robot_id)):
        p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL, force=0)

    # random ball X,Y
    bx = float(np.random.uniform(BALL_X_MIN, BALL_X_MAX))
    by = float(np.random.uniform(BALL_Y_MIN, BALL_Y_MAX))

    ball_id = p.loadURDF(
        "sphere2.urdf",
        basePosition=[bx, by, BALL_Z],
        globalScaling=BALL_SCALE,
        useFixedBase=True
    )

    # SEARCH PHASE
    found, fixed_base = search_center_x(robot_id, ball_id, projection_matrix)

    # LOCK BASE
    lock_base_and_locked(robot_id, fixed_base)
    for _ in range(20):
        p.stepSimulation()
        time.sleep(1 / 240)

    # REACH PHASE (PPO)
    obs = make_obs(robot_id, ball_id, projection_matrix)

    for step in range(1, MAX_EP_STEPS + 1):
        action, _ = model.predict(obs, deterministic=True)

        # keep base locked
        lock_base_and_locked(robot_id, fixed_base)

        # apply action to joints 1,3,5
        for i, j in enumerate(ACTIVE_JOINTS):
            cur = p.getJointState(robot_id, j)[0]
            tgt = cur + float(action[i])
            p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=tgt, force=HOLD_FORCE)

        for _ in range(SIM_STEPS_PER_ACTION):
            p.stepSimulation()
            time.sleep(1 / 240)

        d = dist_to_ball(robot_id, ball_id)
        contact = eef_contact_only(robot_id, ball_id)

        if d <= SUCCESS_DIST or contact:
            reason = "DIST" if d <= SUCCESS_DIST else "EEF_CONTACT"
            return True, reason, d, step, found

        obs = make_obs(robot_id, ball_id, projection_matrix)

    d = dist_to_ball(robot_id, ball_id)
    return False, "TIMEOUT", d, MAX_EP_STEPS, found


def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.resetDebugVisualizerCamera(
        cameraDistance=1.6,
        cameraYaw=90,
        cameraPitch=-35,
        cameraTargetPosition=[0.45, 0.0, 0.2]
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=FOV,
        aspect=WIDTH / HEIGHT,
        nearVal=NEAR,
        farVal=FAR
    )

    print(f"Loading model: {MODEL_PATH}.zip")
    model = PPO.load(str(MODEL_PATH))  # NOTE: no .zip

    EPISODES = 20
    success = 0
    fail = 0
    dist_succ = 0
    contact_succ = 0
    search_fail = 0

    for ep in range(1, EPISODES + 1):
        print("\n===============================")
        print(f"EPISODE {ep}/{EPISODES}")
        print("===============================")

        ok, reason, final_dist, steps, search_found = run_episode(model, projection_matrix)

        if not search_found:
            search_fail += 1

        if ok:
            success += 1
            if reason == "DIST":
                dist_succ += 1
            else:
                contact_succ += 1
            print(f"✅ SUCCESS ({reason}) | steps={steps} | dist={final_dist:.4f} m | search_found={search_found}")
        else:
            fail += 1
            print(f"❌ FAIL ({reason}) | steps={steps} | dist={final_dist:.4f} m | search_found={search_found}")

        acc = 100.0 * success / (success + fail)
        print(f"Running accuracy: {success}/{success+fail} = {acc:.1f}%")

    print("\n===================================")
    print("FINAL RESULT")
    print("===================================")
    print(f"Success = {success}")
    print(f"Fail    = {fail}")
    print(f"Accuracy = {100.0*success/(success+fail):.2f}%")
    print(f"Success by distance = {dist_succ}")
    print(f"Success by EEF contact = {contact_succ}")
    print(f"Search failed episodes = {search_fail}")
    print("===================================")

    p.disconnect()


if __name__ == "__main__":
    main()