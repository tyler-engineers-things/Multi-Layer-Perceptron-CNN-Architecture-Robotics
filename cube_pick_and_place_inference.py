import time
import numpy as np
import mujoco
import os
import shutil
from utils_solution import * 
from scipy.spatial.transform import Rotation as R

## Import your ML dependencies and model builder
## Enter codes here
import torch
from torchvision.io import decode_image, read_image
from torch.utils.data import Dataset, DataLoader
from Second_Train_Policy import PickPlacePolicy

## Enter codes here

# --- Load model/data ---
# Updated path to use the assets directory structure
model = mujoco.MjModel.from_xml_path("./scene_pick_and_place.xml")  # Load the complete scene with robot and breadboard
data  = mujoco.MjData(model)

renderer = mujoco.Renderer(model, height=480, width=640)

# (Optional) viewer - Updated for MuJoCo 3.3.5
import mujoco.viewer
viewer = mujoco.viewer.launch_passive(model, data)
# ---- offscreen renderer (pick your size) ----

# --- Helpers ---
def actuator_id(name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

def joint_id(name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

# Build a stable name->index mapping for actuators
# Updated for new robot: 6 arm joints + 1 gripper actuator
act_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
act_idx   = np.array([actuator_id(n) for n in act_names], dtype=int)

def set_qpos_by_joint_names(qpos_targets: dict):
    """
    Initialize pose by directly setting joint positions, then forward the model.
    Use joint names (not actuator names) for clarity.
    """
    for name, q in qpos_targets.items():
        jid = joint_id(name)
        dof = model.jnt_qposadr[jid]
        data.qpos[dof] = q
    mujoco.mj_forward(model, data)

# Extract actuator ctrl ranges for clamping (shape: [nu, 2])
ctrl_range = model.actuator_ctrlrange.copy()

# Print actuator information for debugging
print("Available actuators:")
for i in range(model.nu):
    actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"  {i}: {actuator_name}")

print(f"\nControl ranges:")
for name in act_names:
    try:
        idx = actuator_id(name)
        lo, hi = ctrl_range[idx]
        print(f"  {name}: [{lo:.3f}, {hi:.3f}]")
    except:
        print(f"  {name}: NOT FOUND")

def set_targets_by_dict(targets: dict):
    """
    targets: dict like {"joint1": 0.0, "joint2": 1.0, ...}
    Writes into data.ctrl in actuator order, with safe clamping.
    """
    for name, value in targets.items():
        i = actuator_id(name)
        lo, hi = ctrl_range[i]
        data.ctrl[i] = np.clip(value, lo, hi)
        
# --- Example: home, move, and operate gripper ---
# 1) Set a comfortable start pose (radians for revolute joints)
# Updated for new robot: 6 arm joints + gripper joints
home = {
    "joint1": 0.0,     # Base rotation
    "joint2": 0.0,     # Shoulder
    "joint3": 0.0,     # Elbow  
    "joint4": 0.0,     # Wrist roll
    "joint5": 0.0,     # Wrist pitch
    "joint6": 0.0,     # Wrist yaw
    "gripper": 0.0,     # Gripper (0 = closed, 0.035 = open)
}
# Set initial joint positions
set_qpos_by_joint_names(home)

# set the init pose of the bowl
# Get index of the first qpos entry for this free joint
bowl_qpos_id = model.joint(name="bowl_joint")
bowl_qpos_addr = bowl_qpos_id.qposadr[0] # start index in data.qpos

# Set new orientation
euler_new = np.array([0., 0., np.pi/2])  # comment this line to randomize the board angle
quat_new = R.from_euler('xyz', euler_new).as_quat()  # [x, y, z, w] order

desired_pos = [0.5, 0.0, 0.04175]

# Update qpos: [x, y, z, qw, qx, qy, qz]
data.qpos[bowl_qpos_addr:bowl_qpos_addr+3] = desired_pos
data.qpos[bowl_qpos_addr+3:bowl_qpos_addr+7] = [quat_new[3], quat_new[0], quat_new[1], quat_new[2]]

# for cube
cube_qpos_id = model.joint(name="cube_joint")
cube_qpos_addr = cube_qpos_id.qposadr[0] # start index in data.qpos

# Set new orientation
rot_values = np.random.uniform(-0.05, 0.05, size=1)
euler_new = np.array([0., 0., np.pi/2 + rot_values[0]])  # comment this line to randomize the board angle
quat_new = R.from_euler('xyz', euler_new).as_quat()  # [x, y, z, w] order

values = np.random.uniform(-0.05, 0.05, size=2)
position_new = [0.35 + values[0], -0.20 + values[1], 0.030]

# Update qpos: [x, y, z, qw, qx, qy, qz]
data.qpos[cube_qpos_addr:cube_qpos_addr+3] = position_new
data.qpos[cube_qpos_addr+3:cube_qpos_addr+7] = [quat_new[3], quat_new[0], quat_new[1], quat_new[2]]

mujoco.mj_forward(model, data)

# zero-out the simulator
data.ctrl[:] = 0.0
for t in range(200):
    mujoco.mj_step(model, data)
    if viewer is not None:
        viewer.sync()

    time.sleep(0.001)  # Small delay for smooth visualization

## Define your policy model and load the checkpoints
## Enter codes here
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = PickPlacePolicy().to(device)
checkpoint_path = os.path.join(os.path.dirname(__file__), "new_sim_pick_place_policy_checkpoint.pt") # Change based on real or sim

if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        policy.load_state_dict(ckpt["model"])
    else:
        policy.load_state_dict(ckpt)
    print(f"Loaded policy checkpoint from: {checkpoint_path}")
else:
    print(f"WARNING: no checkpoint found at {checkpoint_path}. "
          f"Running with randomly initialized policy weights.")

policy.eval()

## Enter codes here

print("Begin simulation evaluation...")
maximum_simulation_steps = 10000

for t in range(maximum_simulation_steps):
    if viewer is not None:
        viewer.sync()

    rgb_top_cam, depth_img = render_rgb_depth(renderer, data, "top_cam")
    d = depth_img.copy()
    m = np.isfinite(d)
    if m.any():
        # normalize depth to [0, 255]
        d_vis = (255 * (d[m] - d[m].min()) / (np.ptp(d[m]) + 1e-8)).astype(np.uint8)
        depth_top_cam = np.zeros_like(d, dtype=np.uint8)
        depth_top_cam[m] = d_vis
    else:
        depth_top_cam = np.zeros_like(d, dtype=np.uint8)

    # rgb_top_cam, and depth_top_cam are the rbgd images of the top camera.
    rgb_side_cam, _ = render_rgb_depth(renderer, data, "side_cam")
    # rgb_side_cam is the rbgd image of the side camera.
    
    cur_robot_state = data.ctrl.copy()
    # cur_robot_state is the current robot state, which might also be the input to your trained model as long as you are not doing visual servoing.
    
    ## Feed the sensor inputs to the model and return the robot actions.
    ## Enter codes here
    top_rgb_t  = torch.from_numpy(rgb_top_cam).permute(2, 0, 1).float() / 255.0
    depth_t    = torch.from_numpy(depth_top_cam).unsqueeze(0).float() / 255.0
    side_rgb_t = torch.from_numpy(rgb_side_cam).permute(2, 0, 1).float() / 255.0

    img_input = torch.cat([top_rgb_t, depth_t, side_rgb_t], dim=0)
    img_input = img_input.unsqueeze(0).to(device)

    state_input = torch.from_numpy(cur_robot_state.astype(np.float32))
    state_input = state_input.unsqueeze(0).to(device)

    with torch.no_grad():
        joint_abs, gripper_prob = policy(img_input, state_input)

    joint_abs = joint_abs.cpu().numpy().reshape(-1)
    g_open = gripper_prob.item()

    cur_robot_state = data.ctrl.copy()
    cur_joints = cur_robot_state[:6]

    alpha = 1.0
    target_joints = alpha * joint_abs + (1.0 - alpha) * cur_joints

    actions = np.zeros(7, dtype=np.float32)
    actions[:6] = target_joints
    actions[6] = 0.035 if g_open > 0.5 else 0.0



    policy_command = {
        "joint1": actions[0],     # Base rotation
        "joint2": actions[1],     # Shoulder
        "joint3": actions[2],     # Elbow  
        "joint4": actions[3],     # Wrist roll
        "joint5": actions[4],     # Wrist pitch
        "joint6": actions[5],     # Wrist yaw
        "gripper": actions[6],     # Gripper (0 = closed, 0.035 = open)
    }
    ## Enter codes here

    set_targets_by_dict(policy_command)
    mujoco.mj_step(model, data)

    current_cube_posi = data.qpos[cube_qpos_addr:cube_qpos_addr+3]
    print("Current cube pose:", current_cube_posi)

    # the distance is smaller than 0.05, then considered as success -> cube being dropped off in the bowl
    if np.linalg.norm(desired_pos - current_cube_posi) <= 0.05:
        break

print("Finish simulation evaluation...")

# Keep the viewer open
if viewer is not None:
    print("Simulation complete. Close the viewer window to exit.")
    try:
        while viewer.is_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Simulation interrupted.")
    finally:
        viewer.close()



        