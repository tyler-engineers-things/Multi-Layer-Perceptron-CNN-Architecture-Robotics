import time
import numpy as np
import mujoco
import os
import matplotlib.pyplot as plt
import shutil
import cv2
import trimesh
import xml.etree.ElementTree as ET
import matplotlib.animation as animation
from datetime import datetime

def render_rgb_depth(renderer, data, camera: str = "top_cam"):
    """Return (rgb uint8 HxWx3, depth float32 HxW in meters) from a named camera."""
    # update the scene from current data and camera
    renderer.update_scene(data, camera=camera)
    
    # RGB 
    renderer.disable_depth_rendering()
    renderer.disable_segmentation_rendering()
    rgb = renderer.render().copy()          

    renderer.enable_segmentation_rendering()

    # Depth
    renderer.enable_depth_rendering()
    renderer.disable_segmentation_rendering()
    depth = renderer.render().copy()        

    return rgb.copy(), depth.copy()

def save_png(path, img_uint8):
    """
    Save an HxWx3 uint8 image to disk using matplotlib.
    """
    plt.imsave(path, img_uint8)

def overlay_mask_on_image(rgb, mask, flow_points, mask_color=(255, 0, 0), alpha=0.5,
                          point_color=(0, 0, 255), point_radius=2):
    """
    Overlay a binary mask and flow points on an RGB image using OpenCV.

    Args:
        rgb (np.ndarray): HxWx3 RGB image (uint8)
        mask (np.ndarray): HxW binary mask (bool or uint8)
        flow_points (np.ndarray): Nx2 array of (x, y) pixel coordinates
        mask_color (tuple): RGB color for the mask (default: red)
        alpha (float): Mask transparency
        point_color (tuple): RGB color for points (default: blue)
        point_radius (int): Radius of the points
    Returns:
        np.ndarray: Image with mask and points overlaid
    """
    overlay = rgb.copy()
    colored_mask = np.zeros_like(rgb, dtype=np.uint8)
    colored_mask[mask > 0] = mask_color

    # Blend the mask
    overlay = np.where(mask[..., None].astype(bool),
                       (1 - alpha) * overlay + alpha * colored_mask,
                       overlay).astype(np.uint8)

    # Draw points on top
    if flow_points is not None and len(flow_points) > 0:
        for pt in flow_points.astype(int):
            x, y = pt
            cv2.circle(overlay, (x, y), radius=point_radius, color=point_color, thickness=-1)

    return overlay

def axis_angle_to_quat_wxyz(axis, angle):
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / n
    s = np.sin(angle / 2.0)
    qw = np.cos(angle / 2.0)
    qx, qy, qz = axis * s
    return np.array([qw, qx, qy, qz])

def rot_mat_to_quat_wxyz(R, eps=1e-8):
    R = np.asarray(R, dtype=float)
    assert R.shape == (3,3)
    tr = np.trace(R)

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0  # 4*qw
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    else:
        # choose the largest diagonal to keep S large
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(max(eps, 1.0 + R[0,0] - R[1,1] - R[2,2])) * 2.0
            qw = (R[2,1] - R[1,2]) / S
            qx = 0.25 * S
            qy = (R[0,1] + R[1,0]) / S
            qz = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(max(eps, 1.0 + R[1,1] - R[0,0] - R[2,2])) * 2.0
            qw = (R[0,2] - R[2,0]) / S
            qx = (R[0,1] + R[1,0]) / S
            qy = 0.25 * S
            qz = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(max(eps, 1.0 + R[2,2] - R[0,0] - R[1,1])) * 2.0
            qw = (R[1,0] - R[0,1]) / S
            qx = (R[0,2] + R[2,0]) / S
            qy = (R[1,2] + R[2,1]) / S
            qz = 0.25 * S

    q = np.array([qw, qx, qy, qz])
    q /= np.linalg.norm(q)
    return q

def normalize_quat_wxyz(q):
    q = np.asarray(q, dtype=float)
    return q / (np.linalg.norm(q) + 1e-12)

def visualize_mano_3d_traj_video(data, output_path="", fps=5):
    """
    Visualize a sequence of 3d point clouds.
    """
    all_coords = data

    T = all_coords.shape[0]

    # Get axis limits
    x_all, y_all, z_all = all_coords[..., 0].ravel(), all_coords[..., 1].ravel(), all_coords[..., 2].ravel()
    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()
    z_min, z_max = z_all.min(), z_all.max()

    expand_ratio = 1.2
    x_half, y_half, z_half = (x_max - x_min) * 0.5 * expand_ratio, (y_max - y_min) * 0.5 * expand_ratio, (z_max - z_min) * 0.5 * expand_ratio
    x_mid, y_mid, z_mid = (x_max + x_min) * 0.5, (y_max + y_min) * 0.5, (z_max + z_min) * 0.5

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(x_mid - x_half, x_mid + x_half)
    ax.set_ylim(y_mid - y_half, y_mid + y_half)
    ax.set_zlim(z_mid - z_half, z_mid + z_half)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f"3D point cloud trajectories.")

    # Scatter points (joints) in blue
    scat = ax.scatter([], [], [], s=15, color='blue')

    def update(frame):
        coords_21 = all_coords[frame]  # shape (21,3)
        xs, ys, zs = coords_21[:, 0], coords_21[:, 1], coords_21[:, 2]
        scat._offsets3d = (xs, ys, zs)

        return [scat]

    ani = animation.FuncAnimation(fig, update, frames=T, blit=False, interval=200)

    if output_path:
        ani.save(output_path, fps=fps, writer='ffmpeg')
        print(f"Animation saved to: {output_path}")
    else:
        plt.show()

def _quat_wxyz_to_R(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),       1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)]
    ], dtype=float)

def _transform_points(pts_local, R, t):
    return pts_local @ R.T + t

def draw_pointclouds_and_pose_in_viewer(
    viewer,
    points,                 # (N,3) array OR list of (Ni,3)
    pos_w,                  # (3,) world position of the object
    quat_wxyz,              # (4,) world quaternion of the object [qw,qx,qy,qz]
    points_in_world=True,   # set False if 'points' are in the object local frame
    clear=True,             # clear previous overlays in user_scn
    point_radius=0.005,
    axis_len=0.05
):
    """Fill viewer.user_scn with point spheres and pose axes. Does NOT call viewer.sync()."""
    if isinstance(points, np.ndarray):
        points = [points]

    if clear:
        viewer.user_scn.ngeom = 0

    pos_w = np.asarray(pos_w, dtype=float)
    quat_wxyz = np.asarray(quat_wxyz, dtype=float)
    R_w_from_obj = _quat_wxyz_to_R(quat_wxyz)

    # ---- draw points as spheres (fixed types/shapes) ----
    rgba_point = np.array([0.95, 0.2, 0.2, 1.0], dtype=np.float32)  # float32
    size_vec  = np.array([point_radius, 0.0, 0.0], dtype=np.float64) # float64 (3,)
    mat_I9    = np.eye(3, dtype=np.float64).reshape(-1)              # float64 (9,)

    for pc in points:  # ensure you made a list beforehand
        if pc is None or len(pc) == 0:
            continue
        pc = np.asarray(pc, dtype=np.float64)
        pc_w = pc if points_in_world else (pc @ R_w_from_obj.T + pos_w)

        for pt in pc_w:
            if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                break

            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size_vec,                         # (3,) float64
                pt.astype(np.float64),            # (3,) float64
                mat_I9,                           # (9,) float64 rotation matrix
                rgba_point                        # (4,) float32
            )
            viewer.user_scn.ngeom += 1

    # ---- draw pose axes (X/Y/Z) as capsules/lines ----
    colors = [
        np.array([1, 0, 0, 1], dtype=np.float32),  # X
        np.array([0, 1, 0, 1], dtype=np.float32),  # Y
        np.array([0, 0, 1, 1], dtype=np.float32),  # Z
    ]
    axes_world = R_w_from_obj @ np.eye(3, dtype=np.float64)
    o = pos_w.astype(np.float64)
    width = 0.003

    for i in range(3):
        a = o
        b = (o + axis_len * axes_world[:, i]).astype(np.float64)

        # MuJoCo 3.x signature: (geom, type, width, from, to)
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
        g.rgba = colors[i]  # set color separately on the geom
        viewer.user_scn.ngeom += 1