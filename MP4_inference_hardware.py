import time
import numpy as np
import mujoco
import os
import shutil
from utils_solution import * 
from scipy.spatial.transform import Rotation as R
from piper_sdk import *
import torch
from Train_policy import BowlPushPolicy

###############################

# --- Hardware Initialization ---
piper = C_PiperInterface_V2()
piper.ConnectPort()
while( not piper.EnablePiper()):
    time.sleep(0.01)
piper.MotionCtrl_2(0x01, 0x00, 7, 0x00)
piper.GripperCtrl(0,1000,0x01, 0)
factor = 57295.7795 #1000*180/3.1415926

# Real camera setups
import cv2 as cv
import pyrealsense2 as rs
import threading
import queue

def read_from_camera(cap, frame_queue, running):
    """ Meant to be run from a thread.  Loads frames into a global queue.

    Args:
        cap: OpenCV capture object (e.g., webcam)
        frame_queue: queue in which frames are put
        running: list containing a Boolean that determines if this function continues running
    """

    while running[0]:
        result = cap.read()
        frame_queue.put(result)

    print('read_from_camera thread stopped')


def main():
    # Create a context object. This object owns the handles to all connected realsense devices
    realsense = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    realsense.start(config)

    logitech = cv.VideoCapture(6)
    if not logitech.isOpened():
        print("Could not open Logitech camera.  Exiting")
        return
    
    # Set width and height of frames in pixels
    logitech.set(3, 640)
    logitech.set(4, 480)

    # HAVE to change codec for frame rate
    codec = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
    logitech.set(cv.CAP_PROP_FOURCC, codec)

    # Apparently, we NEED to set FPS here...
    logitech.set(cv.CAP_PROP_FPS, 30)

    # Start camera capture thread in background
    read_from_camera_running = [True]
    frame_queue = queue.Queue()
    read_from_camera_thread = threading.Thread(target=read_from_camera, args=(logitech, frame_queue, read_from_camera_running))
    read_from_camera_thread.start()

    # Move the arm to the zero position
    joint_positions = [0,0,0,0,0,0,0]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = BowlPushPolicy().to(device)
    checkpoint_path = os.path.join(os.path.dirname(__file__), "push_real_checkpoint_real.pt") # change file based on real or sim

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
    
    piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
    while(True):
        ret, frame = frame_queue.get(timeout=5)
        
        # ret, frame are now set and the queue is empty after this block
        while True:
            try:
                ret, frame = frame_queue.get_nowait()
            except queue.Empty:
                break

        if(not ret):
            print('Could not get frame')
            # Skip this iteration if there is not a valid frame
            continue

        frames = realsense.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue
        
        rs_rgb = np.asanyarray(color_frame.get_data())
        rs_depth = np.asanyarray(depth_frame.get_data())
        logi_rgb = frame

        
        ### --- Your Main Code Start --- ###
        logi_rgb = cv.cvtColor(logi_rgb, cv.COLOR_BGR2RGB)

        H, W = 128, 128
        top_rgb_resized  = cv.resize(rs_rgb, (W, H))
        side_rgb_resized = cv.resize(logi_rgb, (W, H))
        depth_resized    = cv.resize(rs_depth, (W, H))

        top_rgb_t  = torch.from_numpy(top_rgb_resized).permute(2, 0, 1).float() / 255.0
        side_rgb_t = torch.from_numpy(side_rgb_resized).permute(2, 0, 1).float() / 255.0
        depth_t    = torch.from_numpy(depth_resized).unsqueeze(0).float() / 1000.0

        img_input = torch.cat([top_rgb_t, depth_t, side_rgb_t], dim=0).unsqueeze(0).to(device)

        cur_robot_state = np.array(joint_positions, dtype=np.float32)
        state_input = torch.from_numpy(cur_robot_state).unsqueeze(0).to(device)

        with torch.no_grad():
            actions_t = policy(img_input, state_input)
            joint_positions = actions_t.squeeze(0).cpu().numpy()

        ### --- Your Main Code End --- ###


        ### --- Your Main Code End --- ###

        # Move joints
        joint_0 = round(joint_positions[0]*factor)
        joint_1 = round(joint_positions[1]*factor)
        joint_2 = round(joint_positions[2]*factor)
        joint_3 = round(joint_positions[3]*factor)
        joint_4 = round(joint_positions[4]*factor)
        joint_5 = round(joint_positions[5]*factor)
        joint_6 = round(joint_positions[6]*1000*1000)
        piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        piper.GripperCtrl(joint_6, 1000, 0x01, 0)
        time.sleep(0.005)

        # Update video streams
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(rs_depth, alpha=0.03), cv2.COLORMAP_JET)
        cv.imshow('rs_rgb', rs_rgb)
        cv.imshow('rs_depth', depth_colormap)
        cv.imshow('logi_rgb', logi_rgb)

        key = cv.waitKey(1)

        # If 'q' is pressed, quit the main loop
        if(key == ord('q')):
            break

    # Terminate the thread that's running in the background
    read_from_camera_running[0] = False
    read_from_camera_thread.join()

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
