# -*- coding: utf-8 -*-
"""
E2E Reprojection Overlay Diagnostic Tool.
Visualizes MediaPipe 2D landmarks and SMPL joints projected back to the image.
Using MediaPipe Tasks API.
"""

import os
import numpy as np

# 1. Monkey-patch BEFORE other imports
np.bool = np.bool_
np.int = np.int_
np.float = np.float64
np.complex = np.complex128
np.object = np.object_
np.unicode = np.str_
np.str = np.str_

import cv2
import torch
import mediapipe as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Project imports
from src.smplx_engine import SMPLXEngine

# Configuration
VIDEO_PATH = "my_data/2026_02_25/IMG_2633.MOV"
MODEL_DIR = "data/models/smpl"
POSE_MODEL_PATH = "data/models/pose_landmarker_lite.task"
OUTPUT_DIR = "reports/2026-03-01_e2e_validation/assets"
FRAMES_TO_EXTRACT = [100, 300, 600]
TARGET_SIZE = (1280, 720)

# MediaPipe landmark names (standard order)
MP_POSE_NAMES = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER",
    "RIGHT_EYE", "RIGHT_EYE_OUTER", "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT",
    "MOUTH_RIGHT", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX",
    "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL",
    "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]

# Mapping for PnP (8 points)
# MediaPipe index
PNP_MP_INDICES = [11, 12, 23, 24, 25, 26, 27, 28]

# MediaPipe to SMPL mapping for error calculation (MP index: SMPL index)
MP_SMPL_CORRESPONDENCE = {
    11: 16, 12: 17, 13: 18, 14: 19, 15: 20, 16: 21,
    23: 1, 24: 2, 25: 4, 26: 5, 27: 7, 28: 8
}

def estimate_camera(world_pts: np.ndarray, image_pts: np.ndarray, img_w: int, img_h: int):
    """Estimate camera pose using solvePnP."""
    # Initial camera matrix
    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist_coeffs = np.zeros((4, 1)) # Assuming no distortion
    
    success, rvec, tvec = cv2.solvePnP(
        world_pts.astype(np.float32), 
        image_pts.astype(np.float32), 
        camera_matrix, 
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    return success, rvec, tvec, camera_matrix, dist_coeffs

def process_frame(frame_idx: int, frame: np.ndarray, landmarker, engine: SMPLXEngine):
    """Processes a single frame and returns visual results."""
    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    # 2. Run MediaPipe Pose
    results = landmarker.detect(mp_image)
    if not results.pose_landmarks:
        print(f"Frame {frame_idx}: No landmarks detected.")
        return None

    # Extract image landmarks (pixels)
    image_lms_px = []
    # first person
    for lm in results.pose_landmarks[0]:
        image_lms_px.append([lm.x * w, lm.y * h])
    image_lms_px = np.array(image_lms_px)

    # Extract world landmarks (meters, Y-DOWN)
    world_lms_dict = {}
    world_lms_arr = []
    for i, lm in enumerate(results.pose_world_landmarks[0]):
        name = MP_POSE_NAMES[i]
        world_lms_dict[name] = {"x": lm.x, "y": lm.y, "z": lm.z}
        world_lms_arr.append([lm.x, lm.y, lm.z])
    world_lms_arr = np.array(world_lms_arr)

    # 3. Fit SMPL
    print(f"Fitting SMPL for frame {frame_idx}...")
    vertices, joints, meta = engine.fit_frame(world_lms_dict, num_iters=50)
    
    # 4. Camera Estimation using PnP (8 points)
    pnp_world = world_lms_arr[PNP_MP_INDICES]
    pnp_image = image_lms_px[PNP_MP_INDICES]
    
    success, rvec, tvec, K, dist = estimate_camera(pnp_world, pnp_image, w, h)
    
    if not success:
        print(f"Frame {frame_idx}: PnP failed.")
        return None

    # Project SMPL joints back to image
    smpl_joints_mp_world = joints.copy()
    smpl_joints_mp_world[:, 1] *= -1 # Y-flip: Y-UP to Y-DOWN
    
    projected_joints, _ = cv2.projectPoints(smpl_joints_mp_world.astype(np.float32), rvec, tvec, K, dist)
    projected_joints = projected_joints.reshape(-1, 2)
    
    # Resize frame for visualization
    frame_resized = cv2.resize(frame, TARGET_SIZE)
    scale_x = TARGET_SIZE[0] / w
    scale_y = TARGET_SIZE[1] / h
    
    # 5. Visualization
    total_error = 0
    count = 0
    
    vis_frame = frame_resized.copy()
    
    # Draw MediaPipe 2D landmarks (Green)
    for i, (px, py) in enumerate(image_lms_px):
        cv2.circle(vis_frame, (int(px * scale_x), int(py * scale_y)), 6, (0, 255, 0), -1)
        
    # Draw SMPL joints (Blue) and Error lines (Yellow)
    for mp_idx, sm_idx in MP_SMPL_CORRESPONDENCE.items():
        # MP pixel
        mp_px, mp_py = image_lms_px[mp_idx]
        # Projected SMPL pixel
        sm_px, sm_py = projected_joints[sm_idx]
        
        # Calculate error in original pixels
        err = np.sqrt((mp_px - sm_px)**2 + (mp_py - sm_py)**2)
        total_error += err
        count += 1
        
        # Draw on resized frame
        start_pt = (int(mp_px * scale_x), int(mp_py * scale_y))
        end_pt = (int(sm_px * scale_x), int(sm_py * scale_y))
        
        cv2.line(vis_frame, start_pt, end_pt, (0, 255, 255), 1) # Yellow line
        cv2.circle(vis_frame, end_pt, 4, (255, 0, 0), -1) # Blue circle
        
    avg_error = total_error / count if count > 0 else 0
    cv2.putText(vis_frame, f"Avg Reproj Error: {avg_error:.2f} px", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis_frame, f"Frame: {frame_idx}", (30, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return vis_frame, avg_error

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize SMPL Engine
    engine = SMPLXEngine(MODEL_DIR)
    
    # Initialize MediaPipe Pose Task
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=RunningMode.IMAGE
    )
    landmarker = PoseLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return

    summary_lines = []
    
    for f_idx in FRAMES_TO_EXTRACT:
        print(f"Processing frame {f_idx}...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame {f_idx}")
            continue
            
        result = process_frame(f_idx, frame, landmarker, engine)
        if result:
            vis_img, err = result
            out_path = os.path.join(OUTPUT_DIR, f"reprojection_frame{f_idx}.jpg")
            cv2.imwrite(out_path, vis_img)
            print(f"Saved: {out_path} (Error: {err:.2f} px)")
            summary_lines.append(f"Frame {f_idx}: {err:.2f} px")
            
    cap.release()
    landmarker.close()
    
    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "reprojection_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
