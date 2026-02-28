# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
import math
import mediapipe as mp
import matplotlib.pyplot as plt

# Use Agg backend for headless environment
import matplotlib
matplotlib.use('Agg')

# Import OpenSim
try:
    import opensim as osim
except ImportError:
    # If standard opensim is not found, try pyopensim as per context
    import pyopensim as osim

# Add src to path for imports
sys.path.append('/mnt/d/progress/芭蕾呪法')
from src.anatomy_overlay_viewer import (
    landmarks_to_opensim_coords,
    compute_fk,
    OPENSIM_MODEL_PATH
)

# Constants
VIDEO_PATH = '/mnt/d/progress/芭蕾呪法/my_data/2026_02_25/IMG_2633.MOV'
MODEL_PATH = '/mnt/d/progress/芭蕾呪法/data/models/pose_landmarker_lite.task'
VIDEO_FRAME_INDICES = [100, 300, 600]
ASSETS_DIR = '/mnt/d/progress/芭蕾呪法/reports/2026-02-28_vtp_mp_diagnosis/assets/'
os.makedirs(ASSETS_DIR, exist_ok=True)

# Comparison configuration: (mp_start_idx, mp_end_idx, fk_parent, fk_child, label)
BONE_CHAINS = [
    (24, 26, 'femur_r', 'tibia_r', 'R-Thigh'),
    (23, 25, 'femur_l', 'tibia_l', 'L-Thigh'),
    (26, 28, 'tibia_r', 'talus_r', 'R-Shank'),
    (25, 27, 'tibia_l', 'talus_l', 'L-Shank'),
    (12, 14, 'humerus_r', 'radius_r', 'R-Arm'),
]

ORDERING_CHAIN = ['pelvis', 'femur_r', 'tibia_r', 'talus_r', 'calcn_r']

def get_mp_vector(world_lm, start_idx, end_idx):
    p_start = world_lm[start_idx]
    p_end = world_lm[end_idx]
    v = np.array([
        p_end.x - p_start.x,
        -(p_end.y - p_start.y), # Invert Y
        p_end.z - p_start.z
    ])
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return v
    return v / norm

def get_fk_vector(transforms, parent, child):
    if parent not in transforms or child not in transforms:
        return None
    p_pos = transforms[parent][:3, 3]
    c_pos = transforms[child][:3, 3]
    v = c_pos - p_pos
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return v
    return v / norm

def calculate_angle_error(v1, v2):
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return math.degrees(math.acos(dot))

def run_diagnosis():
    # 1. Setup MediaPipe
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
    )
    landmarker = PoseLandmarker.create_from_options(options)

    # 2. Load OpenSim Model
    model = osim.Model(OPENSIM_MODEL_PATH)
    state = model.initSystem()

    # 3. Process Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f'Error: Could not open video {VIDEO_PATH}')
        return

    results = []

    for frame_idx in VIDEO_FRAME_INDICES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f'Warning: Could not read frame {frame_idx}. Skipping.')
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        mp_result = landmarker.detect(mp_image)

        if not mp_result.pose_landmarks or not mp_result.pose_world_landmarks:
            print(f'Warning: No pose detected in frame {frame_idx}. Skipping.')
            continue

        landmarks = mp_result.pose_landmarks[0]
        world_landmarks = mp_result.pose_world_landmarks[0]

        coords = landmarks_to_opensim_coords(landmarks, world_landmarks)
        fk_transforms = compute_fk(model, state, coords)

        errors = {}
        for s_idx, e_idx, p_name, c_name, label in BONE_CHAINS:
            mp_vec = get_mp_vector(world_landmarks, s_idx, e_idx)
            fk_vec = get_fk_vector(fk_transforms, p_name, c_name)
            if fk_vec is not None:
                err = calculate_angle_error(mp_vec, fk_vec)
                errors[label] = err
            else:
                errors[label] = None

        ordering_y = []
        for body_name in ORDERING_CHAIN:
            if body_name in fk_transforms:
                ordering_y.append(fk_transforms[body_name][1, 3])
            else:
                ordering_y.append(None)
        
        ordering_violated = False
        for i in range(len(ordering_y) - 1):
            if ordering_y[i] is not None and ordering_y[i+1] is not None:
                if ordering_y[i] <= ordering_y[i+1]:
                    ordering_violated = True
                    break

        results.append({
            'frame_idx': frame_idx,
            'coords': coords,
            'errors': errors,
            'ordering_y': ordering_y,
            'ordering_violated': ordering_violated
        })

    cap.release()
    landmarker.close()

    generate_plots(results, model)
    generate_summary(results)

def generate_plots(results, model):
    plt.figure(figsize=(10, 6))
    x = np.arange(len(BONE_CHAINS))
    width = 0.2
    
    for i, res in enumerate(results):
        err_vals = [res['errors'][label] if res['errors'][label] is not None else 0 for _, _, _, _, label in BONE_CHAINS]
        plt.bar(x + (i - 1) * width, err_vals, width, label=f'Frame {res["frame_idx"]}')

    plt.axhline(y=15, color='r', linestyle='--', label='15 deg threshold')
    plt.xticks(x, [label for _, _, _, _, label in BONE_CHAINS])
    plt.ylabel('Angle Error (degrees)')
    plt.title('Angular Error Comparison between MP and OpenSim FK')
    plt.legend()
    plt.savefig(os.path.join(ASSETS_DIR, 'angular_error_comparison.png'))
    plt.close()

    coord_set = model.getCoordinateSet()
    for res in results:
        frame_idx = res['frame_idx']
        coords = res['coords']
        
        dof_names = list(coords.keys())
        dof_vals = [math.degrees(coords[name]) for name in dof_names]
        
        plt.figure(figsize=(12, 8))
        y_pos = np.arange(len(dof_names))
        
        for i, name in enumerate(dof_names):
            try:
                c = coord_set.get(name)
                min_val = math.degrees(c.getRangeMin())
                max_val = math.degrees(c.getRangeMax())
                plt.barh(i, max_val - min_val, left=min_val, color='gray', alpha=0.2)
            except:
                pass
        
        plt.barh(y_pos, dof_vals, color='skyblue')
        plt.yticks(y_pos, dof_names)
        plt.xlabel('Value (degrees)')
        plt.title(f'DOF Values for Frame {frame_idx}')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(ASSETS_DIR, f'dof_values_frame{frame_idx}.png'))
        plt.close()

    plt.figure(figsize=(10, 6))
    for res in results:
        y_vals = res['ordering_y']
        valid_y = [y for y in y_vals if y is not None]
        valid_chain = [ORDERING_CHAIN[i] for i, y in enumerate(y_vals) if y is not None]
        
        color = 'green' if not res['ordering_violated'] else 'red'
        plt.plot(valid_chain, valid_y, marker='o', label=f'Frame {res["frame_idx"]}', color=color)

    plt.ylabel('Y Coordinate (meters)')
    plt.title('Chain Ordering Check (Pelvis -> Calcn)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(ASSETS_DIR, 'chain_ordering_check.png'))
    plt.close()

def generate_summary(results):
    summary_path = os.path.join(ASSETS_DIR, 'diag_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('=== VTP vs MediaPipe Diagnosis Summary ===\n\n')
        
        for res in results:
            f.write(f'Frame {res["frame_idx"]}:\n')
            f.write(f'  - Ordering Check: {"FAILED" if res["ordering_violated"] else "PASSED"}\n')
            f.write('  - Angular Errors:\n')
            frame_errors = []
            for label, err in res['errors'].items():
                if err is not None:
                    f.write(f'    {label:10}: {err:6.2f} deg\n')
                    frame_errors.append(err)
                else:
                    f.write(f'    {label:10}: N/A\n')
            
            if frame_errors:
                avg_err = sum(frame_errors) / len(frame_errors)
                max_err = max(frame_errors)
                f.write(f'  - Frame Avg Error: {avg_err:.2f} deg\n')
                f.write(f'  - Frame Max Error: {max_err:.2f} deg\n')
            f.write("\n")
        
        all_errors = []
        for res in results:
            for err in res['errors'].values():
                if err is not None:
                    all_errors.append(err)
        
        if all_errors:
            f.write('Overall Metrics:\n')
            f.write(f'  - Total Avg Error: {sum(all_errors)/len(all_errors):.2f} deg\n')
            f.write(f'  - Total Max Error: {max(all_errors):.2f} deg\n')

if __name__ == "__main__":
    run_diagnosis()
