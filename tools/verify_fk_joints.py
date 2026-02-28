# -*- coding: utf-8 -*-
import json
import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

# src/ 디렉토리를 path에 추가하여 import 가능하게 함
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.anatomy_overlay_viewer import landmarks_to_opensim_coords

class Landmark:
    def __init__(self, d):
        self.x = d.get('x', 0.0)
        self.y = d.get('y', 0.0)
        self.z = d.get('z', 0.0)
        self.visibility = d.get('visibility', 0.0)

# MediaPipe Pose Landmarker names in order (0-32)
LANDMARK_NAMES = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER",
    "RIGHT_EYE", "RIGHT_EYE_OUTER", "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT",
    "MOUTH_RIGHT", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX",
    "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL",
    "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]

def wrap_landmarks(lm_dict):
    wrapped = []
    for name in LANDMARK_NAMES:
        if name in lm_dict:
            wrapped.append(Landmark(lm_dict[name]))
        else:
            wrapped.append(Landmark({'x': 0.0, 'y': 0.0, 'z': 0.0}))
    return wrapped

def main():
    # 1. Load data
    json_path = "data/e2e_output/reference_poses.json"
    if not os.path.exists(json_path):
        json_path = "data/e2e_output/compare_poses.json"
        
    if not os.path.exists(json_path):
        print(f"Error: Data file not found at {json_path}")
        return

    print(f"Loading data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    frames = data.get('frames', [])
    if not frames:
        print("No frames found in the JSON file.")
        return

    # 2. Process first frame
    print("Processing first frame...")
    first_frame = frames[0]
    lm = wrap_landmarks(first_frame['landmarks'])
    world_lm = wrap_landmarks(first_frame['world_landmarks']) if 'world_landmarks' in first_frame else None

    # Reset global EMA state by clearing it
    import src.anatomy_overlay_viewer as aov
    aov._prev_coords = {}
    
    coords_dict = landmarks_to_opensim_coords(lm, world_lm)

    # 3. Save to fk_coords_sample.txt
    out_dir = "reports/2026-02-28_pipeline_validation/assets"
    os.makedirs(out_dir, exist_ok=True)
    out_txt = os.path.join(out_dir, "fk_coords_sample.txt")

    # Important joints to highlight
    target_joints = ['hip_flexion_r', 'knee_angle_r', 'hip_adduction_r']

    with open(out_txt, 'w') as f:
        f.write(f"Sample Frame: {first_frame.get('frame_idx', 0)}\n")
        f.write(f"Timestamp: {first_frame.get('timestamp_sec', 0.0):.3f}s\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Joint Name':<20} | {'Radians':>10} | {'Degrees':>10}\n")
        f.write("-" * 50 + "\n")
        
        # Print target joints first
        for name in target_joints:
            if name in coords_dict:
                val = coords_dict[name]
                deg = math.degrees(val)
                f.write(f"{name:<20} | {val:>10.4f} | {deg:>10.2f} *\n")
        
        f.write("-" * 50 + "\n")
        # Print others
        for name, val in sorted(coords_dict.items()):
            if name not in target_joints:
                deg = math.degrees(val)
                f.write(f"{name:<20} | {val:>10.4f} | {deg:>10.2f}\n")

    print(f"Saved coords to {out_txt}")

    # 4. hip_adduction_r distribution across multiple frames
    print("Calculating hip_adduction_r distribution...")
    hip_add_vals = []
    
    aov._prev_coords = {}
    
    for frame in frames:
        lm = wrap_landmarks(frame['landmarks'])
        wlm = wrap_landmarks(frame['world_landmarks']) if 'world_landmarks' in frame else None
        c = landmarks_to_opensim_coords(lm, wlm)
        if 'hip_adduction_r' in c:
            hip_add_vals.append(math.degrees(c['hip_adduction_r']))

    if hip_add_vals:
        plt.figure(figsize=(10, 6))
        plt.hist(hip_add_vals, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(np.mean(hip_add_vals), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(hip_add_vals):.2f}')
        plt.title("Distribution of hip_adduction_r (Degrees)")
        plt.xlabel("Degrees")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        out_hist = os.path.join(out_dir, "fk_hip_adduction_hist.png")
        plt.savefig(out_hist)
        print(f"Saved histogram to {out_hist}")
    else:
        print("No hip_adduction_r values found to plot.")

if __name__ == "__main__":
    main()
