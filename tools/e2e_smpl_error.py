# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

# 0. Chumpy/SMPL compatibility monkey-patch (must be before smplx import)
np.bool = bool
np.int = int
np.float = float
np.complex = complex
np.object = object
np.unicode = str
np.str = str

# Set matplotlib backend to Agg for headless environments
plt.switch_backend('Agg')

from src.smplx_engine import SMPLXEngine
from src.smplx_mapper import SMPLXToOpenSimMapper

# MediaPipe landmark names in standard order (0-32)
MP_LANDMARK_NAMES = [
    'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER',
    'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT',
    'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX',
    'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP',
    'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL',
    'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
]

def main():
    # Paths
    ref_poses_path = 'data/e2e_output/reference_poses.json'
    model_dir = 'data/models/smpl'
    output_dir = 'reports/2026-03-01_e2e_validation/assets'
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data
    print(f"Loading reference poses from {ref_poses_path}...")
    with open(ref_poses_path, 'r') as f:
        data = json.load(f)
    
    frames = data['frames']
    print(f"Total frames available: {len(frames)}")

    # 2. Select 5 target frames: closest to [200, 500, 800, 1100, 1400] that have landmarks
    target_indices = [200, 500, 800, 1100, 1400]
    selected_frames = []
    
    # Filter frames with world_landmarks
    valid_frames = [f for f in frames if f.get('world_landmarks')]
    valid_indices = [f['frame_idx'] for f in valid_frames]
    
    for target in target_indices:
        # Find nearest frame_idx among valid frames
        idx = np.argmin(np.abs(np.array(valid_indices) - target))
        selected_frames.append(valid_frames[idx])
        print(f"Selected frame_idx {valid_frames[idx]['frame_idx']} for target {target}")

    # 3. Initialize Engine and Mapper
    engine = SMPLXEngine(model_dir)
    mapper = SMPLXToOpenSimMapper()

    # Mappings for validation
    mp_to_smpl_idx = {
        'LEFT_SHOULDER': 16, 'RIGHT_SHOULDER': 17,
        'LEFT_ELBOW': 18, 'RIGHT_ELBOW': 19,
        'LEFT_WRIST': 20, 'RIGHT_WRIST': 21,
        'LEFT_HIP': 1, 'RIGHT_HIP': 2,
        'LEFT_KNEE': 4, 'RIGHT_KNEE': 5,
        'LEFT_ANKLE': 7, 'RIGHT_ANKLE': 8,
    }

    marker_to_mp_map = {
        'ACROMION_L': 'LEFT_SHOULDER', 'ACROMION_R': 'RIGHT_SHOULDER',
        'ELBOW_LAT_L': 'LEFT_ELBOW', 'ELBOW_MED_L': 'LEFT_ELBOW',
        'ELBOW_LAT_R': 'RIGHT_ELBOW', 'ELBOW_MED_R': 'RIGHT_ELBOW',
        'WRIST_RAD_L': 'LEFT_WRIST', 'WRIST_ULN_L': 'LEFT_WRIST',
        'WRIST_RAD_R': 'RIGHT_WRIST', 'WRIST_ULN_R': 'RIGHT_WRIST',
        'GTROCHANTER_L': 'LEFT_HIP', 'GTROCHANTER_R': 'RIGHT_HIP',
        'KNEE_LAT_L': 'LEFT_KNEE', 'KNEE_MED_L': 'LEFT_KNEE',
        'KNEE_LAT_R': 'RIGHT_KNEE', 'KNEE_MED_R': 'RIGHT_KNEE',
        'ANKLE_LAT_L': 'LEFT_ANKLE', 'ANKLE_MED_L': 'LEFT_ANKLE',
        'ANKLE_LAT_R': 'RIGHT_ANKLE', 'ANKLE_MED_R': 'RIGHT_ANKLE',
    }

    joint_errors = {name: [] for name in mp_to_smpl_idx.keys()}
    marker_errors = {name: [] for name in marker_to_mp_map.keys()}
    
    frame_results = []

    # 4. Processing Loop
    for frame_data in selected_frames:
        frame_idx = frame_data['frame_idx']
        mp_world = frame_data['world_landmarks']
        
        # Ensure correct order for engine.fit_frame
        ordered_mp_world = {name: mp_world[name] for name in MP_LANDMARK_NAMES if name in mp_world}
        
        print(f"Fitting frame {frame_idx}...")
        vertices, joints, meta = engine.fit_frame(ordered_mp_world, num_iters=200)
        
        virtual_markers = mapper.extract_virtual_markers(vertices)
        
        current_frame_errors = {}
        
        # Calculate Joint Errors (MP vs SMPL Joints)
        for mp_name, sm_idx in mp_to_smpl_idx.items():
            p_mp = mp_world[mp_name]
            target_pos = np.array([p_mp['x'], -p_mp['y'], p_mp['z']])
            smpl_pos = joints[sm_idx]
            
            err = np.linalg.norm(target_pos - smpl_pos) * 100 # cm
            joint_errors[mp_name].append(err)
            current_frame_errors[mp_name] = err
            
        # Calculate Marker Errors (MP vs SMPL Virtual Markers)
        for m_name, mp_name in marker_to_mp_map.items():
            p_mp = mp_world[mp_name]
            target_pos = np.array([p_mp['x'], -p_mp['y'], p_mp['z']])
            marker_pos = np.array(virtual_markers[m_name])
            
            err = np.linalg.norm(target_pos - marker_pos) * 100 # cm
            marker_errors[m_name].append(err)
            
        frame_results.append({
            'frame_idx': frame_idx,
            'joint_errors': current_frame_errors
        })

    # 5. Output Results
    # a) smpl_mpjpe.txt
    txt_path = os.path.join(output_dir, 'smpl_mpjpe.txt')
    with open(txt_path, 'w') as f:
        f.write("SMPL Fitting MPJPE Report (Units: cm)\n")
        f.write("="*50 + "\n\n")
        
        header = "Frame" + "".join([f"{name[:10]:>12}" for name in mp_to_smpl_idx.keys()])
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        
        all_errs = []
        for res in frame_results:
            line = f"{res['frame_idx']:<5}"
            for name in mp_to_smpl_idx.keys():
                err = res['joint_errors'][name]
                line += f"{err:12.2f}"
                all_errs.append(err)
            f.write(line + "\n")
            
        mean_mpjpe = np.mean(all_errs)
        
        # Calculate mean per joint
        joint_means = {name: np.mean(errors) for name, errors in joint_errors.items()}
        worst_joint = max(joint_means, key=joint_means.get)
        
        f.write("\nSummary:\n")
        f.write(f"Mean MPJPE: {mean_mpjpe:.4f} cm\n")
        f.write(f"Worst Joint: {worst_joint} ({joint_means[worst_joint]:.4f} cm)\n")

    print(f"Text report saved to {txt_path}")

    # b) smpl_error_bar.png
    plt.figure(figsize=(12, 6))
    names = list(joint_means.keys())
    means = [joint_means[n] for n in names]
    
    plt.bar(names, means, color='skyblue')
    plt.axhline(y=5.0, color='r', linestyle='--', label='Target (5cm)')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Error (cm)')
    plt.title('Average SMPL Joint Error vs MediaPipe World Landmarks')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'smpl_error_bar.png'))
    plt.close()

    # c) marker_error_bar.png
    plt.figure(figsize=(14, 6))
    marker_means = {name: np.mean(errors) for name, errors in marker_errors.items()}
    m_names = list(marker_means.keys())
    m_means = [marker_means[n] for n in m_names]
    
    plt.bar(m_names, m_means, color='lightgreen')
    plt.axhline(y=5.0, color='r', linestyle='--', label='Target (5cm)')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Error (cm)')
    plt.title('Average SMPL Virtual Marker Error vs MediaPipe World Landmarks')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'marker_error_bar.png'))
    plt.close()

    print("Charts saved successfully.")

if __name__ == "__main__":
    main()
