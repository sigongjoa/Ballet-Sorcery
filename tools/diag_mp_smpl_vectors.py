# -*- coding: utf-8 -*-
import json
import numpy as np
import os
import sys
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.smpl_visualizer import mediapipe_to_smpl_joints, fit_smpl_to_joints, MaxMixturePrior
import smplx

def diagnose_vectors(json_path):
    print("\n" + "="*60)
    print(" [Diagnostic] MediaPipe vs SMPL Vector Alignment")
    print("="*60)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    mp_landmarks = data['frames'][0]['world_landmarks']
    
    def get_mp_vec(start_name, end_name):
        p1 = mp_landmarks[start_name]
        p2 = mp_landmarks[end_name]
        return np.array([p2['x'] - p1['x'], p2['y'] - p1['y'], p2['z'] - p1['z']])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_dir = "data/models/smpl"
    smpl_model = smplx.create(model_dir, model_type='smpl', gender='neutral', ext='pkl').to(device)
    pose_prior = MaxMixturePrior(prior_folder=model_dir, num_gaussians=8).to(device)
    
    joints_raw, conf = mediapipe_to_smpl_joints(mp_landmarks)
    verts, joints_fit, pose, betas, transl = fit_smpl_to_joints(
        joints_raw, conf, smpl_model, pose_prior, device, num_iters=100)

    print("\n[Vector Comparison] Angle Error between MP and SMPL limbs:")
    
    LIMBS = [
        ('LEFT_SHOULDER', 'LEFT_ELBOW', 16, 18),
        ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 17, 19),
        ('LEFT_HIP', 'LEFT_KNEE', 1, 4),
        ('RIGHT_HIP', 'RIGHT_KNEE', 2, 5),
        ('LEFT_KNEE', 'LEFT_ANKLE', 4, 7),
        ('RIGHT_KNEE', 'RIGHT_ANKLE', 5, 8)
    ]

    for mp_s, mp_e, sm_s, sm_e in LIMBS:
        v_mp = get_mp_vec(mp_s, mp_e)
        v_mp[1] = -v_mp[1] # MP Y-down -> SMPL Y-up
        
        v_sm = joints_fit[sm_e] - joints_fit[sm_s]
        
        unit_mp = v_mp / (np.linalg.norm(v_mp) + 1e-8)
        unit_sm = v_sm / (np.linalg.norm(v_sm) + 1e-8)
        dot = np.clip(np.dot(unit_mp, unit_sm), -1.0, 1.0)
        angle_err = np.degrees(np.arccos(dot))
        
        status = "OK" if angle_err < 15 else "MISMATCH"
        print("   - " + mp_s + " -> " + mp_e + ": Error " + str(round(angle_err, 2)) + " deg [" + status + "]")

    print("\n" + "="*60)

if __name__ == "__main__":
    real_json = "reports/joint_analysis_result/extracted_poses.json"
    diagnose_vectors(real_json)
