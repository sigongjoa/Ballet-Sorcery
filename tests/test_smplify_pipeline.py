# -*- coding: utf-8 -*-
import sys
import os
import torch
import numpy as np
import json
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.smpl_visualizer import fit_smpl_to_joints, mediapipe_to_smpl_joints, MaxMixturePrior
import smplx

def run_verification():
    print("\n" + "="*60)
    print(" [Verification] SMPLify-3D Pipeline & Marker Extraction")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_dir = "data/models/smpl"
    json_path = "reports/joint_analysis_result/extracted_poses.json"
    
    if not os.path.exists(json_path):
        print("Error: Input data not found.")
        return
    
    try:
        smpl_model = smplx.create(model_dir, model_type='smpl', gender='neutral', ext='pkl').to(device)
        pose_prior = MaxMixturePrior(prior_folder=model_dir, num_gaussians=8).to(device)
        with open(json_path, 'r') as f:
            data = json.load(f)
        mp_landmarks = data['frames'][0]['world_landmarks']
    except Exception as e:
        print("Initialization Error: " + str(e))
        return

    # 1. Mapping Validation
    joints, conf = mediapipe_to_smpl_joints(mp_landmarks)
    l_hip_y, l_sho_y = joints[1, 1], joints[16, 1]
    if l_sho_y > l_hip_y:
        print("✅ Axis Check: SUCCESS.")
    else:
        print("❌ Axis Check: FAILED.")

    # 2. Optimization Loop
    print("Running SMPLify-3D Optimization...")
    verts, joints_fit, pose, betas, transl = fit_smpl_to_joints(
        joints, conf, smpl_model, pose_prior, device, num_iters=100)
    
    pose_norm = np.linalg.norm(pose)
    print("   - Pose Param Norm: " + str(round(pose_norm, 4)))
    
    if pose_norm > 0.1:
        print("✅ Optimization Check: SUCCESS.")
    
    # 3. Virtual Marker Extraction
    V_MARKERS = {'C7': 3470, 'ASIS_L': 3120, 'ASIS_R': 6559, 'KNEE_L': 1050, 'HEEL_R': 6787}
    extracted_count = 0
    for name, idx in V_MARKERS.items():
        if not np.isnan(verts[idx]).any(): extracted_count += 1
            
    if extracted_count == len(V_MARKERS):
        print("✅ Extraction Check: SUCCESS (All " + str(len(V_MARKERS)) + " markers extracted).")
        print("\n[Result] Sample ASIS_L: " + str(np.round(verts[3120], 3)))
    else:
        print("❌ Extraction Check: FAILED.")

    print("\n" + "="*60)
    print(" ALL TESTS COMPLETED")
    print("="*60)

if __name__ == "__main__":
    run_verification()
