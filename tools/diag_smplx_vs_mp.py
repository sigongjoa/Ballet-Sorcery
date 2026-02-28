# -*- coding: utf-8 -*-
"""SMPL vs MediaPipe 관절 방향벡터 정밀도 진단 도구.
SMPL 피팅 결과와 MediaPipe World Landmarks의 방향벡터를 비교하여
이전 DOF 방식 대비 정확도 개선 여부를 수치로 확인합니다.
"""

import os
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

# 프로젝트 루트 경로 추가 및 SMPLXEngine 로드
sys.path.insert(0, '/mnt/d/progress/芭蕾呪法')
from src.smplx_engine import SMPLXEngine

# MediaPipe Landmark 이름 정의 (33개)
MP_LANDMARK_NAMES = [
    'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
    'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW',
    'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX',
    'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE',
    'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
]

def main():
    video_path = './my_data/2026_02_25/IMG_2633.MOV'
    model_dir = 'data/models/smpl'
    mp_model_path = 'data/models/pose_landmarker_lite.task'
    report_dir = 'reports/2026-02-28_smplx_ik_pipeline/assets'
    os.makedirs(report_dir, exist_ok=True)

    # 1. 초기화
    engine = SMPLXEngine(model_dir=model_dir)
    
    base_options = python.BaseOptions(model_asset_path=mp_model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=vision.RunningMode.IMAGE)
    detector = vision.PoseLandmarker.create_from_options(options)

    # 2. 프레임 추출 [100, 300, 600]
    cap = cv2.VideoCapture(video_path)
    target_frames = [100, 300, 600]
    results_by_frame = []

    for f_idx in target_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {f_idx}")
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # a. MediaPipe World Landmarks 추출
        detection_result = detector.detect(mp_image)
        if not detection_result.pose_world_landmarks:
            print(f"MP failed for frame {f_idx}")
            continue
        
        world_lms = detection_result.pose_world_landmarks[0]
        world_lm_dict = {MP_LANDMARK_NAMES[i]: {'x': lm.x, 'y': lm.y, 'z': lm.z} 
                         for i, lm in enumerate(world_lms)}

        # b. SMPL 피팅
        # SMPLXEngine.fit_frame returns (vertices, joints, meta)
        vertices, joints, meta = engine.fit_frame(world_lm_dict, num_iters=50)

        # c. 관절 비교 (parent, child, mp_start, mp_end, label)
        chains = [
            (2, 5, 24, 26, 'R-Thigh'),
            (1, 4, 23, 25, 'L-Thigh'),
            (5, 8, 26, 28, 'R-Shank'),
            (4, 7, 25, 27, 'L-Shank'),
            (17, 19, 12, 14, 'R-Arm'),
        ]
        
        frame_errors = {}
        for parent_s, child_s, mp_s, mp_e, label in chains:
            # SMPL 벡터 (output joints: (45, 3))
            s_vec = joints[child_s] - joints[parent_s]
            s_vec /= np.linalg.norm(s_vec) + 1e-6
            
            # MediaPipe 벡터 (y반전 적용)
            m_start = np.array([world_lms[mp_s].x, -world_lms[mp_s].y, world_lms[mp_s].z])
            m_end = np.array([world_lms[mp_e].x, -world_lms[mp_e].y, world_lms[mp_e].z])
            m_vec = m_end - m_start
            m_vec /= np.linalg.norm(m_vec) + 1e-6
            
            # 각도 오차
            cos_sim = np.clip(np.dot(s_vec, m_vec), -1.0, 1.0)
            angle_err = np.degrees(np.arccos(cos_sim))
            frame_errors[label] = angle_err
            
        results_by_frame.append(frame_errors)
    
    cap.release()

    # 3. 데이터 집계 및 시각화
    # 평균 오차 계산
    labels = ['R-Thigh', 'L-Thigh', 'R-Shank', 'L-Shank', 'R-Arm']
    smpl_errors = [np.mean([res[l] for res in results_by_frame]) for l in labels]
    
    # 이전 DOF 방식 고정값
    dof_errors = [41.0, 27.8, 39.3, 29.7, 31.8]
    
    # a. grouped bar chart
    x = np.arange(len(labels))
    width = 0.35

    plt.switch_backend('Agg')
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, dof_errors, width, label='Prev DOF Method (Avg 33.9°)')
    rects2 = ax.bar(x + width/2, smpl_errors, width, label='New SMPL Fitting')

    ax.set_ylabel('Angular Error (degrees)')
    ax.set_title('Joint Vector Error Comparison: Prev DOF vs New SMPL Fitting')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.axhline(y=15, color='r', linestyle='--', label='Target Accuracy (15°)')
    ax.grid(axis='y', linestyle=':', alpha=0.7)

    plt.savefig(os.path.join(report_dir, 'smplx_vs_mp_error.png'))
    print(f"Saved plot to {report_dir}/smplx_vs_mp_error.png")

    # b. summary text
    with open(os.path.join(report_dir, 'smplx_summary.txt'), 'w') as f:
        f.write("=== SMPL Fitting Precision Report ===\n")
        f.write(f"Analyzed Frames: {target_frames}\n\n")
        f.write("Chain Error Comparison (degrees):\n")
        f.write(f"{'Chain':<12} | {'Prev DOF':<10} | {'SMPL Fitting':<12} | {'Improvement'}\n")
        f.write("-" * 55 + "\n")
        
        total_prev = sum(dof_errors)
        total_smpl = sum(smpl_errors)
        
        for i, l in enumerate(labels):
            imp = (dof_errors[i] - smpl_errors[i]) / dof_errors[i] * 100
            f.write(f"{l:<12} | {dof_errors[i]:<10.1f} | {smpl_errors[i]:<12.1f} | {imp:>10.1f}%\n")
        
        f.write("-" * 55 + "\n")
        avg_imp = (total_prev - total_smpl) / total_prev * 100
        f.write(f"{'Average':<12} | {np.mean(dof_errors):<10.1f} | {np.mean(smpl_errors):<12.1f} | {avg_imp:>10.1f}%\n")

    print(f"Saved summary to {report_dir}/smplx_summary.txt")

if __name__ == "__main__":
    main()
