# -*- coding: utf-8 -*-
"""SMPL T-pose 좌우 대칭성 검증 스크립트.
SMPL 모델의 표준 관절과 가상 마커(src/smplx_mapper.py)의 
T-pose 상태에서의 좌우 대칭성을 정밀 검증합니다.
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Chumpy/SMPL compatibility monkey-patch (smplx import 전에 실행)
np.bool = np.bool_
np.int = np.int_
np.float = np.float64
np.complex = np.complex128
np.object = np.object_
np.unicode = np.str_
np.str = np.str_

import smplx
from src.smplx_mapper import VIRTUAL_MARKER_MAP

def verify_symmetry():
    print("[Verification] SMPL T-pose Symmetry Analysis...")
    
    # 1. SMPL 모델 로드
    model_dir = 'data/models/smpl'
    device = torch.device('cpu')
    smpl = smplx.create(model_dir, model_type='smpl', gender='neutral', ext='pkl').to(device)
    smpl.eval()

    # 2. T-pose 생성 (body_pose=0, betas=0)
    with torch.no_grad():
        betas = torch.zeros(1, 10, device=device)
        body_pose = torch.zeros(1, 69, device=device)
        output = smpl(betas=betas, body_pose=body_pose)
        
    joints = output.joints[0].cpu().numpy()  # (45, 3) 또는 (24, 3) 등. SMPL 기본은 45개 포함
    vertices = output.vertices[0].cpu().numpy() # (6890, 3)

    # 결과 저장을 위한 리스트
    report_lines = []
    report_lines.append("=== SMPL T-pose Symmetry Verification Report ===")
    report_lines.append(f"Model Path: {model_dir}")
    report_lines.append("-" * 50)

    # 3. a) SMPL 23개 관절 좌우 대칭성 검증
    # SMPL 23관절 인덱스 (0=Pelvis, 1=L_Hip, 2=R_Hip, ...)
    joint_pairs = [
        ("Hip", 1, 2),
        ("Knee", 4, 5),
        ("Ankle", 7, 8),
        ("Shoulder", 16, 17),
        ("Elbow", 18, 19),
        ("Wrist", 20, 21),
        ("Foot", 10, 11),
    ]

    report_lines.append("[A] Standard Joint Symmetry (Tolerance: 5e-3)")
    for name, l_idx, r_idx in joint_pairs:
        l_pos = joints[l_idx]
        r_pos = joints[r_idx]
        
        # X좌표 부호 및 대칭성 확인
        x_symm = abs(l_pos[0] + r_pos[0]) < 5e-3
        x_sign = (l_pos[0] > 0 and r_pos[0] < 0)
        y_match = abs(l_pos[1] - r_pos[1]) < 5e-3
        z_match = abs(l_pos[2] - r_pos[2]) < 5e-3
        
        status = "PASS" if (x_symm and x_sign and y_match and z_match) else "FAIL"
        report_lines.append(f"{name:10}: L[{l_pos[0]:.4f}, {l_pos[1]:.4f}, {l_pos[2]:.4f}] | "
                            f"R[{r_pos[0]:.4f}, {r_pos[1]:.4f}, {r_pos[2]:.4f}] -> {status}")

    report_lines.append("-" * 50)

    # 3. b) 가상 마커 대칭성 검증 (VIRTUAL_MARKER_MAP)
    report_lines.append("[B] Virtual Marker Symmetry (Check Sign & Magnitude)")
    
    marker_names = sorted(VIRTUAL_MARKER_MAP.keys())
    l_markers = [m for m in marker_names if m.endswith('_L')]
    
    for l_name in l_markers:
        r_name = l_name[:-2] + '_R'
        if r_name in VIRTUAL_MARKER_MAP:
            l_idx = VIRTUAL_MARKER_MAP[l_name]
            r_idx = VIRTUAL_MARKER_MAP[r_name]
            
            l_pos = vertices[l_idx]
            r_pos = vertices[r_idx]
            
            # 요구사항: 좌우 마커 x좌표가 반대 부호인지 확인
            x_sign_ok = (l_pos[0] * r_pos[0] < 0)
            # 추가: 대칭성 확인
            x_symm_err = abs(l_pos[0] + r_pos[0])
            x_symm_ok = x_symm_err < 1e-3
            
            status = "PASS" if (x_sign_ok and x_symm_ok) else "FAIL"
            if not x_sign_ok: status += "(SIGN_ERR)"
            elif not x_symm_ok: status += "(ASYMM)"

            report_lines.append(f"{l_name:20} vs {r_name:20}: {status} | X_sum: {l_pos[0]+r_pos[0]:.6f}")

    # 4. 결과 저장
    report_dir = 'reports/2026-02-28_pipeline_validation/assets'
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, 'tpose_symmetry.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    print(f"[Result] Report saved to {report_path}")

    # 5. 시각화 (Front View X-Y)
    plt.figure(figsize=(10, 12))
    
    # 중앙 관절 (Pelvis, Spine, Neck, Head 등)
    center_indices = [0, 3, 6, 9, 12, 15, 22]
    left_indices = [1, 4, 7, 10, 13, 16, 18, 20]
    right_indices = [2, 5, 8, 11, 14, 17, 19, 21]

    plt.scatter(joints[center_indices, 0], joints[center_indices, 1], c='green', label='Center', s=100)
    plt.scatter(joints[left_indices, 0], joints[left_indices, 1], c='blue', label='Left', s=100)
    plt.scatter(joints[right_indices, 0], joints[right_indices, 1], c='red', label='Right', s=100)

    # 가상 마커 시각화 (작은 점)
    plt.scatter(vertices[:, 0], vertices[:, 1], c='gray', alpha=0.05, s=1)
    
    # 텍스트 라벨링 (주요 관절)
    for i in range(23):
        plt.text(joints[i, 0], joints[i, 1], str(i), fontsize=9)

    plt.title("SMPL T-pose Joint Positions (Front View X-Y)")
    plt.xlabel("X (Left > 0)")
    plt.ylabel("Y (Up > 0)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    viz_path = os.path.join(report_dir, 'tpose_joints_viz.png')
    plt.savefig(viz_path)
    plt.close()
    print(f"[Result] Visualization saved to {viz_path}")

if __name__ == "__main__":
    verify_symmetry()
