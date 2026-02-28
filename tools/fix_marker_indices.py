# -*- coding: utf-8 -*-
"""SMPL 가상 마커 인덱스 자동 수정 및 대칭 최적화 도구 (V3).
SMPL joints 위치를 기준으로 해부학적으로 타당하고 좌우 대칭인 
최적의 버텍스 인덱스를 찾아 src/smplx_mapper.py를 업데이트합니다.
Joint 위치 자체를 대칭화하여 검색 타겟을 설정하고, 대규모 후보군에서 최적 쌍을 찾습니다.
"""

import os
import torch
import numpy as np
from scipy.spatial import KDTree
import re

# Monkey-patch numpy for smplx compatibility
np.bool = np.bool_
np.int = np.int_
np.float = np.float64
np.complex = np.complex128
np.object = np.object_
np.unicode = np.str_
np.str = np.str_

import smplx

def find_best_indices():
    print("[Fix] Loading SMPL model for vertex search...")
    
    # 1. SMPL 모델 로드
    model_dir = 'data/models/smpl'
    device = torch.device('cpu')
    smpl = smplx.create(model_dir, model_type='smpl', gender='neutral', ext='pkl').to(device)
    smpl.eval()

    # 2. T-pose 생성
    with torch.no_grad():
        betas = torch.zeros(1, 10, device=device)
        body_pose = torch.zeros(1, 69, device=device)
        output = smpl(betas=betas, body_pose=body_pose)
        
    joints = output.joints[0].cpu().numpy()  # (45, 3)
    vertices = output.vertices[0].cpu().numpy() # (6890, 3)
    
    kdtree = KDTree(vertices)

    # SMPL Joint Pairs (L_idx, R_idx)
    joint_pairs_map = {
        16: 17, # Shoulder
        18: 19, # Elbow
        20: 21, # Wrist
        1: 2,   # Hip
        4: 5,   # Knee
        7: 8,   # Ankle
        10: 11, # Foot
        13: 14, # Collar
    }

    # 3. 마커별 탐색 정의
    marker_search_queries = [
        ('ACROMION', 16, [0.03, 0.05, 0.0], "Shoulder top-outer"),
        ('HUMERUS_LAT', 16, [0.05, -0.05, 0.0], "Upper arm lateral"),
        ('ELBOW_LAT', 18, [0.05, 0.0, -0.02], "Elbow lateral"),
        ('ELBOW_MED', 18, [-0.04, 0.0, -0.01], "Elbow medial"),
        ('WRIST_RAD', 20, [0.03, 0.0, 0.02], "Wrist radial (thumb side)"),
        ('WRIST_ULN', 20, [0.03, 0.0, -0.03], "Wrist ulnar (pinky side)"),
        ('HAND_BACK', 20, [0.08, -0.02, 0.0], "Hand back"),
        ('MIDDLE_FINGER_BASE', 20, [0.12, -0.03, 0.0], "Middle finger base"),
        ('GTROCHANTER', 1, [0.10, 0.0, 0.0], "Greater trochanter (hip outer)"),
        ('KNEE_LAT', 4, [0.06, 0.0, 0.02], "Knee lateral"),
        ('KNEE_MED', 4, [-0.05, 0.0, 0.02], "Knee medial"),
        ('ANKLE_LAT', 7, [0.04, -0.02, 0.0], "Ankle lateral"),
        ('ANKLE_MED', 7, [-0.04, -0.02, 0.0], "Ankle medial"),
        ('HEEL_POST', 10, [0.0, -0.04, -0.06], "Heel posterior"),
        ('MIDFOOT_UP', 10, [0.0, 0.03, 0.05], "Midfoot top"),
        ('TOE_1ST_BASE', 10, [-0.03, -0.02, 0.12], "Big toe base"),
        ('TOE_5TH_BASE', 10, [0.06, -0.02, 0.10], "Small toe base"),
        ('TOE_TIP', 10, [0.0, -0.02, 0.16], "Toe tip"),
        ('ASIS', 1, [0.08, 0.08, 0.10], "ASIS (pelvis front-outer)"),
        ('PSIS', 1, [0.05, 0.08, -0.08], "PSIS (pelvis back-inner-ish)"),
        ('ILIAC_CREST', 1, [0.12, 0.15, 0.0], "Iliac crest (waist side)"),
        ('RIB_LOW', 6, [0.15, 0.0, 0.0], "Lower rib side"),
    ]

    center_search_queries = [
        ('HEAD_TOP', 15, [0.0, 0.10, 0.0], "Head top"),
        ('SELLION', 15, [0.0, -0.02, 0.08], "Nose bridge"),
        ('C7', 12, [0.0, 0.0, -0.08], "C7 vertebra (neck back)"),
        ('CLAV_NOTCH', 12, [0.0, -0.05, 0.08], "Clavicle notch (neck front)"),
        ('T12', 3, [0.0, 0.0, -0.08], "T12 vertebra"),
        ('T8', 6, [0.0, 0.0, -0.09], "T8 vertebra"),
        ('T4', 9, [0.0, 0.0, -0.08], "T4 vertebra"),
        ('L5', 0, [0.0, 0.05, -0.10], "L5 vertebra"),
        ('STERNUM_UP', 9, [0.0, -0.05, 0.12], "Sternum upper"),
        ('STERNUM_LOW', 6, [0.0, -0.05, 0.13], "Sternum lower"),
        ('PUBIC_SYMPH', 0, [0.0, -0.08, 0.10], "Pubic symphysis"),
        ('SACRUM', 0, [0.0, -0.02, -0.10], "Sacrum"),
    ]

    new_map = {}

    # 4. 좌우 쌍 검색 (Symmetry Optimization)
    for name, l_joint_idx, offset, desc in marker_search_queries:
        l_name = f"{name}_L"
        r_name = f"{name}_R"
        
        # Symmetrize Joint Center
        r_joint_idx = joint_pairs_map.get(l_joint_idx, l_joint_idx)
        j_l = joints[l_joint_idx]
        j_r = joints[r_joint_idx]
        
        avg_y = (j_l[1] + j_r[1]) / 2
        avg_z = (j_l[2] + j_r[2]) / 2
        avg_x = (j_l[0] - j_r[0]) / 2
        
        sym_j_l = np.array([avg_x, avg_y, avg_z])
        target_l = sym_j_l + np.array(offset)
        
        # Candidate search (find 100 nearest)
        dists, idxs = kdtree.query(target_l, k=100)
        
        best_pair = (idxs[0], 0)
        min_score = 1e9
        
        for l_idx in idxs:
            v_l = vertices[l_idx]
            # Mirror target for R
            target_r_v = v_l.copy()
            target_r_v[0] = -target_r_v[0]
            
            # Find best R candidate for this L candidate
            _, r_idx = kdtree.query(target_r_v)
            v_r = vertices[r_idx]
            
            # Symmetry score: prioritize X-sum, then Y/Z diff
            x_sum = abs(v_l[0] + v_r[0])
            y_diff = abs(v_l[1] - v_r[1])
            z_diff = abs(v_l[2] - v_r[2])
            score = x_sum * 10 + y_diff + z_diff # X-symmetry is most important
            
            if score < min_score:
                min_score = score
                best_pair = (l_idx, r_idx)
        
        l_final, r_final = best_pair
        new_map[l_name] = int(l_final)
        new_map[r_name] = int(r_final)
        v_l_f = vertices[l_final]
        v_r_f = vertices[r_final]
        print(f"Found {l_name}/{r_name}: {l_final}/{r_final} (X_sum: {v_l_f[0]+v_r_f[0]:.6f})")

    # 5. 중앙 마커 검색 (최대한 X=0 근처 버텍스)
    for name, joint_idx, offset, desc in center_search_queries:
        target = joints[joint_idx] + np.array(offset)
        target[0] = 0
        
        dists, idxs = kdtree.query(target, k=50)
        best_idx = idxs[0]
        min_x = 1e9
        for idx in idxs:
            if abs(vertices[idx][0]) < min_x:
                min_x = abs(vertices[idx][0])
                best_idx = idx
        
        new_map[name] = int(best_idx)
        print(f"Found {name}: {best_idx} (X: {vertices[best_idx][0]:.6f})")

    # 6. src/smplx_mapper.py 업데이트
    mapper_path = 'src/smplx_mapper.py'
    with open(mapper_path, 'r', encoding='utf-8') as f:
        content = f.read()

    sorted_items = sorted(new_map.items())
    map_str = "VIRTUAL_MARKER_MAP = {\n"
    current_group = ""
    for k, v in sorted_items:
        group = k.split('_')[0]
        if group != current_group:
            map_str += f"    # --- {group} ---\n"
            current_group = group
        map_str += f"    '{k}': {v},\n"
    map_str += "}"

    pattern = r"VIRTUAL_MARKER_MAP = \{.*?\}"
    new_content = re.sub(pattern, map_str, content, flags=re.DOTALL)

    with open(mapper_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"\n[Success] Updated {mapper_path} with {len(new_map)} markers.")

if __name__ == "__main__":
    find_best_indices()
    
    # 7. 검증 실행
    print("\n[Verification] Running symmetry check...")
    import subprocess
    subprocess.run(["python3", "tools/verify_tpose_symmetry.py"])
