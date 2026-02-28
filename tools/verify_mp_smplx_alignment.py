# -*- coding: utf-8 -*-
"""실제 MediaPipe 데이터를 활용한 SMPL-X 관절 정합성 정밀 검증.
dummy가 아닌 실제 분석 결과 파일을 읽어 오차를 계산함.
"""

import json
import numpy as np
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.smpl_visualizer import mediapipe_to_smpl_joints

def verify_real_data_alignment(json_path):
    print("\n[Verification] Real-world Data Alignment: MediaPipe <-> SMPLX")
    
    if not os.path.exists(json_path):
        print(f"❌ 실패: 데이터 파일이 없습니다: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 첫 번째 유효 프레임 선택
    frame = data['frames'][0]
    mp_world_landmarks = frame.get('world_landmarks')
    
    if not mp_world_landmarks:
        print("❌ 실패: world_landmarks 데이터가 프레임에 없습니다.")
        return

    # 1. MediaPipe -> SMPL-X 변환 실행 (src/smpl_visualizer.py 로직 재사용)
    # 로직: y축 반전 (-y) 포함됨
    joints, confidence = mediapipe_to_smpl_joints(mp_world_landmarks)
    
    print(f"1. Data Integrity: {int(confidence.sum())} joints mapped from MediaPipe.")

    # 2. 정합성 수치 검사 (Relative Proportions)
    # 골반 너비 vs 어깨 너비 비율이 인체 공학적으로 타당한지 체크
    l_hip, r_hip = joints[1], joints[2]
    l_sho, r_sho = joints[16], joints[17]
    
    pelvis_w = np.linalg.norm(l_hip - r_hip)
    shoulder_w = np.linalg.norm(l_sho - r_sho)
    ratio = shoulder_w / pelvis_w if pelvis_w > 0 else 0
    
    print(f"2. Biometric Proportions:")
    print(f"   - Pelvis Width: {pelvis_w:.3f}m")
    print(f"   - Shoulder Width: {shoulder_w:.3f}m")
    print(f"   - Shoulder/Pelvis Ratio: {ratio:.2f} (Normal: 1.2~1.6)")
    
    if 1.0 <= ratio <= 2.0:
        print("   ✅ 정합성 통과: 인체 비례가 SMPL-X 모델에 주입하기에 적절함.")
    else:
        print("   ⚠️ 경고: 비정상적 비례 감지. 데이터 노이즈 확인 필요.")

    # 3. Y-Axis Orientation Re-check
    # 무릎이 골반보다 아래(-Y)에 있는지 확인 (SMPL-X는 +Y가 위)
    l_knee = joints[4]
    knee_rel_y = l_knee[1] - l_hip[1]
    print(f"3. Vertical Orientation Check (Knee rel to Hip): {knee_rel_y:.3f}m")
    
    if knee_rel_y < 0:
        print("   ✅ 수직 방향성 일치: 무릎이 골반 아래에 위치함.")
    else:
        print("   ❌ 수직 방향성 오류: MediaPipe 좌표계가 제대로 반전되지 않음.")

if __name__ == "__main__":
    # 프로젝트 내 실제 데이터 경로 사용
    real_json = "reports/joint_analysis_result/extracted_poses.json"
    verify_real_data_alignment(real_json)
