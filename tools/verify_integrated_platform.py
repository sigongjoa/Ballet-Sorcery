# -*- coding: utf-8 -*-
"""통합 분석 플랫폼(EXE) 엔진 최종 검증 스크립트."""

import os
import sys
import shutil

# 프로젝트 루트 추가
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.joint_analyzer import JointAnalyzerApp

def verify_platform(video_path):
    output_dir = os.path.join("reports", "joint_analysis_result")
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir) # 깨끗한 상태에서 시작
        
    print(f"\n[Platform] 통합 분석 플랫폼 엔진 가동 중: {video_path}")
    
    # GUI 없이 앱의 핵심 로직만 테스트
    app = JointAnalyzerApp(video_path)
    
    # 1. Scaling 성공 여부 확인
    scaled_model = os.path.join(output_dir, "scaled_model.osim")
    if os.path.exists(scaled_model):
        print(f"✅ UC-2 (Scaling) 성공: {scaled_model}")
    else:
        print("❌ UC-2 (Scaling) 실패")

    # 2. IK(Inverse Kinematics) 성공 여부 확인
    ik_mot = os.path.join(output_dir, "extracted_poses_ik.mot")
    if os.path.exists(ik_mot):
        print(f"✅ UC-3 (IK Analysis) 성공: {ik_mot}")
    else:
        print("❌ UC-3 (IK Analysis) 실패")

    # 3. 통합 결과 데이터 확인
    print(f"\n[Platform] 통합 폴더 '{output_dir}' 내용물:")
    os.system(f"ls -lh {output_dir}")

    app._cleanup()

if __name__ == "__main__":
    video = "my_data/2026_02_25/IMG_2633.MOV"
    verify_platform(video)
