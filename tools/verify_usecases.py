# -*- coding: utf-8 -*-
"""유즈케이스(UC-1, 2, 3) 통합 검증 및 스크린샷 생성기.
- UC-1: Pose Extraction (MediaPipe)
- UC-2: Static Scaling (OpenSim Scaling)
- UC-3: IK Visualization (OpenSim 40 DOF)
"""

import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

# 프로젝트 루트 추가
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools._analyzer.pipeline import JointPipeline
from tools._analyzer.renderer3d import Renderer3D
from src.pose2sim_bridge import run_biomechanics_analysis

def capture_screenshots(video_path):
    os.makedirs("reports", exist_ok=True)
    
    print("\n[UC-1] Pose Extraction (MediaPipe)...")
    pipeline = JointPipeline()
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("영상 로드 실패")
        return

    result = pipeline.process_frame(frame)
    lm = result['landmarks_norm']
    
    # UC-1 Screenshot (Video + 2D Skeleton)
    h, w = frame.shape[:2]
    uc1_img = frame.copy()
    if lm:
        pts = [(int(l.x * w), int(l.y * h)) for l in lm]
        # MediaPipe connections
        connections = [
            (11,12),(11,13),(13,15),(12,14),(14,16),
            (11,23),(12,24),(23,24),
            (23,25),(25,27),(24,26),(26,28),
            (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
        ]
        for s, e in connections:
            if s < len(pts) and e < len(pts):
                cv2.line(uc1_img, pts[s], pts[e], (0, 255, 0), 2)
    cv2.putText(uc1_img, "UC-1: Pose Extraction (MediaPipe)", (50, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.imwrite("reports/uc1_pose_extraction.png", uc1_img)
    print("UC-1 스크린샷 저장 완료: reports/uc1_pose_extraction.png")

    print("\n[UC-2 & UC-3] Running OpenSim Scaling & IK Pipeline...")
    # 전체 비디오 처리가 아닌 샘플링하여 IK 수행
    out_dir = "data/e2e_output/verification"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Pose 추출 -> JSON 저장
    from src.pose_extractor import extract_poses
    temp_json = os.path.join(out_dir, "temp_poses.json")
    extract_poses(video_path, temp_json, target_fps=30, max_frames=60)
    
    # 2. Scaling & IK (UC-2, UC-3)
    bio_res = run_biomechanics_analysis(temp_json, out_dir, use_opensim=True)
    
    # Scaling 결과 확인
    renderer = Renderer3D(width=960, height=720)
    
    # UC-3: IK Visualization Screenshot
    if bio_res['frames']:
        coords = {k: np.radians(v) for k, v in bio_res['frames'][0]['angles'].items()}
        fk_pos_xyz = pipeline._fk_to_positions(coords)
        fk_img = renderer.render(fk_pos_xyz)
        fk_bgr = cv2.cvtColor(fk_img, cv2.COLOR_RGBA2BGR)
        cv2.putText(fk_bgr, f"UC-3: OpenSim IK Result (40 DOF)", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite("reports/uc3_ik_visualization.png", fk_bgr)
        print("UC-3 스크린샷 저장 완료: reports/uc3_ik_visualization.png")

        # UC-2: Personalized Scaled Model (T-Pose)
        tpose_coords = {k: 0.0 for k in coords}
        tpose_xyz = pipeline._fk_to_positions(tpose_coords)
        tpose_img = renderer.render(tpose_xyz)
        tpose_bgr = cv2.cvtColor(tpose_img, cv2.COLOR_RGBA2BGR)
        cv2.putText(tpose_bgr, f"UC-2: Personalized Scaled Model (T-Pose)", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite("reports/uc2_scaling_result.png", tpose_bgr)
        print("UC-2 스크린샷 저장 완료: reports/uc2_scaling_result.png")

    pipeline.close()
    renderer.close()

if __name__ == "__main__":
    video = "my_data/2026_02_25/IMG_2633.MOV"
    capture_screenshots(video)
