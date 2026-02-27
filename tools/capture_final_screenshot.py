# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.joint_analyzer import JointAnalyzerApp

def capture_combined_screenshot(video_path):
    print("[Capture] 최종 분석 리포트 이미지 생성 중...")
    app = JointAnalyzerApp(video_path)
    
    # 0번 프레임 데이터 획득
    app.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = app.cap.read()
    data = app.pipeline.get_frame_data(0)
    
    # 1. 2D 랜드마크 오버레이
    video_img = app._draw_2d_skeleton(frame, data.get('landmarks_norm'))
    video_img = cv2.resize(video_img, (640, 480))
    
    # 2. 3D 뼈대 렌더링
    fk_pos = data.get('fk_positions', {})
    fk_img_rgba = app.renderer.render(fk_pos)
    fk_img = cv2.cvtColor(fk_img_rgba, cv2.COLOR_RGBA2BGR)
    fk_img = cv2.resize(fk_img, (640, 480))
    
    # 3. 이미지 병합 (좌: 영상, 우: 3D)
    combined = np.hstack((video_img, fk_img))
    
    # 하단에 캡션 추가
    canvas = np.zeros((combined.shape[0] + 60, combined.shape[1], 3), dtype=np.uint8)
    canvas[:480, :, :] = combined
    cv2.putText(canvas, "Ballet Joint Analyzer [Pro] - Integrated Analysis View", (40, 515), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    save_path = "reports/final_platform_screenshot.png"
    cv2.imwrite(save_path, canvas)
    print(f"✅ 성공: 리포트 이미지가 저장되었습니다: {save_path}")
    
    app._cleanup()

if __name__ == "__main__":
    capture_combined_screenshot("my_data/2026_02_25/IMG_2633.MOV")
