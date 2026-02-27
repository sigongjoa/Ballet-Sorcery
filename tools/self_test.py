# -*- coding: utf-8 -*-
"""엔진 및 렌더러 자가 진단 스크립트 (Fix)."""

import os
import sys
import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools._analyzer.pipeline import JointPipeline
from tools._analyzer.renderer3d import Renderer3D

def run_self_verification(video_path):
    os.makedirs("reports", exist_ok=True)
    print("\n[Self-Test] Verification Started: " + video_path)
    
    pipeline = JointPipeline()
    renderer = Renderer3D(width=640, height=480)
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read video.")
        return False

    # 1. Pipeline Test
    print("[Self-Test] Step 1: Testing Pipeline (MediaPipe + OpenSim)...")
    data = pipeline.get_frame_data(0, frame)
    
    if not data['coords']:
        print("Error: OpenSim coordinate data is empty.")
        return False
    print("Success: Got OpenSim coordinates (" + str(len(data['coords'])) + " DOFs)")

    # 2. Renderer Test
    print("[Self-Test] Step 2: Testing 3D Renderer output...")
    fk_img = renderer.render(data['fk_positions'])
    
    # 3. Analyze Output
    rgb_sum = np.sum(fk_img[:, :, :3])
    print("   - 3D Rendering Pixel Sum: " + str(rgb_sum))
    
    if rgb_sum < 1000:
        print("Error: 3D Render result is a black screen.")
        return False
    print("Success: 3D Renderer produced valid graphics.")

    # 4. Save Evidence
    cv2.imwrite("reports/test_step1_video.png", frame)
    cv2.imwrite("reports/test_step2_3d.png", cv2.cvtColor(fk_img, cv2.COLOR_RGBA2BGR))
    print("Success: Evidence saved to reports/test_step*.png")
    
    pipeline.close()
    renderer.close()
    return True

if __name__ == "__main__":
    video = "my_data/2026_02_25/IMG_2633.MOV"
    if run_self_verification(video):
        sys.exit(0)
    else:
        sys.exit(1)
