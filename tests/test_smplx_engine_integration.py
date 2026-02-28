# -*- coding: utf-8 -*-
import sys
import os
import json
import torch
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.smplx_engine import SMPLXEngine
from src.smplx_mapper import SMPLXToOpenSimMapper

def test_full_pipeline():
    print("\n[Test] SMPL-X Fitting & Virtual Marker Extraction Pipeline")
    
    # 1. 데이터 로드
    json_path = "reports/joint_analysis_result/extracted_poses.json"
    if not os.path.exists(json_path):
        print("Error: Input data not found.")
        return
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    frame_data = data['frames'][0]['world_landmarks']

    # 2. 엔진 초기화
    model_dir = "data/models/smpl"
    try:
        engine = SMPLXEngine(model_dir)
        mapper = SMPLXToOpenSimMapper()
    except Exception as e:
        print("Error during initialization: " + str(e))
        return

    # 3. 피팅 수행
    print("   - Running SMPLify-3D Optimization (100 iters)...")
    verts, meta = engine.fit_frame(frame_data, num_iters=100)
    print("   - Final Fitting Loss: " + str(round(meta['loss'], 4)))

    # 4. 가상 마커 추출
    print("   - Extracting 56 High-Density Markers...")
    markers = mapper.extract_virtual_markers(verts)
    
    # 5. 최종 검증
    if len(markers) == 56:
        print("✅ SUCCESS: 56 Virtual Markers Extracted.")
        print("   - Sample [C7]: " + str(markers['C7']))
        print("   - Sample [TOE_TIP_R]: " + str(markers['TOE_TIP_R']))
    else:
        print("FAILURE: Marker count mismatch (" + str(len(markers)) + " / 56)")

if __name__ == "__main__":
    test_full_pipeline()
