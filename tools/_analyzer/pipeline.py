# -*- coding: utf-8 -*-
"""MediaPipe + OpenSim IK 분석 엔진 (Pre-processing Version).
실시간 분석을 배제하고, 사전 계산된 데이터를 로드하여 GUI에 공급함.
"""

import sys
import os
import json
import numpy as np
from typing import Dict, Any, List

# 프로젝트 루트 추가
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.pose2sim_bridge import run_biomechanics_analysis
from src.anatomy_overlay_viewer import compute_fk, OPENSIM_MODEL_PATH

class JointPipeline:
    """분석 플랫폼용 오프라인 엔진: 계산과 시각화를 분리함."""

    def __init__(self, osim_path: str = OPENSIM_MODEL_PATH):
        import pyopensim
        self.osim_path = osim_path
        self.osim_model = None
        self.osim_state = None
        self.analysis_data = None # 전체 프레임 분석 결과 저장소

    def run_offline_analysis(self, video_path: str, output_dir: str):
        """[중요] GUI가 뜨기 전에 모든 계산을 마침. MediaPipe vs PyVista 충돌 방지."""
        print(f"[Engine] 단계 1: 포즈 추출 및 OpenSim IK 분석 시작...")
        
        # 1. 포즈 추출
        from src.pose_extractor import extract_poses
        temp_json = os.path.join(output_dir, "extracted_poses.json")
        if not os.path.exists(temp_json):
            extract_poses(video_path, temp_json, target_fps=30)
        
        # 2. OpenSim 파이프라인 (Scaling -> IK -> ID)
        self.analysis_data = run_biomechanics_analysis(temp_json, output_dir, use_opensim=True)
        
        # 3. 모델 로드
        import pyopensim
        model_path = self.analysis_data.get('scaled_model_path', self.osim_path)
        self.osim_model = pyopensim.Model(model_path)
        self.osim_state = self.osim_model.initSystem()
        print(f"[Engine] 단계 2: 모든 분석 데이터 로드 완료.")

    def get_frame_data(self, frame_idx: int) -> dict:
        if not self.analysis_data or frame_idx >= len(self.analysis_data['frames']):
            return {'coords': {}, 'fk_positions': {}, 'landmarks_norm': None}

        raw_angles = self.analysis_data['frames'][frame_idx]['angles']
        coords = {k: np.radians(v) for k, v in raw_angles.items()}
        fk_positions = self.compute_fk_for_coords(coords)
        lm_norm = self.analysis_data['frames'][frame_idx].get('landmarks_norm')

        return {
            'landmarks_norm': lm_norm,
            'coords':         coords,
            'fk_positions':   fk_positions
        }

    def compute_fk_for_coords(self, coords: dict) -> dict:
        if self.osim_model is None: return {}
        try:
            from src.anatomy_overlay_viewer import compute_fk
            transforms = compute_fk(self.osim_model, self.osim_state, coords)
            return {name: [T[0, 3], T[1, 3], T[2, 3]] for name, T in transforms.items()}
        except Exception as e:
            return {}

    def close(self):
        pass
