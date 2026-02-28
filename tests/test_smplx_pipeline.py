# -*- coding: utf-8 -*-
import unittest
import numpy as np
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.smplx_mapper import SMPLXToOpenSimMapper
from src.smpl_visualizer import mediapipe_to_smpl_joints

class TestSMPLXPipeline(unittest.TestCase):
    def test_mapping_consistency(self):
        """매퍼가 정의된 개수만큼 마커를 정확히 추출하는지 테스트."""
        mapper = SMPLXToOpenSimMapper()
        dummy_verts = np.zeros((6890, 3))  # SMPL 6890 정점
        markers = mapper.extract_virtual_markers(dummy_verts)

        self.assertEqual(len(markers), len(mapper.mapping))
        self.assertIn('ACROMION_L', markers)   # 구 SHOULDER_L → ACROMION_L
        self.assertIn('KNEE_LAT_R', markers)   # 구 KNEE_R_LAT → KNEE_LAT_R

    def test_mediapipe_conversion(self):
        """MediaPipe 랜드마크가 SMPL 관절 형식으로 정상 변환되는지 테스트."""
        dummy_mp = {
            'LEFT_HIP': {'x': 0.1, 'y': 0.5, 'z': 0.0},
            'RIGHT_HIP': {'x': -0.1, 'y': 0.5, 'z': 0.0},
            'LEFT_KNEE': {'x': 0.1, 'y': 0.8, 'z': 0.0}
        }
        joints, conf = mediapipe_to_smpl_joints(dummy_mp)
        
        self.assertEqual(joints.shape, (22, 3))
        self.assertGreater(conf[1], 0) # LEFT_HIP
        self.assertGreater(conf[4], 0) # LEFT_KNEE
        self.assertEqual(conf[10], 0)  # Unmapped joint

if __name__ == "__main__":
    unittest.main()
