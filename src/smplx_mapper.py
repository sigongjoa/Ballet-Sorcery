# -*- coding: utf-8 -*-
"""SMPL 메쉬 정점 -> OpenSim 고밀도 가상 마커 매핑 모듈.
SMPL (6890 vertices) 기준 고밀도 해부학적 마커 인덱스.
바이오메카닉스 분석의 정밀도를 극대화합니다.
"""

import numpy as np

# SMPL (6890 vertices) 기준 고밀도 해부학적 마커 인덱스
# 모든 인덱스는 6890 미만이어야 함.
VIRTUAL_MARKER_MAP = {
    # Head & Neck (4)
    'HEAD_TOP': 411, 'SELLION': 336, 'C7': 829, 'CLAV_NOTCH': 3049,
    # Torso & Spine (8)
    'T12': 3022, 'T8': 3021, 'T4': 3020, 'L5': 3019,
    'STERNUM_UP': 3171, 'STERNUM_LOW': 3175,
    'RIB_L_LOW': 1336, 'RIB_R_LOW': 4800,
    # Pelvis (8)
    'ASIS_L': 3134, 'ASIS_R': 6260,
    'PSIS_L': 3018, 'PSIS_R': 6514,
    'ILIAC_CREST_L': 1242, 'ILIAC_CREST_R': 4708,
    'PUBIC_SYMPH': 1769, 'SACRUM': 3014,
    # Upper Limbs - 좌우 각 8개 = 16
    'ACROMION_L': 3011, 'ACROMION_R': 6258,
    'HUMERUS_LAT_L': 1369, 'HUMERUS_LAT_R': 4836,
    'ELBOW_LAT_L': 1657, 'ELBOW_LAT_R': 5121,
    'ELBOW_MED_L': 1647, 'ELBOW_MED_R': 5111,
    'WRIST_RAD_L': 2112, 'WRIST_RAD_R': 5573,
    'WRIST_ULN_L': 2108, 'WRIST_ULN_R': 5568,
    'HAND_BACK_L': 2214, 'HAND_BACK_R': 5674,
    'MIDDLE_FINGER_BASE_L': 2300, 'MIDDLE_FINGER_BASE_R': 5750,
    # Lower Limbs - 좌우 각 10개 = 20
    'GTROCHANTER_L': 1249, 'GTROCHANTER_R': 4715,
    'KNEE_LAT_L': 1051, 'KNEE_LAT_R': 4515,
    'KNEE_MED_L': 1044, 'KNEE_MED_R': 4508,
    'ANKLE_LAT_L': 3327, 'ANKLE_LAT_R': 6778,
    'ANKLE_MED_L': 3314, 'ANKLE_MED_R': 6765,
    'HEEL_POST_L': 3387, 'HEEL_POST_R': 6838,
    'MIDFOOT_UP_L': 3433, 'MIDFOOT_UP_R': 6884,
    'TOE_1ST_BASE_L': 3346, 'TOE_1ST_BASE_R': 6797,
    'TOE_5TH_BASE_L': 3355, 'TOE_5TH_BASE_R': 6806,
    'TOE_TIP_L': 3365, 'TOE_TIP_R': 6816,
}

# 6890 미만 클램핑 보장
VIRTUAL_MARKER_MAP = {k: min(v, 6889) for k, v in VIRTUAL_MARKER_MAP.items()}

class SMPLXToOpenSimMapper:
    """SMPL-X 엔진 클래스명을 유지하면서 SMPL 메쉬를 지원."""
    def __init__(self, mapping=VIRTUAL_MARKER_MAP):
        self.mapping = mapping

    def extract_virtual_markers(self, vertices: np.ndarray) -> dict:
        """SMPL 메쉬 정점 배열(N, 3)에서 가상 마커 좌표를 추출함."""
        markers = {}
        for name, v_idx in self.mapping.items():
            if v_idx < len(vertices):
                markers[name] = vertices[v_idx].tolist()
        return markers

    def get_marker_names(self):
        return list(self.mapping.keys())

if __name__ == "__main__":
    mapper = SMPLXToOpenSimMapper()
    names = mapper.get_marker_names()
    print(f"SMPL 고밀도 마커셋 구축 완료: 총 {len(names)}개 포인트")
