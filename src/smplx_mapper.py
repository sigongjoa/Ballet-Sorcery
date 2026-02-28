# -*- coding: utf-8 -*-
"""SMPL-X 메쉬 정점 -> OpenSim 고밀도 가상 마커 매핑 모듈.
SMPL-X의 10,475개 정점 중 50개 이상의 해부학적 지점들을 선정하여
바이오메카닉스 분석의 정밀도를 극대화합니다.
"""

import numpy as np

# SMPL-X (10475 vertices) 기준 고밀도 해부학적 마커 인덱스
# 발레 동작(포인, 턴아웃, 척추 정렬)에 최적화된 마커셋
VIRTUAL_MARKER_MAP = {
    # --- Head & Neck (4) ---
    'HEAD_TOP': 895, 'SELLION': 331, 'C7': 3470, 'CLAV_NOTCH': 3506,

    # --- Torso & Spine (8) ---
    'T12': 3050, 'T8': 3060, 'T4': 3070, 'L5': 3040,
    'STERNUM_UP': 3507, 'STERNUM_LOW': 3515,
    'RIB_L_LOW': 3100, 'RIB_R_LOW': 6539,

    # --- Pelvis (8) ---
    'ASIS_L': 3120, 'ASIS_R': 6559,
    'PSIS_L': 3125, 'PSIS_R': 6564,
    'ILIAC_CREST_L': 3130, 'ILIAC_CREST_R': 6569,
    'PUBIC_SYMPH': 3140, 'SACRUM': 3030,

    # --- Upper Limbs (좌우 각 8개 = 16) ---
    'ACROMION_L': 3220, 'ACROMION_R': 6659,
    'HUMERUS_LAT_L': 3250, 'HUMERUS_LAT_R': 6689,
    'ELBOW_LAT_L': 3310, 'ELBOW_LAT_R': 6749,
    'ELBOW_MED_L': 3315, 'ELBOW_MED_R': 6754,
    'WRIST_RAD_L': 3440, 'WRIST_RAD_R': 6879,
    'WRIST_ULN_L': 3445, 'WRIST_ULN_R': 6884,
    'HAND_BACK_L': 3450, 'HAND_BACK_R': 6889,
    'MIDDLE_FINGER_BASE_L': 2300, 'MIDDLE_FINGER_BASE_R': 5750,

    # --- Lower Limbs (좌우 각 10개 = 20) ---
    'GTROCHANTER_L': 3150, 'GTROCHANTER_R': 6589,
    'KNEE_LAT_L': 1050, 'KNEE_LAT_R': 4510,
    'KNEE_MED_L': 1055, 'KNEE_MED_R': 4515,
    'ANKLE_LAT_L': 1120, 'ANKLE_LAT_R': 4580,
    'ANKLE_MED_L': 1125, 'ANKLE_MED_R': 4585,
    'HEEL_POST_L': 8846, 'HEEL_POST_R': 9963,
    'MIDFOOT_UP_L': 8860, 'MIDFOOT_UP_R': 9977,
    'TOE_1ST_BASE_L': 8900, 'TOE_1ST_BASE_R': 10017,
    'TOE_5TH_BASE_L': 8910, 'TOE_5TH_BASE_R': 10027,
    'TOE_TIP_L': 8920, 'TOE_TIP_R': 10037,
}

class SMPLXToOpenSimMapper:
    def __init__(self, mapping=VIRTUAL_MARKER_MAP):
        self.mapping = mapping

    def extract_virtual_markers(self, vertices: np.ndarray) -> dict:
        """SMPL-X 메쉬 정점 배열(N, 3)에서 가상 마커 좌표를 추출함."""
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
    print(f"고밀도 마커셋 구축 완료: 총 {len(names)}개 포인트")
    # 대칭성 검증용 출력
    lefts = [n for n in names if n.endswith('_L')]
    rights = [n for n in names if n.endswith('_R')]
    print(f"  - 좌측 마커: {len(lefts)}개 / 우측 마커: {len(rights)}개")
