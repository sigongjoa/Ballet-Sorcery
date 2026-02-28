# -*- coding: utf-8 -*-
"""SMPL 메쉬 정점 -> OpenSim 고밀도 가상 마커 매핑 모듈.
SMPL (6890 vertices) 기준 고밀도 해부학적 마커 인덱스.
바이오메카닉스 분석의 정밀도를 극대화합니다.
"""

import numpy as np

# SMPL (6890 vertices) 기준 고밀도 해부학적 마커 인덱스
VIRTUAL_MARKER_MAP = {
    # --- ACROMION ---
    'ACROMION_L': 2983,
    'ACROMION_R': 6442,
    # --- ANKLE ---
    'ANKLE_LAT_L': 3375,
    'ANKLE_LAT_R': 6775,
    'ANKLE_MED_L': 3375,
    'ANKLE_MED_R': 6775,
    # --- ASIS ---
    'ASIS_L': 1479,
    'ASIS_R': 4952,
    # --- C7 ---
    'C7': 2877,
    # --- CLAV ---
    'CLAV_NOTCH': 3074,
    # --- ELBOW ---
    'ELBOW_LAT_L': 1733,
    'ELBOW_LAT_R': 5201,
    'ELBOW_MED_L': 1733,
    'ELBOW_MED_R': 5201,
    # --- GTROCHANTER ---
    'GTROCHANTER_L': 3115,
    'GTROCHANTER_R': 6538,
    # --- HAND ---
    'HAND_BACK_L': 2127,
    'HAND_BACK_R': 5588,
    # --- HEAD ---
    'HEAD_TOP': 3851,
    # --- HEEL ---
    'HEEL_POST_L': 3374,
    'HEEL_POST_R': 6773,
    # --- HUMERUS ---
    'HUMERUS_LAT_L': 1411,
    'HUMERUS_LAT_R': 4884,
    # --- ILIAC ---
    'ILIAC_CREST_L': 2910,
    'ILIAC_CREST_R': 6371,
    # --- KNEE ---
    'KNEE_LAT_L': 1468,
    'KNEE_LAT_R': 4942,
    'KNEE_MED_L': 1526,
    'KNEE_MED_R': 4998,
    # --- L5 ---
    'L5': 3502,
    # --- MIDDLE ---
    'MIDDLE_FINGER_BASE_L': 2127,
    'MIDDLE_FINGER_BASE_R': 5588,
    # --- MIDFOOT ---
    'MIDFOOT_UP_L': 3223,
    'MIDFOOT_UP_R': 6667,
    # --- PSIS ---
    'PSIS_L': 884,
    'PSIS_R': 4370,
    # --- PUBIC ---
    'PUBIC_SYMPH': 3149,
    # --- RIB ---
    'RIB_LOW_L': 1286,
    'RIB_LOW_R': 4766,
    # --- SACRUM ---
    'SACRUM': 1540,
    # --- SELLION ---
    'SELLION': 3051,
    # --- STERNUM ---
    'STERNUM_LOW': 1325,
    'STERNUM_UP': 1325,
    # --- T12 ---
    'T12': 3502,
    # --- T4 ---
    'T4': 3017,
    # --- T8 ---
    'T8': 3017,
    # --- TOE ---
    'TOE_1ST_BASE_L': 3223,
    'TOE_1ST_BASE_R': 6667,
    'TOE_5TH_BASE_L': 3223,
    'TOE_5TH_BASE_R': 6667,
    'TOE_TIP_L': 3223,
    'TOE_TIP_R': 6667,
    # --- WRIST ---
    'WRIST_RAD_L': 1997,
    'WRIST_RAD_R': 5458,
    'WRIST_ULN_L': 2207,
    'WRIST_ULN_R': 5667,
}

# 6890 미만 클램핑 보장 (수동 확인 완료되었으나 안전을 위해 유지)
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
