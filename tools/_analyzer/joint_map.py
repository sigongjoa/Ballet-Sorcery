# -*- coding: utf-8 -*-
"""관절 체인 정의, DOF 범위, 표시명."""

# 관절 체인 (부모 → 자식 목록)
CHAIN = {
    'pelvis':    ['femur_r', 'femur_l', 'torso'],
    'femur_r':   ['tibia_r'],
    'tibia_r':   ['calcn_r'],
    'calcn_r':   [],
    'femur_l':   ['tibia_l'],
    'tibia_l':   ['calcn_l'],
    'calcn_l':   [],
    'torso':     ['humerus_r', 'humerus_l'],
    'humerus_r': ['radius_r'],
    'radius_r':  [],
    'humerus_l': ['radius_l'],
    'radius_l':  [],
}

# 모든 body 이름 목록
BODY_NAMES = list(CHAIN.keys())

# MediaPipe에서 계산되는 DOF와 범위 (단위: 라디안)
DOF_INFO = {
    'knee_angle_r':    {'range': (0.0, 2.7),    'unit': 'rad', 'body': 'tibia_r'},
    'knee_angle_l':    {'range': (0.0, 2.7),    'unit': 'rad', 'body': 'tibia_l'},
    'hip_flexion_r':   {'range': (-0.52, 2.09), 'unit': 'rad', 'body': 'femur_r'},
    'hip_flexion_l':   {'range': (-0.52, 2.09), 'unit': 'rad', 'body': 'femur_l'},
    'hip_adduction_r': {'range': (-0.87, 0.35), 'unit': 'rad', 'body': 'femur_r'},
    'hip_adduction_l': {'range': (-0.87, 0.35), 'unit': 'rad', 'body': 'femur_l'},
    'ankle_angle_r':   {'range': (-1.0, 0.6),   'unit': 'rad', 'body': 'calcn_r'},
    'ankle_angle_l':   {'range': (-1.0, 0.6),   'unit': 'rad', 'body': 'calcn_l'},
    'arm_flex_r':      {'range': (-0.5, 2.5),   'unit': 'rad', 'body': 'humerus_r'},
    'arm_flex_l':      {'range': (-0.5, 2.5),   'unit': 'rad', 'body': 'humerus_l'},
    'arm_add_r':       {'range': (-2.09, 1.22), 'unit': 'rad', 'body': 'humerus_r'},
    'arm_add_l':       {'range': (-2.09, 1.22), 'unit': 'rad', 'body': 'humerus_l'},
    'elbow_flex_r':    {'range': (0.0, 2.6),    'unit': 'rad', 'body': 'radius_r'},
    'elbow_flex_l':    {'range': (0.0, 2.6),    'unit': 'rad', 'body': 'radius_l'},
}

# 3D 렌더용 체인 세그먼트 (a → b)
FK_SEGMENTS = [
    ('pelvis', 'femur_r'),   ('femur_r', 'tibia_r'),  ('tibia_r', 'calcn_r'),
    ('pelvis', 'femur_l'),   ('femur_l', 'tibia_l'),  ('tibia_l', 'calcn_l'),
    ('pelvis', 'torso'),
    ('torso', 'humerus_r'),  ('humerus_r', 'radius_r'),
    ('torso', 'humerus_l'),  ('humerus_l', 'radius_l'),
]


def get_ancestors(body_name: str) -> list:
    """body_name의 부모 체인(루트 pelvis까지) 반환."""
    # 역방향 맵 빌드
    parent = {}
    for p, children in CHAIN.items():
        for c in children:
            parent[c] = p

    ancestors = []
    cur = body_name
    while cur in parent:
        cur = parent[cur]
        ancestors.append(cur)
    return ancestors  # [직접부모, ..., 'pelvis']


def get_chain_bodies(selected_body: str) -> list:
    """selected_body와 그 모든 자손 body 목록 반환 (selected_body 포함)."""
    result = []
    stack = [selected_body]
    while stack:
        cur = stack.pop()
        result.append(cur)
        for child in CHAIN.get(cur, []):
            stack.append(child)
    return result
