# -*- coding: utf-8 -*-
"""
발레 특화 관절 각도 계산 모듈 (순수 NumPy, OpenSim 불필요).

MediaPipe world_landmarks 3D 좌표에서 발레 핵심 바이오메카닉스 지표를 계산합니다.

지표 목록:
  - 무릎 굴곡 (좌/우): HIP → KNEE → ANKLE
  - 고관절 굴곡 (좌/우): SHOULDER → HIP → KNEE (시상면)
  - 턴아웃/외회전 (좌/우): 발 방향 vs 무릎 방향 (수평면)
  - 발목 저굴/배굴 (좌/우): KNEE → ANKLE → FOOT_INDEX
  - 척추 측굴: 어깨 중심 → 골반 중심 vs 수직축
  - 어깨 대칭성: 좌우 어깨 높이 차이 (정규화)

MediaPipe 좌표계: Y-DOWN
  → 3D 각도 계산 시 Y 부호는 일관성 유지 (절대값 사용하지 않음)
"""

import json
import math
from typing import Optional
import numpy as np


# ──── 발레 기준값 (이상적 각도) ────────────────────────────────────────────
BALLET_REFERENCE = {
    'left_knee_flexion':        {'ideal': 0.0,   'range': 10.0,  'desc': '무릎 굴곡 (휴식)'},
    'right_knee_flexion':       {'ideal': 0.0,   'range': 10.0,  'desc': '무릎 굴곡 (휴식)'},
    'left_hip_flexion':         {'ideal': 0.0,   'range': 15.0,  'desc': '고관절 굴곡'},
    'right_hip_flexion':        {'ideal': 0.0,   'range': 15.0,  'desc': '고관절 굴곡'},
    'left_turnout':             {'ideal': 90.0,  'range': 20.0,  'desc': '좌 턴아웃 (외회전)'},
    'right_turnout':            {'ideal': 90.0,  'range': 20.0,  'desc': '우 턴아웃 (외회전)'},
    'left_ankle_plantarflexion':{'ideal': 90.0,  'range': 15.0,  'desc': '좌 발목 저굴'},
    'right_ankle_plantarflexion':{'ideal': 90.0, 'range': 15.0,  'desc': '우 발목 저굴'},
    'spine_lateral':            {'ideal': 0.0,   'range': 5.0,   'desc': '척추 측굴'},
    'shoulder_symmetry':        {'ideal': 0.0,   'range': 0.05,  'desc': '어깨 대칭성 (정규화 거리)'},
}


def _vec(lm_dict: dict, name: str) -> Optional[np.ndarray]:
    """랜드마크 dict에서 3D 벡터를 반환합니다. 없으면 None."""
    lm = lm_dict.get(name)
    if lm is None:
        return None
    return np.array([lm['x'], lm['y'], lm['z']], dtype=float)


def calc_angle(p1: np.ndarray, vertex: np.ndarray, p2: np.ndarray) -> float:
    """
    3점으로 정의된 각도를 계산합니다 (단위: 도).

    Args:
        p1: 첫 번째 점
        vertex: 꼭짓점 (각도의 중심)
        p2: 두 번째 점

    Returns:
        0~180도 범위의 각도. 계산 불가 시 0.0 반환.
    """
    v1 = p1 - vertex
    v2 = p2 - vertex
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_a = np.dot(v1, v2) / (n1 * n2)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def calc_signed_angle_xz(v_ref: np.ndarray, v_target: np.ndarray) -> float:
    """
    수평면(XZ 평면)에서 v_ref → v_target 의 부호 있는 각도를 계산합니다 (단위: 도).
    턴아웃 계산에 사용합니다.
    """
    ref_xz = np.array([v_ref[0], v_ref[2]])
    tgt_xz = np.array([v_target[0], v_target[2]])
    n_ref = np.linalg.norm(ref_xz)
    n_tgt = np.linalg.norm(tgt_xz)
    if n_ref < 1e-8 or n_tgt < 1e-8:
        return 0.0
    ref_xz = ref_xz / n_ref
    tgt_xz = tgt_xz / n_tgt
    cos_a = np.clip(np.dot(ref_xz, tgt_xz), -1.0, 1.0)
    angle = math.degrees(math.acos(cos_a))
    return angle


def extract_ballet_angles(frame: dict) -> dict:
    """
    단일 프레임에서 발레 관절 각도 지표를 추출합니다.

    Args:
        frame: {'world_landmarks': {'LEFT_HIP': {'x': ..., 'y': ..., 'z': ...}, ...}}

    Returns:
        관절 각도 dict. 계산 불가한 항목은 None.
    """
    wl = frame.get('world_landmarks', {})
    if not wl:
        return {}

    def v(name):
        return _vec(wl, name)

    angles = {}

    # ── 무릎 굴곡 (Knee Flexion) ──────────────────────────────────────
    # HIP → KNEE → ANKLE, 완전 신전 = 180°, 굴곡 = 180° - angle
    for side, hip, knee, ankle in [
        ('left',  'LEFT_HIP',  'LEFT_KNEE',  'LEFT_ANKLE'),
        ('right', 'RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'),
    ]:
        ph, pk, pa = v(hip), v(knee), v(ankle)
        if ph is not None and pk is not None and pa is not None:
            raw = calc_angle(ph, pk, pa)
            angles[f'{side}_knee_flexion'] = max(0.0, 180.0 - raw)
        else:
            angles[f'{side}_knee_flexion'] = None

    # ── 고관절 굴곡 (Hip Flexion) ─────────────────────────────────────
    # SHOULDER → HIP → KNEE 각도 (몸통 vs 넓적다리 각도)
    for side, sho, hip, knee in [
        ('left',  'LEFT_SHOULDER',  'LEFT_HIP',  'LEFT_KNEE'),
        ('right', 'RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE'),
    ]:
        ps, ph, pk = v(sho), v(hip), v(knee)
        if ps is not None and ph is not None and pk is not None:
            raw = calc_angle(ps, ph, pk)
            angles[f'{side}_hip_flexion'] = max(0.0, 180.0 - raw)
        else:
            angles[f'{side}_hip_flexion'] = None

    # ── 턴아웃 / 외회전 (Turnout) ─────────────────────────────────────
    # 발 방향(발목→발끝)의 XZ 투영이 신체 외측 방향(hip axis 기반)과 이루는 각도
    # 이상적: 90° (발이 고관절 축과 평행 = 완전 외회전)
    # MediaPipe Y-DOWN: 수평면 = XZ 평면
    p_lhip, p_rhip = v('LEFT_HIP'), v('RIGHT_HIP')
    if p_lhip is not None and p_rhip is not None:
        # 고관절 축 XZ: LEFT_HIP → RIGHT_HIP (신체 우측 방향)
        hip_ax = np.array([p_rhip[0] - p_lhip[0], p_rhip[2] - p_lhip[2]])
        n_hip = np.linalg.norm(hip_ax)
        if n_hip > 1e-8:
            hip_ax = hip_ax / n_hip  # 정규화된 우측 방향
            # LEFT 외측 = 좌측 = 우측 반대 방향, RIGHT 외측 = 우측 방향
            for side, ankle_name, foot_name, sign in [
                ('left',  'LEFT_ANKLE',  'LEFT_FOOT_INDEX',  -1.0),  # 외측 = -hip_ax
                ('right', 'RIGHT_ANKLE', 'RIGHT_FOOT_INDEX',  1.0),  # 외측 = +hip_ax
            ]:
                pa, pf = v(ankle_name), v(foot_name)
                if pa is not None and pf is not None:
                    foot_xz = np.array([pf[0] - pa[0], pf[2] - pa[2]])
                    n_foot = np.linalg.norm(foot_xz)
                    if n_foot > 1e-8:
                        foot_xz = foot_xz / n_foot
                        outward = sign * hip_ax  # 해당 발의 외측 방향
                        cos_a = np.clip(np.dot(foot_xz, outward), -1.0, 1.0)
                        angle_from_outward = math.degrees(math.acos(cos_a))
                        # 턴아웃 = 90° - 외측방향과의 각도
                        # (외측과 평행 = 0° → 완전 턴아웃 = 90°)
                        angles[f'{side}_turnout'] = max(0.0, 90.0 - angle_from_outward)
                    else:
                        angles[f'{side}_turnout'] = None
                else:
                    angles[f'{side}_turnout'] = None
        else:
            angles['left_turnout'] = None
            angles['right_turnout'] = None
    else:
        angles['left_turnout'] = None
        angles['right_turnout'] = None

    # ── 발목 저굴 (Ankle Plantarflexion) ─────────────────────────────
    # KNEE → ANKLE → FOOT_INDEX 각도
    # 90° = 중립, >90° = 저굴(포인트), <90° = 배굴
    for side, knee, ankle, foot in [
        ('left',  'LEFT_KNEE',  'LEFT_ANKLE',  'LEFT_FOOT_INDEX'),
        ('right', 'RIGHT_KNEE', 'RIGHT_ANKLE', 'RIGHT_FOOT_INDEX'),
    ]:
        pk, pa, pf = v(knee), v(ankle), v(foot)
        if pk is not None and pa is not None and pf is not None:
            angles[f'{side}_ankle_plantarflexion'] = calc_angle(pk, pa, pf)
        else:
            angles[f'{side}_ankle_plantarflexion'] = None

    # ── 척추 측굴 (Spine Lateral Flexion) ────────────────────────────
    # 어깨 중심 → 골반 중심 vs 수직축(Y 방향) 각도
    lsho, rsho = v('LEFT_SHOULDER'), v('RIGHT_SHOULDER')
    lhip, rhip = v('LEFT_HIP'), v('RIGHT_HIP')
    if all(p is not None for p in [lsho, rsho, lhip, rhip]):
        shoulder_mid = (lsho + rsho) / 2.0
        pelvis_mid   = (lhip + rhip) / 2.0
        spine_vec = shoulder_mid - pelvis_mid
        vertical = np.array([0.0, -1.0, 0.0])  # MediaPipe Y-DOWN → 위쪽 = -Y
        spine_angle = calc_angle(
            pelvis_mid + vertical,
            pelvis_mid,
            pelvis_mid + spine_vec
        )
        angles['spine_lateral'] = spine_angle
    else:
        angles['spine_lateral'] = None

    # ── 어깨 대칭성 (Shoulder Symmetry) ──────────────────────────────
    # 좌우 어깨 Y 높이 차이를 어깨너비로 정규화
    if lsho is not None and rsho is not None:
        shoulder_width = np.linalg.norm(lsho - rsho)
        if shoulder_width > 1e-6:
            dy = abs(lsho[1] - rsho[1])
            angles['shoulder_symmetry'] = dy / shoulder_width
        else:
            angles['shoulder_symmetry'] = 0.0
    else:
        angles['shoulder_symmetry'] = None

    return angles


def extract_sequence_angles(poses_json_path: str) -> list:
    """
    전체 포즈 시퀀스에서 프레임별 발레 각도를 추출합니다.

    Args:
        poses_json_path: extract_poses()가 생성한 JSON 파일 경로.

    Returns:
        [{'frame_idx': int, 'angles': dict}, ...] 리스트.
    """
    with open(poses_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    frames = data.get('frames', [])
    result = []
    for frame in frames:
        angles = extract_ballet_angles(frame)
        result.append({
            'frame_idx': frame.get('frame_idx', 0),
            'angles': angles,
        })
    return result


def compute_summary(sequence_angles: list) -> dict:
    """
    시퀀스 각도 결과의 요약 통계 및 발레 스코어를 계산합니다.

    Returns:
        {
          'mean_angles': {angle_name: float},
          'std_angles':  {angle_name: float},
          'ballet_scores': {
            'turnout_score': float,       # 0~1, 1이 이상적
            'knee_alignment_score': float,
            'spine_alignment_score': float,
          }
        }
    """
    if not sequence_angles:
        return {}

    # 각도명 수집
    angle_names = set()
    for item in sequence_angles:
        angle_names.update(item.get('angles', {}).keys())

    # 유효값만 수집
    values_by_name: dict[str, list] = {name: [] for name in angle_names}
    for item in sequence_angles:
        for name in angle_names:
            val = item['angles'].get(name)
            if val is not None:
                values_by_name[name].append(val)

    mean_angles = {}
    std_angles = {}
    for name, vals in values_by_name.items():
        if vals:
            mean_angles[name] = float(np.mean(vals))
            std_angles[name]  = float(np.std(vals))
        else:
            mean_angles[name] = None
            std_angles[name]  = None

    # ── 발레 스코어 계산 ──────────────────────────────────────────────
    def score_angle(mean_val, ideal, tolerance):
        """각도가 이상값에 가까울수록 1.0에 가까운 스코어."""
        if mean_val is None:
            return None
        deviation = abs(mean_val - ideal)
        return float(max(0.0, 1.0 - deviation / (tolerance * 2)))

    left_to  = mean_angles.get('left_turnout')
    right_to = mean_angles.get('right_turnout')
    turnout_vals = [v for v in [left_to, right_to] if v is not None]
    turnout_score = score_angle(
        np.mean(turnout_vals) if turnout_vals else None,
        ideal=90.0, tolerance=20.0
    )

    left_knee  = mean_angles.get('left_knee_flexion')
    right_knee = mean_angles.get('right_knee_flexion')
    knee_vals = [v for v in [left_knee, right_knee] if v is not None]
    # 무릎 굴곡은 동작 중 변동이 크므로 대칭성 기반 스코어
    if left_knee is not None and right_knee is not None:
        asymmetry = abs(left_knee - right_knee)
        knee_alignment_score = float(max(0.0, 1.0 - asymmetry / 20.0))
    else:
        knee_alignment_score = None

    spine_score = score_angle(
        mean_angles.get('spine_lateral'),
        ideal=0.0, tolerance=5.0
    )

    return {
        'mean_angles': mean_angles,
        'std_angles':  std_angles,
        'ballet_scores': {
            'turnout_score':        turnout_score,
            'knee_alignment_score': knee_alignment_score,
            'spine_alignment_score': spine_score,
        }
    }


def compare_angle_sequences(ref_angles: list, comp_angles: list) -> dict:
    """
    두 시퀀스의 관절 각도를 프레임별로 비교합니다.

    Args:
        ref_angles:  extract_sequence_angles() 결과 (reference)
        comp_angles: extract_sequence_angles() 결과 (compare)

    Returns:
        {
          'per_frame': [{'frame_idx': int, 'ref': dict, 'comp': dict, 'diff': dict}],
          'summary': {
            'mean_diff': {angle_name: float},
            'ref_scores': {...},
            'comp_scores': {...},
          }
        }
    """
    # frame_idx → angles 매핑
    ref_map  = {item['frame_idx']: item['angles'] for item in ref_angles}
    comp_map = {item['frame_idx']: item['angles'] for item in comp_angles}

    all_frames = sorted(set(ref_map.keys()) | set(comp_map.keys()))
    per_frame = []
    diff_by_name: dict[str, list] = {}

    for fidx in all_frames:
        ref_a  = ref_map.get(fidx,  {})
        comp_a = comp_map.get(fidx, {})
        diff_a = {}
        for name in set(list(ref_a.keys()) + list(comp_a.keys())):
            rv = ref_a.get(name)
            cv = comp_a.get(name)
            if rv is not None and cv is not None:
                d = cv - rv
                diff_a[name] = round(d, 4)
                diff_by_name.setdefault(name, []).append(abs(d))
            else:
                diff_a[name] = None

        per_frame.append({
            'frame_idx': fidx,
            'ref':  {k: (round(v, 4) if v is not None else None) for k, v in ref_a.items()},
            'comp': {k: (round(v, 4) if v is not None else None) for k, v in comp_a.items()},
            'diff': diff_a,
        })

    mean_diff = {
        name: float(np.mean(vals)) if vals else None
        for name, vals in diff_by_name.items()
    }

    ref_summary  = compute_summary(ref_angles)
    comp_summary = compute_summary(comp_angles)

    return {
        'per_frame': per_frame,
        'summary': {
            'mean_diff':   mean_diff,
            'ref_scores':  ref_summary.get('ballet_scores', {}),
            'comp_scores': comp_summary.get('ballet_scores', {}),
        }
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/e2e_output/reference_poses.json"
    seq = extract_sequence_angles(path)
    print(f"총 {len(seq)} 프레임 처리 완료")
    if seq:
        print("첫 번째 프레임 각도:")
        for k, v in seq[0]['angles'].items():
            print(f"  {k}: {v:.2f}°" if v is not None else f"  {k}: N/A")
        summary = compute_summary(seq)
        print("\n평균 각도:")
        for k, v in summary['mean_angles'].items():
            print(f"  {k}: {v:.2f}°" if v is not None else f"  {k}: N/A")
        print("\n발레 스코어:")
        for k, v in summary['ballet_scores'].items():
            print(f"  {k}: {v:.3f}" if v is not None else f"  {k}: N/A")
