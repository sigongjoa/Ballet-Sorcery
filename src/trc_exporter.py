# -*- coding: utf-8 -*-
"""
MediaPipe world_landmarks JSON → OpenSim .trc 파일 변환기.

TRC (Track Row Column) 포맷은 OpenSim/Pose2Sim에서 마커 궤적 데이터를
저장하는 표준 포맷입니다. MediaPipe의 33개 랜드마크 중 Pose2Sim BlazePose 설정과
호환되는 25개 마커를 OpenSim 마커 이름으로 매핑하여 내보냅니다.

변경 사항:
- Pose2Sim BlazePose 25개 마커 이름 사용 (소문자 전체 이름)
- 가상 마커(PELV) 제거 (Pose2Sim에서 자동 생성 권장)
- Y축 반전 유지 (MediaPipe Y-DOWN → OpenSim Y-UP)

좌표계 변환:
  MediaPipe world_landmarks: Y-DOWN (positive = down)
  OpenSim TRC: Y-UP (positive = up)
  → Y축 부호 반전 적용 (y = -y)
"""

import json
import os
from typing import Optional, Tuple


# MediaPipe 랜드마크 이름 → OpenSim BlazePose 호환 마커 이름 매핑 (25개)
MEDIAPIPE_TO_TRC = {
    'NOSE':             'nose',
    'LEFT_EYE':         'left_eye',
    'RIGHT_EYE':        'right_eye',
    'LEFT_SHOULDER':    'left_shoulder',
    'RIGHT_SHOULDER':   'right_shoulder',
    'LEFT_ELBOW':       'left_elbow',
    'RIGHT_ELBOW':      'right_elbow',
    'LEFT_WRIST':       'left_wrist',
    'RIGHT_WRIST':      'right_wrist',
    'LEFT_HIP':         'left_hip',
    'RIGHT_HIP':        'right_hip',
    'LEFT_KNEE':        'left_knee',
    'RIGHT_KNEE':       'right_knee',
    'LEFT_ANKLE':       'left_ankle',
    'RIGHT_ANKLE':      'right_ankle',
    'LEFT_FOOT_INDEX':  'left_foot_index',
    'LEFT_HEEL':        'left_heel',
    'RIGHT_FOOT_INDEX': 'right_foot_index',
    'RIGHT_HEEL':       'right_heel',
    'RIGHT_THUMB':      'right_thumb',
    'RIGHT_PINKY':      'right_pinky',
    'RIGHT_INDEX':      'right_index',
    'LEFT_THUMB':       'left_thumb',
    'LEFT_PINKY':       'left_pinky',
    'LEFT_INDEX':       'left_index',
}

# 마커 순서 (25개)
MARKER_ORDER = list(MEDIAPIPE_TO_TRC.values())
NUM_MARKERS = len(MARKER_ORDER)


def _get_xyz(lm_dict: dict, name: str) -> Optional[tuple]:
    """랜드마크 dict에서 (x, y_flipped, z) 반환. Y축 반전(MediaPipe Y-DOWN → Y-UP)."""
    lm = lm_dict.get(name)
    if lm is None:
        return None
    return (lm['x'], -lm['y'], lm['z'])


def export_trc(poses_json_path: str, output_trc_path: str) -> Tuple[str, float]:
    """
    MediaPipe pose JSON → OpenSim .trc 파일 변환.
    발 마커들을 기준으로 Y축 오프셋을 적용하여 지면(y=0.01m)에 맞춥니다.

    Args:
        poses_json_path: extract_poses()가 생성한 JSON 파일 경로.
        output_trc_path: 출력 .trc 파일 경로.

    Returns:
        (생성된 .trc 파일의 절대 경로, 적용된 y_offset)
    """
    with open(poses_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    frames = data.get('frames', [])
    if not frames:
        raise ValueError(f"포즈 데이터가 없습니다: {poses_json_path}")

    metadata = data.get('metadata', {})
    fps = metadata.get('effective_fps', 30.0)
    if fps <= 0:
        fps = 30.0

    n_frames = len(frames)

    # 1. feet_on_floor Y 오프셋 계산
    foot_markers = ['LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
    all_y_values = []
    for frame in frames:
        wl = frame.get('world_landmarks', {})
        for fm in foot_markers:
            xyz = _get_xyz(wl, fm)
            if xyz:
                all_y_values.append(xyz[1])
    
    if all_y_values:
        min_y = min(all_y_values)
        # 발이 y=0.01m에 오도록 (min_y + y_offset = 0.01)
        y_offset = -(min_y - 0.01)
    else:
        y_offset = 0.0
    
    print(f"feet_on_floor 오프셋 적용: {y_offset:+.4f}m")

    os.makedirs(os.path.dirname(os.path.abspath(output_trc_path)), exist_ok=True)

    with open(output_trc_path, 'w', encoding='utf-8', newline='\r\n') as f:
        # ── 헤더 행 1: 파일 타입 ──
        trc_filename = os.path.basename(output_trc_path)
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{trc_filename}\n")

        # ── 헤더 행 2: 파라미터 이름 ──
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
                "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")

        # ── 헤더 행 3: 파라미터 값 ──
        f.write(f"{fps:.4f}\t{fps:.4f}\t{n_frames}\t{NUM_MARKERS}\tm\t"
                f"{fps:.4f}\t1\t{n_frames}\n")

        # ── 헤더 행 4: 마커 이름 (Frame#, Time, 마커명×3열씩) ──
        marker_header = "Frame#\tTime"
        for marker in MARKER_ORDER:
            marker_header += f"\t{marker}\t\t"
        f.write(marker_header + "\n")

        # ── 헤더 행 5: X1 Y1 Z1 컬럼명 ──
        coord_header = "\t"
        for i, _ in enumerate(MARKER_ORDER, start=1):
            coord_header += f"\tX{i}\tY{i}\tZ{i}"
        f.write(coord_header + "\n")

        # ── 빈 행 ──
        f.write("\n")

        # ── 데이터 행 ──
        for row_idx, frame in enumerate(frames, start=1):
            wl = frame.get('world_landmarks', {})
            time_sec = frame.get('timestamp_sec', (row_idx - 1) / fps)

            row = f"{row_idx}\t{time_sec:.4f}"

            # 25개 마커 데이터 기록
            for mp_name in MEDIAPIPE_TO_TRC:
                xyz = _get_xyz(wl, mp_name)
                if xyz:
                    # y_flipped + y_offset 적용
                    row += f"\t{xyz[0]:.6f}\t{xyz[1] + y_offset:.6f}\t{xyz[2]:.6f}"
                else:
                    # OpenSim TRC parser가 trailing empty field를 무시하므로
                    # 결측 마커는 0으로 채워 필드 수 일관성 유지
                    row += "\t0.000000\t0.000000\t0.000000"

            f.write(row + "\n")

    abs_path = os.path.abspath(output_trc_path)
    print(f"TRC 파일 생성 완료: {abs_path} ({n_frames} frames, {NUM_MARKERS} markers)")
    return abs_path, y_offset


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        path, offset = export_trc(sys.argv[1], sys.argv[2])
    else:
        # 기본 테스트
        path, offset = export_trc(
            "data/e2e_output/reference_poses.json",
            "data/e2e_output/reference_poses.trc"
        )
