# -*- coding: utf-8 -*-
"""
고밀도 TRC v2 생성기.
SMPL-X 가상 마커 56개 시퀀스를 OpenSim TRC v2 포맷으로 출력하는 모듈입니다.
발레 동작의 정밀 분석을 위해 56개의 해부학적 마커를 포함합니다.
"""

import os
from typing import List, Dict, Tuple
from src.smplx_mapper import VIRTUAL_MARKER_MAP

# 마커 순서 (56개)
MARKER_ORDER = list(VIRTUAL_MARKER_MAP.keys())
NUM_MARKERS = len(MARKER_ORDER)


def export_trc_v2(markers_sequence: List[Dict[str, List[float]]], output_trc_path: str, fps: float = 30.0) -> Tuple[str, float]:
    """
    SMPL-X 가상 마커 시퀀스를 OpenSim .trc 파일로 변환합니다.
    발 마커들을 기준으로 Y축 오프셋을 적용하여 지면(y=0.01m)에 맞춥니다.

    Args:
        markers_sequence: 프레임별 마커 딕셔너리 리스트. 각 항목은 {'마커명': [x, y, z], ...} (56개)
        output_trc_path: 출력 .trc 파일 경로.
        fps: 프레임레이트 (기본 30.0)

    Returns:
        (생성된 .trc 파일의 절대 경로, 적용된 y_offset)
    """
    n_frames = len(markers_sequence)
    if n_frames == 0:
        raise ValueError("내보낼 마커 시퀀스 데이터가 없습니다.")

    # 1. feet_on_floor Y 오프셋 계산
    # 기준 마커: HEEL_POST_L, HEEL_POST_R, ANKLE_LAT_L, ANKLE_LAT_R, TOE_TIP_L, TOE_TIP_R
    ground_markers = ['HEEL_POST_L', 'HEEL_POST_R', 'ANKLE_LAT_L', 'ANKLE_LAT_R', 'TOE_TIP_L', 'TOE_TIP_R']
    all_y_values = []
    for frame in markers_sequence:
        for gm in ground_markers:
            xyz = frame.get(gm)
            if xyz:
                # SMPL-X는 이미 Y-UP이므로 그대로 사용
                all_y_values.append(xyz[1])

    if all_y_values:
        min_y = min(all_y_values)
        # 발이 y=0.01m에 오도록 (min_y + y_offset = 0.01)
        y_offset = -(min_y - 0.01)
    else:
        y_offset = 0.0
    
    # 디렉토리 생성
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
        for i in range(1, NUM_MARKERS + 1):
            coord_header += f"\tX{i}\tY{i}\tZ{i}"
        f.write(coord_header + "\n")

        # ── 빈 행 ──
        f.write("\n")

        # ── 데이터 행 ──
        for row_idx, frame_markers in enumerate(markers_sequence, start=1):
            time_sec = (row_idx - 1) / fps
            row = f"{row_idx}\t{time_sec:.4f}"

            # 56개 마커 데이터 기록
            for m_name in MARKER_ORDER:
                xyz = frame_markers.get(m_name)
                if xyz:
                    # Y 오프셋 적용 (SMPL-X는 이미 Y-UP이므로 Y 반전 불필요)
                    row += f"\t{xyz[0]:.6f}\t{xyz[1] + y_offset:.6f}\t{xyz[2]:.6f}"
                else:
                    # 결측 마커는 0으로 채워 필드 수 일관성 유지
                    row += "\t0.000000\t0.000000\t0.000000"

            f.write(row + "\n")

    abs_path = os.path.abspath(output_trc_path)
    return abs_path, y_offset


if __name__ == "__main__":
    # 간단한 가상 데이터 테스트
    # MARKER_ORDER 순서대로 모든 마커 포함
    test_markers = {name: [0.0, 0.0, 0.0] for name in MARKER_ORDER}
    # 발 위치를 -0.05m로 설정하여 오프셋 확인
    test_markers['HEEL_POST_L'] = [0.0, -0.05, 0.0]
    
    test_sequence = [test_markers] * 5
    path, offset = export_trc_v2(test_sequence, "data/e2e_output/test_v2.trc", fps=30.0)
    print(f"테스트 완료: {path}")
    print(f"적용된 y_offset: {offset:.4f}m (기대값: 0.0600)")
